import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


class FeatureNormalizer:
    def __init__(
        self,
        kg_vocab: Set[str],
        sent_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        merge_threshold: float = 0.85,
        device: str = "cuda",
    ):
        self.kg_vocab = kg_vocab
        self.merge_threshold = merge_threshold
        self.canonical_map: Dict[str, str] = {}

        self.sent_model = SentenceTransformer(sent_model_name, device=device)
        kg_list = sorted(kg_vocab)
        self.kg_names = kg_list
        self.kg_norms = {normalize_text(k): k for k in kg_list}

        if kg_list:
            self.kg_embeddings = self.sent_model.encode(
                kg_list, show_progress_bar=False, convert_to_numpy=True
            )
        else:
            self.kg_embeddings = np.array([])

        for k in kg_list:
            norm = normalize_text(k)
            self.canonical_map[norm] = k

    def normalize(self, feature: str) -> str:
        norm = normalize_text(feature)

        if norm in self.canonical_map:
            return self.canonical_map[norm]

        if norm in self.kg_norms:
            canonical = self.kg_norms[norm]
            self.canonical_map[norm] = canonical
            return canonical

        if len(self.kg_embeddings) > 0:
            feat_emb = self.sent_model.encode(
                [feature], show_progress_bar=False, convert_to_numpy=True
            )
            sims = np.dot(self.kg_embeddings, feat_emb.T).flatten()
            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self.merge_threshold:
                canonical = self.kg_names[best_idx]
                self.canonical_map[norm] = canonical
                return canonical

        self.canonical_map[norm] = norm
        return norm

    def normalize_set(self, features: Set[str]) -> Set[str]:
        return {self.normalize(f) for f in features}


class KGVocabExtractor:
    def __init__(
        self,
        entity_vocab: Set[str],
        max_ngram: int = 3,
        min_token_len: int = 2,
    ):
        self.vocab_normalized: Dict[str, str] = {}
        for ent in entity_vocab:
            norm = normalize_text(ent)
            tokens = norm.split()
            if len(tokens) >= min_token_len or len(norm) >= 4:
                self.vocab_normalized[norm] = ent

        self.sorted_vocab = sorted(
            self.vocab_normalized.keys(),
            key=lambda x: len(x.split()),
            reverse=True,
        )
        self.max_ngram = max_ngram

    def extract(self, text: str) -> Set[str]:
        norm_text = normalize_text(text)
        found = set()
        remaining = norm_text
        for vocab_item in self.sorted_vocab:
            if vocab_item in remaining:
                original = self.vocab_normalized[vocab_item]
                found.add(original)
                remaining = remaining.replace(vocab_item, " " * len(vocab_item))
        return found

    def extract_with_positions(self, text: str) -> List[Tuple[str, int, int]]:
        norm_text = normalize_text(text)
        results = []
        for vocab_item in self.sorted_vocab:
            start = 0
            while True:
                pos = norm_text.find(vocab_item, start)
                if pos == -1:
                    break
                original = self.vocab_normalized[vocab_item]
                results.append((original, pos, pos + len(vocab_item)))
                start = pos + 1
        results.sort(key=lambda x: x[1])
        return results


class NLPPhraseExtractor:
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        use_transformer_sentiment: bool = True,
        device: str = "cuda",
    ):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)

        self.use_transformer_sentiment = use_transformer_sentiment
        if use_transformer_sentiment:
            try:
                self.sentiment_pipe = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if device == "cuda" else -1,
                    truncation=True,
                    max_length=512,
                )
            except Exception:
                self.use_transformer_sentiment = False
                self.sentiment_pipe = None
        else:
            self.sentiment_pipe = None

        self.stopfeatures = {
            "it", "this", "that", "movie", "book", "film", "show",
            "story", "thing", "something", "everything", "nothing",
            "place", "restaurant", "time", "way", "lot", "bit",
        }

    def extract(self, text: str) -> Set[str]:
        doc = self.nlp(text)
        features = set()

        for chunk in doc.noun_chunks:
            phrase = chunk.text.lower().strip()
            core = chunk.root.text.lower()
            if core not in self.stopfeatures and len(phrase) >= 3:
                features.add(phrase)

        for token in doc:
            if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                head_text = token.head.text.lower()
                if head_text not in self.stopfeatures:
                    phrase = f"{token.text} {token.head.text}".lower()
                    features.add(phrase)

        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"}:
                features.add(ent.text.lower())

        return features

    def extract_with_sentiment(self, text: str) -> List[Tuple[str, str, str, float]]:
        doc = self.nlp(text)
        results = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_doc = sent.as_doc() if hasattr(sent, "as_doc") else self.nlp(sent_text)

            for chunk in sent_doc.noun_chunks:
                feature = chunk.text.lower().strip()
                core = chunk.root.text.lower()
                if core in self.stopfeatures or len(feature) < 3:
                    continue

                opinions = []
                for token in chunk:
                    for child in token.children:
                        if child.pos_ == "ADJ":
                            opinions.append(child.text.lower())
                for child in chunk.root.children:
                    if child.pos_ == "ADJ" and child.text.lower() not in [
                        t.text.lower() for t in chunk
                    ]:
                        opinions.append(child.text.lower())

                opinion = opinions[0] if opinions else ""
                score = self._compute_sentiment(sent_text, feature)
                results.append((feature, opinion, sent_text, score))

        return results

    def _compute_sentiment(self, sentence: str, feature: str) -> float:
        if self.use_transformer_sentiment and self.sentiment_pipe is not None:
            try:
                result = self.sentiment_pipe(sentence[:512])[0]
                label = result["label"]
                conf = result["score"]
                return conf if label == "POSITIVE" else -conf
            except Exception:
                pass
        return self._rule_based_sentiment(sentence)

    @staticmethod
    def _rule_based_sentiment(text: str) -> float:
        text_lower = text.lower()
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "beautiful",
            "inspiring", "heartwarming", "powerful", "compelling", "brilliant",
            "outstanding", "touching", "enjoyable", "fantastic", "superb",
            "delightful", "charming", "captivating", "uplifting", "refreshing",
            "satisfying", "impressive", "engaging", "lovely", "pleasant",
            "comforting", "healing", "warm", "cozy", "fun", "hilarious",
            "recommend", "love", "loved", "favorite", "best", "perfect",
        }
        negative_words = {
            "bad", "terrible", "awful", "horrible", "poor", "boring",
            "disappointing", "violent", "dark", "grim", "depressing",
            "confusing", "mediocre", "bland", "overrated", "annoying",
            "frustrating", "dull", "weak", "slow", "waste", "worst",
            "hate", "hated", "dislike", "avoid", "unpleasant", "cold",
        }
        negation_words = {
            "not", "no", "never", "neither", "hardly", "barely",
            "dont", "doesnt", "didnt",
        }

        tokens = text_lower.split()
        pos_count = 0
        neg_count = 0
        negate = False

        for i, token in enumerate(tokens):
            clean = token.strip(string.punctuation)
            if clean in negation_words:
                negate = True
                continue
            if clean in positive_words:
                neg_count += 1 if negate else 0
                pos_count += 0 if negate else 1
                negate = False
            elif clean in negative_words:
                pos_count += 1 if negate else 0
                neg_count += 0 if negate else 1
                negate = False
            elif i > 0:
                negate = False

        total = pos_count + neg_count
        return 0.0 if total == 0 else (pos_count - neg_count) / total


class SimpleFeatureExtractor:
    def __init__(
        self,
        entity_vocab: Set[str],
        normalize_features: bool = True,
        merge_threshold: float = 0.85,
        device: str = "cuda",
    ):
        self.kg_extractor = KGVocabExtractor(entity_vocab)

        self.normalizer = (
            FeatureNormalizer(
                kg_vocab=entity_vocab,
                merge_threshold=merge_threshold,
                device=device,
            )
            if normalize_features
            else None
        )

    def extract(self, text: str) -> Set[str]:
        feats = self.kg_extractor.extract(text)
        if self.normalizer is not None:
            feats = self.normalizer.normalize_set(feats)
        return feats

    def batch_extract(self, texts: List[str]) -> List[Set[str]]:
        return [self.extract(t) for t in texts]


class FeatureExtractor:
    def __init__(
        self,
        entity_vocab: Optional[Set[str]] = None,
        use_nlp: bool = True,
        use_sentires: bool = False,
        sentires_path: Optional[str] = None,
        sentiment_threshold: float = 0.0,
        normalize_features: bool = True,
        merge_threshold: float = 0.85,
        device: str = "cuda",
    ):
        self.extractors = []
        self.sentiment_threshold = sentiment_threshold

        if entity_vocab:
            self.kg_extractor = KGVocabExtractor(entity_vocab)
            self.extractors.append(("kg", self.kg_extractor))
        else:
            self.kg_extractor = None

        if use_nlp:
            self.nlp_extractor = NLPPhraseExtractor(device=device)
            self.extractors.append(("nlp", self.nlp_extractor))
        else:
            self.nlp_extractor = None

        self.normalize_features = normalize_features
        if normalize_features and entity_vocab:
            self.normalizer = FeatureNormalizer(
                kg_vocab=entity_vocab,
                merge_threshold=merge_threshold,
                device=device,
            )
        else:
            self.normalizer = None

    def extract(self, text: str) -> Set[str]:
        all_features: Set[str] = set()
        for name, extractor in self.extractors:
            feats = extractor.extract(text)
            all_features.update(feats)

        if self.normalizer is not None:
            all_features = self.normalizer.normalize_set(all_features)

        return all_features

    def extract_positive_features(
        self,
        texts: List[str],
        aggregation: str = "mean",
    ) -> Set[str]:
        feature_scores: Dict[str, List[float]] = defaultdict(list)

        for text in texts:
            if self.nlp_extractor is not None:
                quadruples = self.nlp_extractor.extract_with_sentiment(text)
                for feat, op, sent, score in quadruples:
                    if score > 0:
                        norm_feat = (
                            self.normalizer.normalize(feat)
                            if self.normalizer
                            else normalize_text(feat)
                        )
                        feature_scores[norm_feat].append(score)
            else:
                for feat in self.extract(text):
                    feature_scores[feat].append(1.0)

        positive_features: Set[str] = set()
        for feat, scores in feature_scores.items():
            if aggregation == "mean":
                agg_score = sum(scores) / len(scores)
            elif aggregation == "sum":
                agg_score = sum(scores)
            elif aggregation == "max":
                agg_score = max(scores)
            else:
                agg_score = sum(scores) / len(scores)

            if agg_score > self.sentiment_threshold:
                positive_features.add(feat)

        return positive_features

    def build_item_feature_set(
        self,
        item_description: str,
        item_kg_features: Optional[Set[str]] = None,
    ) -> Set[str]:
        text_features = self.extract(item_description)
        if item_kg_features:
            kg_normalized = (
                self.normalizer.normalize_set(item_kg_features)
                if self.normalizer is not None
                else item_kg_features
            )
            return text_features | kg_normalized
        return text_features

    def batch_extract(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[Set[str]]:
        iterator = texts
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Extracting features")
        return [self.extract(t) for t in iterator]

    def batch_extract_positive(
        self,
        user_review_lists: List[List[str]],
        show_progress: bool = False,
    ) -> List[Set[str]]:
        iterator = user_review_lists
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(user_review_lists, desc="Extracting user preferences")
        return [self.extract_positive_features(reviews) for reviews in iterator]
