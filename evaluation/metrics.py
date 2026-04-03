import math
import random
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from .feature_extractor import SimpleFeatureExtractor


def compute_f_ehr(
    generated_features: List[Set[str]],
    item_features: List[Set[str]],
) -> float:
    rates = []
    for gen_feats, item_feats in zip(generated_features, item_features):
        if len(gen_feats) == 0:
            continue
        hallucinated = gen_feats - item_feats
        rates.append(len(hallucinated) / len(gen_feats))
    return float(np.mean(rates)) if rates else 0.0


def compute_p_ehr(
    generated_features: List[Set[str]],
    user_pos_features: List[Set[str]],
    feature_embeddings: Dict[str, np.ndarray],
    user_intent_vectors: List[Optional[np.ndarray]],
    tau: float = 0.40,
) -> float:
    rates = []
    for gen_feats, pos_feats, h_u in zip(
        generated_features, user_pos_features, user_intent_vectors
    ):
        if len(gen_feats) == 0:
            continue
        if h_u is None:
            continue
        penalty_count = 0
        for f in gen_feats:
            if f in pos_feats:
                continue
            f_emb = feature_embeddings.get(f)
            if f_emb is not None:
                cos_sim = np.dot(f_emb, h_u) / (
                    np.linalg.norm(f_emb) * np.linalg.norm(h_u) + 1e-8
                )
                if cos_sim >= tau:
                    continue
            penalty_count += 1
        rates.append(penalty_count / len(gen_feats))
    return float(np.mean(rates)) if rates else 0.0


class PreferenceEHRCalculator:
    def __init__(
        self,
        sent_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        tau: float = 0.40,
    ):
        self.sent_model = SentenceTransformer(sent_model_name)
        self.tau = tau
        self._feature_emb_cache: Dict[str, np.ndarray] = {}

    def encode_features(self, features: List[str]) -> Dict[str, np.ndarray]:
        new_features = [f for f in features if f not in self._feature_emb_cache]
        if new_features:
            embs = self.sent_model.encode(new_features, show_progress_bar=False)
            for f, emb in zip(new_features, embs):
                self._feature_emb_cache[f] = emb
        return {
            f: self._feature_emb_cache[f]
            for f in features
            if f in self._feature_emb_cache
        }

    def compute_user_intent_vector(
        self, user_pos_features: Set[str]
    ) -> Optional[np.ndarray]:
        if not user_pos_features:
            return None
        embs = self.encode_features(list(user_pos_features))
        if not embs:
            return None
        return np.mean(list(embs.values()), axis=0)

    def compute(
        self,
        generated_features: List[Set[str]],
        user_pos_features_list: List[Set[str]],
    ) -> float:
        all_features: Set[str] = set()
        for gf in generated_features:
            all_features.update(gf)
        for upf in user_pos_features_list:
            all_features.update(upf)
        feature_embs = self.encode_features(list(all_features))

        user_intents = [
            self.compute_user_intent_vector(upf)
            for upf in user_pos_features_list
        ]

        return compute_p_ehr(
            generated_features=generated_features,
            user_pos_features=user_pos_features_list,
            feature_embeddings=feature_embs,
            user_intent_vectors=user_intents,
            tau=self.tau,
        )


def compute_bleu4(predictions: List[str], references: List[str]) -> float:
    smooth = SmoothingFunction().method1
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if len(pred_tokens) == 0:
            scores.append(0.0)
            continue
        scores.append(
            sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth,
            )
        )
    return float(np.mean(scores))


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return float(np.mean(scores)) * 100


def compute_fmr(
    generated_features: List[Set[str]],
    item_features: List[Set[str]],
) -> float:
    rates = []
    for gen_feats, item_feats in zip(generated_features, item_features):
        if len(item_feats) == 0:
            continue
        matched = gen_feats & item_feats
        rates.append(len(matched) / len(item_feats))
    return float(np.mean(rates)) if rates else 0.0


def compute_fcr(
    generated_features: List[Set[str]],
    item_features: List[Set[str]],
) -> float:
    rates = []
    for gen_feats, item_feats in zip(generated_features, item_features):
        if len(item_feats) == 0:
            continue
        rates.append(len(gen_feats & item_feats) / len(item_feats))
    return float(np.mean(rates)) if rates else 0.0


def compute_diversity(
    predictions: List[str],
    max_pairs: int = 1000,
) -> float:
    if len(predictions) < 2:
        return 0.0
    smooth = SmoothingFunction().method1
    all_pairs = [
        (i, j)
        for i in range(len(predictions))
        for j in range(i + 1, len(predictions))
    ]
    sampled = random.sample(all_pairs, min(max_pairs, len(all_pairs)))
    scores = []
    for i, j in sampled:
        tokens_i = predictions[i].split()
        tokens_j = predictions[j].split()
        if tokens_i and tokens_j:
            scores.append(
                sentence_bleu(
                    [tokens_i],
                    tokens_j,
                    weights=(0.5, 0.5, 0, 0),
                    smoothing_function=smooth,
                )
            )
    return float(np.mean(scores)) if scores else 0.0


def compute_hr_at_k(
    predictions: List[List[int]],
    targets: List[int],
    k: int = 5,
) -> float:
    hits = sum(1 for pred_list, target in zip(predictions, targets) if target in pred_list[:k])
    return hits / len(targets) if targets else 0.0


def compute_ndcg_at_k(
    predictions: List[List[int]],
    targets: List[int],
    k: int = 5,
) -> float:
    ndcgs = []
    for pred_list, target in zip(predictions, targets):
        top_k = pred_list[:k]
        if target in top_k:
            rank = top_k.index(target)
            ndcgs.append(1.0 / math.log2(rank + 2))
        else:
            ndcgs.append(0.0)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


class PUREEvaluator:
    def __init__(
        self,
        feature_vocab: Set[str],
        tau: float = 0.40,
        eval_k: int = 5,
    ):
        self.feature_extractor = SimpleFeatureExtractor(feature_vocab)
        self.p_ehr_calculator = PreferenceEHRCalculator(tau=tau)
        self.eval_k = eval_k

    def evaluate_explanations(
        self,
        predictions: List[str],
        references: List[str],
        item_features: List[Set[str]],
        user_pos_features: List[Set[str]],
    ) -> Dict[str, float]:
        gen_features = [self.feature_extractor.extract(p) for p in predictions]
        return {
            "F-EHR":   compute_f_ehr(gen_features, item_features),
            "P-EHR":   self.p_ehr_calculator.compute(gen_features, user_pos_features),
            "BLEU-4":  compute_bleu4(predictions, references),
            "ROUGE-L": compute_rouge_l(predictions, references),
            "FMR":     compute_fmr(gen_features, item_features),
            "FCR":     compute_fcr(gen_features, item_features),
            "DIV":     compute_diversity(predictions),
        }

    def evaluate_ranking(
        self,
        predictions: List[List[int]],
        targets: List[int],
    ) -> Dict[str, float]:
        return {
            f"HR@{self.eval_k}":   compute_hr_at_k(predictions, targets, self.eval_k),
            f"NDCG@{self.eval_k}": compute_ndcg_at_k(predictions, targets, self.eval_k),
        }
