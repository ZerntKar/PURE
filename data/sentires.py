"""Adapter for trusted Sentires-Guide quadruple exports."""

from collections import defaultdict
import json
from pathlib import Path
import pickle
import re
import string
from typing import Dict, List, Set


def normalize_feature(value: str) -> str:
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


class SentiresGuideExtractor:
    """Access (feature, opinion, sentence, score) by review identity."""

    def __init__(self, records_path: str):
        path = Path(records_path)
        if path.suffix == ".json":
            records = json.loads(path.read_text(encoding="utf-8"))
        else:
            # The official guide writes pickle; only open trusted local output.
            with path.open("rb") as stream:
                records = pickle.load(stream)
        self.by_identity = {}
        self.by_text = {}
        for record in records:
            text = normalize_feature(record["text"])
            quadruples = record.get("sentence", [])
            user = record.get("user", record.get("user_id"))
            item = record.get("item", record.get("item_id"))
            if user is not None and item is not None:
                identity = (str(user), str(item), text)
                self.by_identity.setdefault(identity, []).extend(quadruples)
            self.by_text.setdefault(text, []).extend(quadruples)

    def quadruples(self, text: str, user=None, item=None):
        key = normalize_feature(text)
        if user is not None and item is not None:
            identity = (str(user), str(item), key)
            if identity not in self.by_identity:
                raise KeyError(f"Review is missing from Sentires-Guide output: {identity}")
            return self.by_identity[identity]
        if key not in self.by_text:
            raise KeyError("Text is missing from the supplied Sentires-Guide output")
        return self.by_text[key]

    def extract(self, text: str, user=None, item=None) -> Set[str]:
        return {
            normalize_feature(str(feature))
            for feature, _, _, _ in self.quadruples(text, user=user, item=item)
            if str(feature).strip()
        }

    def extract_positive_features(self, reviews: List) -> Set[str]:
        scores: Dict[str, List[float]] = defaultdict(list)
        for review in reviews:
            if isinstance(review, dict):
                quadruples = self.quadruples(
                    review["text"],
                    user=review.get("user_id", review.get("user")),
                    item=review.get("item_id", review.get("item")),
                )
            else:
                quadruples = self.quadruples(review)
            for feature, _, _, score in quadruples:
                if str(feature).strip():
                    scores[normalize_feature(str(feature))].append(float(score))
        return {
            feature for feature, values in scores.items()
            if sum(values) / len(values) > 0
        }
