"""Dataset wrappers for PURE training and inference."""

from typing import Dict, List

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from data.kg import KnowledgeGraph


class RecommendationDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        kg: KnowledgeGraph,
        tokenizer: AutoTokenizer,
        max_history: int = 10,
        max_explanation_len: int = 128,
        split: str = "train",
    ):
        self.data = data
        self.kg = kg
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.max_explanation_len = max_explanation_len
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]

        user_pos_features = sample.get("user_positive_features", [])

        return {
            "user_id":           sample["user_id"],
            "history":           sample["history"][-self.max_history:],
            "target_item":       sample["target_item"],
            "explanation_text":  sample.get("explanation", ""),
            "item_features":     sample.get("item_features", []),
            "user_pos_features": user_pos_features,
            "negative_items":    sample.get("negative_items", []),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "user_ids":         [b["user_id"]           for b in batch],
        "histories":        [b["history"]            for b in batch],
        "target_items":     [b["target_item"]        for b in batch],
        "explanation_texts":[b["explanation_text"]   for b in batch],
        "item_features":    [b["item_features"]      for b in batch],
        "user_pos_features":[b["user_pos_features"]  for b in batch],
        "negative_items":   [b["negative_items"]     for b in batch],
    }
