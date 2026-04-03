from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import random

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class KGTriple:
    def __init__(self, head: int, relation: int, tail: int):
        self.head = head
        self.relation = relation
        self.tail = tail


class KnowledgeGraph:
    def __init__(self, entity2id: Dict, relation2id: Dict, triples: List[KGTriple]):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.triples = triples
        self.n_entities = len(entity2id)
        self.n_relations = len(relation2id)

        self.adj: Dict[int, List[Tuple[int, int]]] = {}
        for t in triples:
            self.adj.setdefault(t.head, []).append((t.relation, t.tail))
            self.adj.setdefault(t.tail, []).append((t.relation, t.head))

        self.degree = {node: len(neighbors) for node, neighbors in self.adj.items()}

    def get_neighbors(self, node: int) -> List[Tuple[int, int]]:
        return self.adj.get(node, [])

    def get_degree(self, node: int) -> int:
        return self.degree.get(node, 0)

    def multi_hop_paths(
        self,
        src: int,
        dst: int,
        max_hop: int = 3,
        max_paths: int = 200,
        max_neighbors: int = 50,
    ) -> List[Dict]:
        paths = []

        def dfs(node: int, path: List[Tuple[int, int, int]], visited: set):
            if len(paths) >= max_paths:
                return

            if len(path) > 0 and node == dst:
                paths.append({
                    "path": path[:],
                    "hop": len(path),
                })
                return

            if len(path) >= max_hop:
                return

            neighbors = self.get_neighbors(node)
            if len(neighbors) > max_neighbors:
                neighbors = random.sample(neighbors, max_neighbors)

            for rel, neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append((node, rel, neighbor))
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.discard(neighbor)

        dfs(src, [], {src})
        return paths

    @classmethod
    def from_files(cls, entity_file: str, relation_file: str, triple_file: str):
        with open(entity_file) as f:
            entity2id = json.load(f)
        with open(relation_file) as f:
            relation2id = json.load(f)
        with open(triple_file) as f:
            raw_triples = json.load(f)
        triples = [KGTriple(t[0], t[1], t[2]) for t in raw_triples]
        return cls(entity2id, relation2id, triples)


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

        user_pos_features = sample.get("user_positive_features", None)
        if user_pos_features is None:
            raise KeyError(
                f"Sample {idx} (user_id={sample.get('user_id')}) is missing "
                f"'user_positive_features'. Check your data JSON field name."
            )

        return {
            "user_id":           sample["user_id"],
            "history":           sample["history"][-self.max_history:],
            "target_item":       sample["target_item"],
            "explanation_text":  sample.get("explanation", ""),
            "item_features":     sample.get("item_features", []),
            "user_pos_features": user_pos_features,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "user_ids":         [b["user_id"]           for b in batch],
        "histories":        [b["history"]            for b in batch],
        "target_items":     [b["target_item"]        for b in batch],
        "explanation_texts":[b["explanation_text"]   for b in batch],
        "item_features":    [b["item_features"]      for b in batch],
        "user_pos_features":[b["user_pos_features"]  for b in batch],
    }
