"""Offline path indexing and preference-aware evidence selection."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

KGPath = List[Tuple[int, int, int]]


class TargetAwareUserIntent(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5

    def forward(self, target_emb, history_embs, history_mask=None):
        if history_embs.size(1) == 0:
            return self.W_V(target_emb)
        query = self.W_Q(target_emb).unsqueeze(1)
        keys = self.W_K(history_embs)
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) * self.scale
        if history_mask is not None:
            scores = scores.masked_fill(~history_mask.bool(), torch.finfo(scores.dtype).min)
        alpha = F.softmax(scores, dim=-1)
        values = self.W_V(history_embs)
        return torch.bmm(alpha.unsqueeze(1), values).squeeze(1)


class PathEncoder(nn.Module):
    """Frozen BERT encoder for linearized reasoning paths."""

    def __init__(self, model_name: str = "bert-large-uncased"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @staticmethod
    def linearize_path(path, id2entity, id2relation):
        return " | ".join(
            f"{id2entity.get(h, str(h))} -[{id2relation.get(r, str(r))}]-> "
            f"{id2entity.get(t, str(t))}"
            for h, r, t in path
        )

    @torch.no_grad()
    def encode_paths(self, path_texts: List[str]) -> torch.Tensor:
        if not path_texts:
            return torch.empty(0, self.hidden_size, device=self.device)
        enc = self.tokenizer(
            path_texts, padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        ).to(self.device)
        out = self.encoder(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)


class PathIndex:
    """Persist graph path vectors for search and BERT vectors for scoring."""

    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.paths: List[KGPath] = []
        self.targets: List[int] = []
        self.starts: List[int] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.search_embeddings: Optional[torch.Tensor] = None
        self.metadata: Dict = {}
        self.by_target: Dict[int, List[int]] = {}

    def build(
        self, kg, samples, encoder: PathEncoder, node_embeddings,
        id2entity, id2relation,
        max_hop=3, max_paths_per_target=200, max_neighbors=50, batch_size=32,
    ):
        if encoder.hidden_size != node_embeddings.shape[-1]:
            raise ValueError(
                "BERT path vectors and RGAT node vectors must share a dimension"
            )
        targets = {
            int(candidate)
            for sample in samples
            for candidate in [sample["target_item"], *sample.get("negative_items", [])]
        }
        unique = set()
        for target in sorted(targets):
            for record in kg.paths_around_target(
                target, max_hop=max_hop,
                max_paths=max_paths_per_target, max_neighbors=max_neighbors,
            ):
                path = tuple(tuple(edge) for edge in record["path"])
                if path:
                    unique.add((target, path[0][0], path))
        ordered = sorted(unique)
        self.targets = [target for target, _, _ in ordered]
        self.starts = [source for _, source, _ in ordered]
        self.paths = [list(path) for _, _, path in ordered]
        graph_vectors = node_embeddings.detach().float().cpu()
        search_vectors = [
            graph_vectors[sorted({node for h, _, t in path for node in (h, t)})].mean(0)
            for path in self.paths
        ]
        self.search_embeddings = (
            F.normalize(torch.stack(search_vectors), dim=-1)
            if search_vectors else torch.empty(0, graph_vectors.shape[-1])
        )
        vectors = []
        for start in range(0, len(self.paths), batch_size):
            texts = [
                encoder.linearize_path(path, id2entity, id2relation)
                for path in self.paths[start:start + batch_size]
            ]
            vectors.append(encoder.encode_paths(texts).float().cpu())
        self.embeddings = (
            F.normalize(torch.cat(vectors), dim=-1)
            if vectors else torch.empty(0, encoder.hidden_size)
        )
        self.metadata = {
            "version": 2,
            "graph_dim": graph_vectors.shape[-1],
            "path_plm_name": encoder.model_name,
            "max_hop": max_hop,
            "max_paths_per_target": max_paths_per_target,
            "max_neighbors": max_neighbors,
            "indexed_targets": sorted(targets),
        }
        self._make_lookup()
        self.save()
        return self

    def _make_lookup(self):
        self.by_target = {}
        for index, target in enumerate(self.targets):
            self.by_target.setdefault(target, []).append(index)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "paths": self.paths, "targets": self.targets,
                "starts": self.starts, "embeddings": self.embeddings,
                "search_embeddings": self.search_embeddings,
                "metadata": self.metadata,
            },
            self.index_path,
        )

    def load(self):
        data = torch.load(self.index_path, map_location="cpu", weights_only=True)
        self.paths = data["paths"]
        self.targets = data["targets"]
        self.starts = data["starts"]
        self.embeddings = data["embeddings"]
        self.search_embeddings = data["search_embeddings"]
        self.metadata = data["metadata"]
        if len(self.search_embeddings) != len(self.paths) or len(self.embeddings) != len(self.paths):
            raise ValueError("Path index vectors and paths have different lengths")
        self._make_lookup()
        return self

    def candidates(self, target: int, history: List[int]):
        # The paper indexes paths around the target. History conditions their
        # scores through the intent vector, but does not restrict this pool.
        indices = self.by_target.get(target, [])
        if not indices:
            return (
                [], torch.empty(0, self.search_embeddings.shape[-1]),
                torch.empty(0, self.embeddings.shape[-1]),
            )
        return (
            [self.paths[index] for index in indices],
            self.search_embeddings[indices], self.embeddings[indices],
        )


class PreferenceAwarePathRetrieval(nn.Module):
    def __init__(
        self, embed_dim, specificity_scorer, top_n=5, mmr_gamma=0.6,
        candidate_pool=40, score_threshold=0.0,
    ):
        super().__init__()
        self.intent_model = TargetAwareUserIntent(embed_dim)
        self.specificity = specificity_scorer
        self.top_n = top_n
        self.mmr_gamma = mmr_gamma
        self.candidate_pool = candidate_pool
        self.score_threshold = score_threshold

    def score_path(self, path, path_emb, user_intent, node_embeddings, node_degrees, adj):
        semantic = F.cosine_similarity(user_intent, path_emb, dim=0)
        nodes = sorted({node for edge in path for node in (edge[0], edge[2])})
        node_ids = torch.tensor(nodes, dtype=torch.long, device=node_embeddings.device)
        degrees = torch.tensor(
            [node_degrees.get(node, 0) for node in nodes],
            dtype=torch.float32, device=node_embeddings.device,
        )
        specificity = self.specificity.compute(
            node_ids=nodes, node_embeddings=node_embeddings[node_ids],
            degrees=degrees, adj=adj, user_intent=user_intent,
        )
        return semantic * specificity.mean()

    def mmr_rerank(self, candidates):
        selected = []
        remaining = list(candidates)
        while remaining and len(selected) < self.top_n:
            best_index = None
            best_value = float("-inf")
            for index, (path, score, embedding) in enumerate(remaining):
                redundancy = max(
                    (
                        F.cosine_similarity(embedding, chosen[2], dim=0).item()
                        for chosen in selected
                    ),
                    default=0.0,
                )
                value = self.mmr_gamma * score - (1 - self.mmr_gamma) * redundancy
                if value > best_value:
                    best_index, best_value = index, value
            selected.append(remaining.pop(best_index))
        return selected

    def retrieve(
        self, target_emb, history_embs, target_item, history, path_index,
        node_embeddings, node_degrees, adj, history_mask=None,
    ):
        user_intent = self.intent_model(
            target_emb.unsqueeze(0), history_embs.unsqueeze(0),
            history_mask=history_mask,
        ).squeeze(0)
        paths, search_embeddings, text_embeddings = path_index.candidates(target_item, history)
        if not paths:
            return [], user_intent
        if search_embeddings.size(-1) != user_intent.numel() or text_embeddings.size(-1) != user_intent.numel():
            raise ValueError(
                "Path index dimensions do not match "
                f"graph intent dimension {user_intent.numel()}"
            )
        search_embeddings = search_embeddings.to(user_intent.device)
        text_embeddings = text_embeddings.to(user_intent.device)
        approximate = F.cosine_similarity(
            search_embeddings, user_intent.unsqueeze(0), dim=-1,
        )
        candidate_indices = torch.topk(
            approximate, k=min(self.candidate_pool, len(paths)),
        ).indices.tolist()
        scored = []
        for index in candidate_indices:
            score = self.score_path(
                paths[index], text_embeddings[index], user_intent,
                node_embeddings, node_degrees, adj,
            ).item()
            if score >= self.score_threshold:
                scored.append((paths[index], score, text_embeddings[index]))
        chosen = self.mmr_rerank(scored)
        return [path for path, _, _ in chosen], user_intent

    @torch.no_grad()
    def score_item(
        self, target_emb, history_embs, target_item, history, path_index,
        node_embeddings, node_degrees, adj,
    ) -> float:
        """Evidence score for a sampled recommendation candidate.

        An item's strongest preference-aligned path supplies its ranking score.
        """
        intent = self.intent_model(
            target_emb.unsqueeze(0), history_embs.unsqueeze(0),
        ).squeeze(0)
        paths, search_embeddings, text_embeddings = path_index.candidates(target_item, history)
        if not paths:
            return F.cosine_similarity(target_emb, intent, dim=0).item()
        search_embeddings = search_embeddings.to(intent.device)
        text_embeddings = text_embeddings.to(intent.device)
        similarities = F.cosine_similarity(search_embeddings, intent.unsqueeze(0), dim=-1)
        indices = torch.topk(similarities, min(self.candidate_pool, len(paths))).indices
        return max(
            self.score_path(
                paths[index], text_embeddings[index], intent,
                node_embeddings, node_degrees, adj,
            ).item()
            for index in indices.tolist()
        )
