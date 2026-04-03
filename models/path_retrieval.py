from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TargetAwareUserIntent(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5

    def forward(
        self,
        target_emb: torch.Tensor,
        history_embs: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.W_Q(target_emb).unsqueeze(1)
        keys  = self.W_K(history_embs)
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) * self.scale

        if history_mask is not None:
            scores = scores.masked_fill(~history_mask.bool(), float("-inf"))

        alpha  = F.softmax(scores, dim=-1)
        values = self.W_V(history_embs)
        intent = torch.bmm(alpha.unsqueeze(1), values).squeeze(1)
        return intent


class PathEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder   = AutoModel.from_pretrained(model_name)

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def linearize_path(
        self,
        path: List[Tuple[int, int, int]],
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
    ) -> str:
        parts = []
        for h, r, t in path:
            h_name = id2entity.get(h, str(h))
            r_name = id2relation.get(r, str(r))
            t_name = id2entity.get(t, str(t))
            parts.append(f"{h_name} -[{r_name}]-> {t_name}")
        return " | ".join(parts)

    def encode_paths(self, path_texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            path_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        out  = self.encoder(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return embs


class PreferenceAwarePathRetrieval(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        specificity_scorer: "NodeSpecificityScorer",
        path_encoder: PathEncoder,
        top_n: int = 5,
        mmr_gamma: float = 0.6,
    ):
        super().__init__()
        self.intent_model = TargetAwareUserIntent(embed_dim)
        self.specificity  = specificity_scorer
        self.path_encoder = path_encoder
        self.top_n        = top_n
        self.mmr_gamma    = mmr_gamma

    def score_path(
        self,
        path: List[Tuple[int, int, int]],
        path_emb: torch.Tensor,
        user_intent: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_degrees: Dict[int, int],
        adj: Dict[int, List[int]],
    ) -> float:
        cos_sim = F.cosine_similarity(
            user_intent.unsqueeze(0),
            path_emb.unsqueeze(0),
        ).item()

        all_nodes    = [h for h, r, t in path] + [t for h, r, t in path]
        valid_nodes  = list(set(n for n in all_nodes if n < node_embeddings.size(0)))
        if not valid_nodes:
            return cos_sim

        node_ids  = torch.tensor(valid_nodes, dtype=torch.long, device=node_embeddings.device)
        node_embs = node_embeddings[node_ids]
        degrees   = torch.tensor(
            [node_degrees.get(n, 1) for n in valid_nodes],
            dtype=torch.float32,
            device=node_embs.device,
        )

        spec_scores = self.specificity.compute(
            node_ids=valid_nodes,
            node_embeddings=node_embs,
            degrees=degrees,
            adj=adj,
            user_intent=user_intent,
        )
        return cos_sim * spec_scores.mean().item()

    def mmr_rerank(
        self,
        candidate_paths: List[Tuple[List, float, torch.Tensor]],
    ) -> List[Tuple[List, float]]:
        selected: List[Tuple[List, float, torch.Tensor]] = []
        sel_emb_list: List[torch.Tensor] = []
        remaining = list(candidate_paths)

        while len(selected) < self.top_n and remaining:
            best_idx   = -1
            best_score = float("-inf")

            for i, (path, score, emb) in enumerate(remaining):
                if not sel_emb_list:
                    mmr_score = self.mmr_gamma * score
                else:
                    sel_embs = torch.stack(sel_emb_list)
                    sims     = F.cosine_similarity(
                        emb.unsqueeze(0).expand(sel_embs.size(0), -1),
                        sel_embs,
                    )
                    mmr_score = (
                        self.mmr_gamma * score
                        - (1 - self.mmr_gamma) * sims.max().item()
                    )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx   = i

            chosen = remaining.pop(best_idx)
            selected.append(chosen)
            sel_emb_list.append(chosen[2])

        return [(path, score) for path, score, _ in selected]

    def retrieve(
        self,
        target_emb: torch.Tensor,
        history_embs: torch.Tensor,
        candidate_paths: List[List[Tuple[int, int, int]]],
        node_embeddings: torch.Tensor,
        node_degrees: Dict[int, int],
        adj: Dict[int, List[int]],
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
        history_mask: Optional[torch.Tensor] = None,
    ) -> List[List[Tuple[int, int, int]]]:
        if not candidate_paths:
            return []

        user_intent = self.intent_model(
            target_emb.unsqueeze(0),
            history_embs.unsqueeze(0),
            history_mask=history_mask,
        ).squeeze(0)

        path_texts = [
            self.path_encoder.linearize_path(p, id2entity, id2relation)
            for p in candidate_paths
        ]
        path_embs = self.path_encoder.encode_paths(path_texts)

        scored = [
            (path, self.score_path(path, path_emb, user_intent, node_embeddings, node_degrees, adj), path_emb)
            for path, path_emb in zip(candidate_paths, path_embs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected = self.mmr_rerank(scored[:self.top_n * 2])
        return [path for path, _ in selected]
