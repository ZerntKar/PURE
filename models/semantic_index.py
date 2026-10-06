import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer


class NodeSpecificityScorer(nn.Module):
    def __init__(
        self,
        lambda_s: float = 0.27,
        lambda_m: float = 0.31,
        lambda_p: float = 0.42,
        alpha_struct: float = 1.0,
        eps: float = 1e-6,
        n_clusters: int = 64,
    ):
        super().__init__()
        total = lambda_s + lambda_m + lambda_p
        assert abs(total - 1.0) < 1e-4, (
            f"lambda_s + lambda_m + lambda_p must sum to 1.0, got {total:.4f}"
        )
        self.lambda_s  = lambda_s
        self.lambda_m  = lambda_m
        self.lambda_p  = lambda_p
        self.alpha     = alpha_struct
        self.eps       = eps
        self.n_clusters = n_clusters

        self.cluster_labels: Optional[np.ndarray] = None
        self.label_by_id: Dict[int, int] = {}
        self.kmeans: Optional[KMeans] = None
        self._sem_cache: Dict[int, float] = {}

    def fit_clusters(
        self,
        node_embeddings: np.ndarray,
        all_node_ids: List[int],
        adj: Dict[int, List[int]],
    ) -> None:
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(node_embeddings)
        if len(all_node_ids) != len(self.cluster_labels):
            raise ValueError("One embedding is required for each node ID")
        self.label_by_id = dict(zip(all_node_ids, self.cluster_labels.tolist()))
        self.kmeans = kmeans
        self._precompute_semantic_specificity(all_node_ids, adj)

    def _precompute_semantic_specificity(
        self,
        all_node_ids: List[int],
        adj: Dict[int, List[int]],
    ) -> None:
        scores = self._compute_semantic_scores(all_node_ids, adj)
        self._sem_cache = {nid: s for nid, s in zip(all_node_ids, scores)}

    def _compute_semantic_scores(
        self,
        node_ids: List[int],
        adj: Dict[int, List[int]],
    ) -> List[float]:
        max_entropy = math.log(self.n_clusters)
        scores: List[float] = []
        for nid in node_ids:
            neighbors = adj.get(nid, [])
            if not neighbors or self.cluster_labels is None:
                scores.append(1.0)
                continue
            neighbor_clusters = [
                self.label_by_id[neighbor if isinstance(neighbor, int) else neighbor[1]]
                for neighbor in neighbors
                if (neighbor if isinstance(neighbor, int) else neighbor[1]) in self.label_by_id
            ]
            if not neighbor_clusters:
                scores.append(1.0)
                continue
            counts  = np.bincount(neighbor_clusters, minlength=self.n_clusters)
            probs   = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            scores.append(float(1.0 - entropy / max_entropy))
        return scores

    def structural_specificity(self, degrees: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.alpha * torch.log(degrees.float() + self.eps))

    def semantic_specificity(
        self,
        node_ids: List[int],
        adj: Dict[int, List[int]],
        device: torch.device,
    ) -> torch.Tensor:
        if self._sem_cache:
            scores = [self._sem_cache.get(nid, 1.0) for nid in node_ids]
        else:
            scores = self._compute_semantic_scores(node_ids, adj)
        return torch.tensor(scores, dtype=torch.float32, device=device)

    def preference_specificity(
        self,
        node_embeddings: torch.Tensor,
        user_intent: torch.Tensor,
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(
            node_embeddings,
            user_intent.unsqueeze(0).expand_as(node_embeddings),
            dim=-1,
        )
        return 0.5 * (1.0 + cos_sim)

    def compute(
        self,
        node_ids: List[int],
        node_embeddings: torch.Tensor,
        degrees: torch.Tensor,
        adj: Dict[int, List[int]],
        user_intent: torch.Tensor,
    ) -> torch.Tensor:
        device   = node_embeddings.device
        i_struct = self.structural_specificity(degrees).to(device)
        i_sem    = self.semantic_specificity(node_ids, adj, device)
        i_pref   = self.preference_specificity(node_embeddings, user_intent)
        return self.lambda_s * i_struct + self.lambda_m * i_sem + self.lambda_p * i_pref


class StructureEnhancedIndex:
    def __init__(
        self,
        plm_name: str,
        rgat_model: nn.Module,
        index_path: str = "./index",
    ):
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(plm_name)
        self.plm       = AutoModel.from_pretrained(plm_name)
        for p in self.plm.parameters():
            p.requires_grad = False
        self.plm.eval()

        self.rgat = rgat_model

    @property
    def device(self) -> torch.device:
        return next(self.plm.parameters()).device

    def to(self, device) -> "StructureEnhancedIndex":
        self.plm  = self.plm.to(device)
        self.rgat = self.rgat.to(device)
        return self

    @torch.no_grad()
    def encode_entities(
        self, entity_texts: List[str], batch_size: int = 256
    ) -> torch.Tensor:
        self.plm.eval()
        all_embeddings = []
        for i in range(0, len(entity_texts), batch_size):
            batch = entity_texts[i : i + batch_size]
            enc   = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors="pt",
            ).to(self.device)
            out  = self.plm(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_embeddings.append(emb.cpu())
        return torch.cat(all_embeddings, dim=0)

    def fit_graph_encoder(
        self,
        x: torch.Tensor,
        relation_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        epochs: int,
        learning_rate: float,
    ) -> None:
        """Learn the offline RGAT encoder from KG edges before indexing.

        Relation-aware link reconstruction trains the graph encoder without
        using explanation labels.
        """
        if epochs <= 0:
            return
        relation_decoder = nn.Embedding(relation_embeddings.size(0), self.rgat.layers[-1].heads * self.rgat.layers[-1].out_channels).to(self.device)
        optimizer = torch.optim.AdamW(
            list(self.rgat.parameters()) + list(relation_decoder.parameters()),
            lr=learning_rate,
        )
        n_edges = edge_index.size(1)
        for epoch in range(epochs):
            self.rgat.train()
            optimizer.zero_grad()
            node_repr = self.rgat(x, edge_index, edge_type, relation_embeddings)
            sampled = torch.randperm(n_edges, device=self.device)[: min(n_edges, 50_000)]
            source = edge_index[0, sampled]
            target = edge_index[1, sampled]
            negative = torch.randint(x.size(0), target.shape, device=self.device)
            relation = relation_decoder(edge_type[sampled])
            positive_logits = ((node_repr[source] + relation) * node_repr[target]).sum(-1)
            negative_logits = ((node_repr[source] + relation) * node_repr[negative]).sum(-1)
            loss = F.softplus(-positive_logits).mean() + F.softplus(negative_logits).mean()
            loss.backward()
            optimizer.step()
            print(f"RGAT index pretraining {epoch + 1}/{epochs}: loss={loss.item():.4f}")

    def build_index(
        self,
        entity_texts: List[str],
        relation_texts: List[str],
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        pretrain_epochs: int = 0,
        pretrain_lr: float = 1e-4,
    ) -> torch.Tensor:
        print("Encoding entities with PLM...")
        x = self.encode_entities(entity_texts).to(self.device)
        relation_embeddings = self.encode_entities(relation_texts).to(self.device)

        self.fit_graph_encoder(
            x, relation_embeddings, edge_index.to(self.device), edge_type.to(self.device),
            epochs=pretrain_epochs, learning_rate=pretrain_lr,
        )

        print("Running RGAT for structural encoding...")
        training_state = self.rgat.training
        self.rgat.eval()
        try:
            with torch.no_grad():
                node_embs = self.rgat(
                    x,
                    edge_index.to(self.device),
                    edge_type.to(self.device),
                    relation_embeddings,
                )
        finally:
            self.rgat.train(training_state)

        save_path = self.index_path / "node_embeddings.pt"
        torch.save(node_embs.cpu(), save_path)
        torch.save(self.rgat.state_dict(), self.index_path / "rgat_encoder.pt")
        print(f"Index saved to {save_path}, shape: {node_embs.shape}")
        return node_embs

    def load_index(self) -> torch.Tensor:
        save_path = self.index_path / "node_embeddings.pt"
        if not save_path.exists():
            raise FileNotFoundError(f"Index not found at {save_path}. Run build_index first.")
        return torch.load(save_path, weights_only=True)
