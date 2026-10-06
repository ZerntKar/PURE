from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    LoraConfig, PeftModel, TaskType, get_peft_model,
    prepare_model_for_kbit_training,
)
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .graph_transformer import GraphTransformer
from .path_retrieval import PreferenceAwarePathRetrieval


class SubgraphProjector(nn.Module):
    def __init__(self, graph_dim: int, llm_dim: int, n_soft_tokens: int = 8):
        super().__init__()
        self.n_soft_tokens = n_soft_tokens
        self.llm_dim = llm_dim
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim, graph_dim * 2),
            nn.GELU(),
            nn.Linear(graph_dim * 2, n_soft_tokens * llm_dim),
        )

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        return self.mlp(h_graph).view(-1, self.n_soft_tokens, self.llm_dim)


class AlignmentProjector(nn.Module):
    def __init__(self, graph_dim: int, align_dim: int = 256):
        super().__init__()
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, align_dim),
            nn.LayerNorm(align_dim),
        )

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        return self.graph_proj(h_graph)


class PUREModel(nn.Module):
    def __init__(
        self,
        config,
        kg,
        node_embeddings: torch.Tensor,
        specificity_scorer,
        path_index,
        adapter_path: Optional[str] = None,
        adapter_trainable: bool = False,
    ):
        super().__init__()
        self.config = config
        self.kg = kg
        self.path_index = path_index

        self.register_buffer("node_embeddings", node_embeddings)

        self.graph_transformer = GraphTransformer(
            in_channels=node_embeddings.shape[-1],
            hidden_channels=config.gt_hidden,
            num_relations=kg.n_relations,
            num_layers=config.gt_layers,
            heads=config.gt_heads,
        )

        self.projector = SubgraphProjector(
            graph_dim=config.gt_hidden,
            llm_dim=config.llm_hidden,
            n_soft_tokens=config.n_soft_tokens,
        )

        self.sent_encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.sent_encoder.requires_grad_(False)
        self.align_projector = AlignmentProjector(
            graph_dim=config.gt_hidden,
            align_dim=self.sent_encoder.get_sentence_embedding_dimension(),
        )

        self.path_retrieval = PreferenceAwarePathRetrieval(
            embed_dim=node_embeddings.shape[-1],
            specificity_scorer=specificity_scorer,
            top_n=config.top_n_paths,
            mmr_gamma=config.mmr_gamma,
            candidate_pool=config.candidate_pool,
            score_threshold=config.path_score_threshold,
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_name)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        base_llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        base_llm = prepare_model_for_kbit_training(base_llm)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        self.llm = (
            PeftModel.from_pretrained(
                base_llm, adapter_path, is_trainable=adapter_trainable,
            )
            if adapter_path else get_peft_model(base_llm, lora_cfg)
        )

        self.lambda_align = config.lambda_align

    def train(self, mode: bool = True):
        super().train(mode)
        # The alignment target is supplied by a frozen sentence encoder.
        self.sent_encoder.eval()
        return self

    def move_graph_modules(self, device):
        """Keep the 8-bit device-mapped LLM in place while moving graph modules."""
        self.node_embeddings = self.node_embeddings.to(device)
        self.graph_transformer.to(device)
        self.projector.to(device)
        self.align_projector.to(device)
        self.path_retrieval.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.graph_transformer.parameters()).device

    def encode_subgraph(
        self,
        selected_paths: List[List[Tuple[int, int, int]]],
        user_intent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not selected_paths:
            return self.node_embeddings.new_zeros(1, self.config.gt_hidden)

        nodes: set = set()
        edges: List[Tuple[int, int, int]] = []
        for path in selected_paths:
            for h, r, t in path:
                nodes.add(h)
                nodes.add(t)
                edges.append((h, r, t))

        node_list = sorted(nodes)
        node2idx  = {n: i for i, n in enumerate(node_list)}
        x         = self.node_embeddings[node_list]
        if user_intent is not None:
            preference = 0.5 * (
                1.0 + F.cosine_similarity(x, user_intent.unsqueeze(0), dim=-1)
            )
            x = x * preference.unsqueeze(-1)

        if edges:
            src        = torch.tensor([node2idx[h] for h, r, t in edges], device=self.device)
            dst        = torch.tensor([node2idx[t] for h, r, t in edges], device=self.device)
            edge_index = torch.stack([src, dst], dim=0)
            edge_type  = torch.tensor([r for h, r, t in edges], device=self.device)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            edge_type  = torch.zeros(0, dtype=torch.long, device=self.device)

        batch   = torch.zeros(len(node_list), dtype=torch.long, device=self.device)
        h_graph = self.graph_transformer(x, edge_index, edge_type, batch)
        return h_graph

    def encode_subgraph_batch(
        self,
        paths_batch: List[List[List[Tuple[int, int, int]]]],
        intents: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        all_x:          List[torch.Tensor] = []
        all_edge_index: List[torch.Tensor] = []
        all_edge_type:  List[torch.Tensor] = []
        all_batch:      List[torch.Tensor] = []
        node_offset = 0

        for batch_idx, selected_paths in enumerate(paths_batch):
            if not selected_paths:
                all_x.append(self.node_embeddings.new_zeros(1, self.node_embeddings.shape[-1]))
                all_batch.append(torch.tensor([batch_idx], device=self.device))
                node_offset += 1
                continue

            nodes: set = set()
            edges: List[Tuple[int, int, int]] = []
            for path in selected_paths:
                for h, r, t in path:
                    nodes.add(h)
                    nodes.add(t)
                    edges.append((h, r, t))

            node_list = sorted(nodes)
            node2idx  = {n: i + node_offset for i, n in enumerate(node_list)}

            node_features = self.node_embeddings[node_list]
            if intents is not None:
                # Differentiable preference weighting lets the paper's two
                # losses train the target-aware attention projection.
                preference = 0.5 * (
                    1.0 + F.cosine_similarity(
                        node_features, intents[batch_idx].unsqueeze(0), dim=-1,
                    )
                )
                node_features = node_features * preference.unsqueeze(-1)
            all_x.append(node_features)
            all_batch.append(
                torch.full((len(node_list),), batch_idx, dtype=torch.long, device=self.device)
            )

            if edges:
                src = torch.tensor([node2idx[h] for h, r, t in edges], device=self.device)
                dst = torch.tensor([node2idx[t] for h, r, t in edges], device=self.device)
                all_edge_index.append(torch.stack([src, dst], dim=0))
                all_edge_type.append(
                    torch.tensor([r for h, r, t in edges], device=self.device)
                )

            node_offset += len(node_list)

        x            = torch.cat(all_x, dim=0)
        batch_tensor = torch.cat(all_batch, dim=0)

        if all_edge_index:
            edge_index = torch.cat(all_edge_index, dim=1)
            edge_type  = torch.cat(all_edge_type, dim=0)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            edge_type  = torch.zeros(0, dtype=torch.long, device=self.device)

        return self.graph_transformer(x, edge_index, edge_type, batch_tensor)

    def build_prompt(
        self,
        history_texts: List[str],
        target_text: str,
        selected_paths: List[List[Tuple[int, int, int]]],
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
    ) -> str:
        sys_instruction = (
            "You are a helpful recommendation assistant. "
            "Generate a personalized explanation for why the user would enjoy the recommended item. "
            "Focus on aspects that align with the user's demonstrated preferences."
        )
        history_str  = "User interaction history: " + ", ".join(
            history_texts[-self.config.history_len:]
        )
        path_strs = []
        for path in selected_paths:
            parts = []
            for h, r, t in path:
                h_name = id2entity.get(h, str(h))
                r_name = id2relation.get(r, str(r))
                t_name = id2entity.get(t, str(t))
                parts.append(f"{h_name} --[{r_name}]--> {t_name}")
            path_strs.append(" | ".join(parts))

        paths_section  = "Reasoning paths:\n" + "\n".join(
            f"  Path {i + 1}: {p}" for i, p in enumerate(path_strs)
        )
        target_section = f"Recommended item: {target_text}"

        return (
            f"{sys_instruction}\n\n"
            f"{history_str}\n\n"
            f"{target_section}\n\n"
            f"{paths_section}\n\n"
            f"Explanation:"
        )

    def forward(
        self,
        batch: Dict,
        node_degrees: Dict[int, int],
        adj: Dict[int, List[int]],
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
    ) -> Dict[str, torch.Tensor]:
        B = len(batch["target_items"])

        all_selected_paths: List[List[List[Tuple[int, int, int]]]] = []
        all_hard_prompts:   List[str] = []
        all_intents:        List[torch.Tensor] = []

        for i in range(B):
            target_item      = batch["target_items"][i]
            history          = batch["histories"][i]
            explanation_text = batch["explanation_texts"][i]

            target_emb   = self.node_embeddings[target_item]
            history_embs = self.node_embeddings[history]

            selected_paths, user_intent = self.path_retrieval.retrieve(
                target_emb=target_emb,
                history_embs=history_embs,
                target_item=target_item,
                history=history,
                path_index=self.path_index,
                node_embeddings=self.node_embeddings,
                node_degrees=node_degrees,
                adj=adj,
            )
            all_selected_paths.append(selected_paths)
            all_intents.append(user_intent)

            history_texts = [id2entity.get(h, str(h)) for h in history]
            target_text   = id2entity.get(target_item, str(target_item))
            hard_prompt   = self.build_prompt(
                history_texts, target_text, selected_paths, id2entity, id2relation
            )
            all_hard_prompts.append(hard_prompt)

        h_graphs     = self.encode_subgraph_batch(all_selected_paths, all_intents)
        soft_prompts = self.projector(h_graphs)

        # Tokenize the prompt and supervised continuation separately. Counting
        # tokens in the prompt alone and then re-tokenizing their concatenated
        # text can move the BPE boundary, supervising a prompt token by mistake.
        sequences = []
        label_sequences = []
        for prompt, explanation in zip(all_hard_prompts, batch["explanation_texts"]):
            prompt_ids = self.llm_tokenizer(
                prompt, add_special_tokens=True,
            )["input_ids"]
            answer_ids = self.llm_tokenizer(
                " " + explanation, add_special_tokens=False, truncation=True,
                max_length=128,
            )["input_ids"]
            answer_ids.append(self.llm_tokenizer.eos_token_id)
            sequences.append(prompt_ids + answer_ids)
            label_sequences.append([-100] * len(prompt_ids) + answer_ids)

        max_length = max(map(len, sequences))
        pad_id = self.llm_tokenizer.pad_token_id
        input_ids = torch.tensor([
            row + [pad_id] * (max_length - len(row)) for row in sequences
        ], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([
            [1] * len(row) + [0] * (max_length - len(row))
            for row in sequences
        ], dtype=torch.long, device=self.device)
        labels = torch.tensor([
            row + [-100] * (max_length - len(row))
            for row in label_sequences
        ], dtype=torch.long, device=self.device)

        token_embs    = self.llm.get_input_embeddings()(input_ids)
        combined_embs = torch.cat([soft_prompts.to(token_embs), token_embs], dim=1)
        soft_mask     = torch.ones(B, self.config.n_soft_tokens, device=token_embs.device)
        combined_mask = torch.cat([soft_mask, attention_mask], dim=1)

        soft_labels     = torch.full(
            (B, self.config.n_soft_tokens), -100, dtype=torch.long, device=input_ids.device
        )
        combined_labels = torch.cat([soft_labels, labels], dim=1)

        outputs  = self.llm(
            inputs_embeds=combined_embs,
            attention_mask=combined_mask,
            labels=combined_labels,
        )
        loss_gen = outputs.loss

        with torch.no_grad():
            h_y_np = self.sent_encoder.encode(
                batch["explanation_texts"], show_progress_bar=False
            )
        h_y = torch.tensor(h_y_np, dtype=torch.float32, device=self.device)

        h_g_proj   = self.align_projector(h_graphs)
        h_y_normed = F.normalize(h_y, dim=-1)
        h_g_normed = F.normalize(h_g_proj.float(), dim=-1)

        loss_align = (1.0 - (h_g_normed * h_y_normed).sum(dim=-1)).mean()
        loss       = loss_gen + self.lambda_align * loss_align

        return {
            "loss":       loss,
            "loss_gen":   loss_gen,
            "loss_align": loss_align,
        }

    @torch.no_grad()
    def rank_candidates(
        self, history: List[int], candidates: List[int],
        node_degrees, adj,
    ) -> List[int]:
        history_embs = self.node_embeddings[history]
        scored = [
            (
                item,
                self.path_retrieval.score_item(
                    self.node_embeddings[item], history_embs, item, history,
                    self.path_index, self.node_embeddings, node_degrees, adj,
                ),
            )
            for item in candidates
        ]
        return [item for item, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)]

    @torch.no_grad()
    def generate(
        self,
        target_item: int,
        history: List[int],
        node_degrees: Dict[int, int],
        adj: Dict[int, List[int]],
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
        max_new_tokens: int = 128,
    ) -> str:
        target_emb   = self.node_embeddings[target_item]
        history_embs = self.node_embeddings[history]

        selected_paths, user_intent = self.path_retrieval.retrieve(
            target_emb=target_emb, history_embs=history_embs,
            target_item=target_item, history=history, path_index=self.path_index,
            node_embeddings=self.node_embeddings, node_degrees=node_degrees, adj=adj,
        )

        h_graph      = self.encode_subgraph(selected_paths, user_intent)
        soft_prompts = self.projector(h_graph)

        history_texts = [id2entity.get(h, str(h)) for h in history]
        target_text   = id2entity.get(target_item, str(target_item))
        hard_prompt   = self.build_prompt(
            history_texts, target_text, selected_paths, id2entity, id2relation
        )

        enc = self.llm_tokenizer(
            hard_prompt, return_tensors="pt"
        ).to(self.device)

        token_embs    = self.llm.get_input_embeddings()(enc["input_ids"])
        combined_embs = torch.cat([soft_prompts.to(token_embs), token_embs], dim=1)
        soft_mask     = torch.ones(1, self.config.n_soft_tokens, device=token_embs.device)
        combined_mask = torch.cat([soft_mask, enc["attention_mask"]], dim=1)

        generated  = self.llm.generate(
            input_ids=enc["input_ids"],
            inputs_embeds=combined_embs,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.llm_tokenizer.eos_token_id,
        )
        output_ids = generated[0, enc["input_ids"].shape[1]:]
        return self.llm_tokenizer.decode(output_ids, skip_special_tokens=True)
