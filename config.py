from dataclasses import dataclass, field

import torch


@dataclass
class PUREConfig:
    dataset:            str   = "movies"
    data_dir:           str   = "./data"
    kg_entity_file:     str   = "kg_entities.json"
    kg_relation_file:   str   = "kg_relations.json"
    history_len:        int   = 10

    plm_name:           str   = "sentence-transformers/all-MiniLM-L6-v2"
    plm_hidden:         int   = 384

    rgat_hidden:        int   = 1024
    rgat_heads:         int   = 4
    rgat_layers:        int   = 3
    rgat_out_channels:  int   = 256

    gt_hidden:          int   = 256
    gt_heads:           int   = 2
    gt_layers:          int   = 2
    max_hop:            int   = 3

    top_n_paths:        int   = 5
    mmr_gamma:          float = 0.6
    candidate_pool:     int   = 40

    n_clusters:         int   = 10
    alpha_struct:       float = 1.0
    eps_smooth:         float = 1e-6
    lambda_s:           float = 0.27
    lambda_m:           float = 0.31
    lambda_p:           float = 0.42

    llm_name:           str   = "meta-llama/Llama-3.1-8B"
    llm_hidden:         int   = 4096
    lora_r:             int   = 8
    lora_alpha:         int   = 16
    lora_dropout:       float = 0.1
    n_soft_tokens:      int   = 8

    batch_size:         int   = 8
    grad_accum:         int   = 2
    epochs:             int   = 20
    lr:                 float = 1e-5
    lambda_align:       float = 0.1
    seed:               int   = 42
    device:             str   = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    tau:                float = 0.40
    eval_top_k:         int   = 5

    def __post_init__(self):
        total = self.lambda_s + self.lambda_m + self.lambda_p
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"lambda_s + lambda_m + lambda_p must sum to 1.0, got {total:.4f}"
            )
        if not (0.0 < self.tau < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {self.tau}")
        if self.top_n_paths > self.candidate_pool:
            raise ValueError(
                f"top_n_paths ({self.top_n_paths}) must be <= "
                f"candidate_pool ({self.candidate_pool})"
            )
        if self.rgat_out_channels != self.gt_hidden:
            raise ValueError(
                f"rgat_out_channels ({self.rgat_out_channels}) must equal "
                f"gt_hidden ({self.gt_hidden})"
            )
