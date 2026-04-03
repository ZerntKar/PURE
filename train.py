import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PUREConfig
from data.dataset import KnowledgeGraph, RecommendationDataset, collate_fn
from evaluation.metrics import PUREEvaluator
from models.path_retrieval import PathEncoder
from models.pure_model import PUREModel
from models.rgat import RGATConv
from models.semantic_index import NodeSpecificityScorer, StructureEnhancedIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train.log"),
    ],
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_data(config: PUREConfig) -> Dict:
    data_dir = Path(config.data_dir) / config.dataset
    logger.info(f"Loading data from {data_dir}")

    kg = KnowledgeGraph.from_files(
        str(data_dir / config.kg_entity_file),
        str(data_dir / config.kg_relation_file),
        str(data_dir / "kg_triples.json"),
    )
    logger.info(
        f"KG loaded: {kg.n_entities} entities, "
        f"{kg.n_relations} relations, {len(kg.triples)} triples"
    )

    with open(data_dir / "entity_texts.json") as f:
        entity_texts: List[str] = json.load(f)
    with open(data_dir / "id2entity.json") as f:
        id2entity: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}
    with open(data_dir / "id2relation.json") as f:
        id2relation: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}

    splits = {}
    for split in ["train", "valid", "test"]:
        with open(data_dir / f"{split}.json") as f:
            splits[split] = json.load(f)

    logger.info(
        f"Dataset splits — train: {len(splits['train'])}, "
        f"valid: {len(splits['valid'])}, test: {len(splits['test'])}"
    )
    return {
        "kg":           kg,
        "entity_texts": entity_texts,
        "id2entity":    id2entity,
        "id2relation":  id2relation,
        **splits,
    }


def build_candidate_paths(
    sample: Dict,
    kg: KnowledgeGraph,
    max_hop: int = 3,
) -> List[List]:
    target    = sample["target_item"]
    history   = sample["history"]
    all_paths = []
    for hist_item in history:
        paths = kg.multi_hop_paths(hist_item, target, max_hop=max_hop)
        all_paths.extend(paths)
    return all_paths


def build_edge_tensors(
    kg: KnowledgeGraph,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    src        = [t.head     for t in kg.triples]
    dst        = [t.tail     for t in kg.triples]
    rel        = [t.relation for t in kg.triples]
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_type  = torch.tensor(rel,        dtype=torch.long, device=device)
    return edge_index, edge_type


def build_or_load_index(
    config: PUREConfig,
    kg: KnowledgeGraph,
    entity_texts: List[str],
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
) -> torch.Tensor:
    index_path = Path(config.data_dir) / config.dataset / "index"
    emb_path   = index_path / "node_embeddings.pt"

    if emb_path.exists():
        logger.info(f"Loading precomputed index from {emb_path}")
        return torch.load(emb_path, map_location=config.device, weights_only=True)

    logger.info("Building structure-enhanced semantic index (offline)...")
    rgat = RGATConv(
        in_channels=config.plm_hidden,
        out_channels=config.rgat_out_channels,
        num_relations=kg.n_relations,
        heads=config.rgat_heads,
    )
    index_builder = StructureEnhancedIndex(
        plm_name=config.plm_name,
        rgat_model=rgat,
        index_path=str(index_path),
    ).to(config.device)

    node_embs = index_builder.build_index(entity_texts, edge_index, edge_type)
    return node_embs.to(config.device)


class Trainer:
    def __init__(
        self,
        config: PUREConfig,
        model: PUREModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        kg: KnowledgeGraph,
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
        evaluator: PUREEvaluator,
        output_dir: str = "./checkpoints",
    ):
        self.config       = config
        self.model        = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.kg           = kg
        self.id2entity    = id2entity
        self.id2relation  = id2relation
        self.evaluator    = evaluator
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(
            f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}"
        )

        self.optimizer = AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        total_steps    = len(train_loader) // config.grad_accum * config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-7
        )
        self.scaler = GradScaler()

        self.best_p_ehr  = float("inf")
        self.best_epoch  = 0
        self.global_step = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss       = 0.0
        total_gen_loss   = 0.0
        total_align_loss = 0.0
        n_batches        = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.epochs}",
            leave=False,
        )
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            candidate_paths_batch = [
                build_candidate_paths(
                    {"target_item": batch["target_items"][i], "history": batch["histories"][i]},
                    self.kg,
                    self.config.max_hop,
                )
                for i in range(len(batch["target_items"]))
            ]

            with autocast():
                outputs = self.model(
                    batch=batch,
                    node_degrees=self.kg.degree,
                    adj=self.kg.adj,
                    id2entity=self.id2entity,
                    id2relation=self.id2relation,
                    candidate_paths_batch=candidate_paths_batch,
                )
                loss = outputs["loss"] / self.config.grad_accum

            self.scaler.scale(loss).backward()

            total_loss       += outputs["loss"].item()
            total_gen_loss   += outputs["loss_gen"].item()
            total_align_loss += outputs["loss_align"].item()
            n_batches        += 1

            if (step + 1) % self.config.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            pbar.set_postfix({
                "loss":  f"{total_loss  / n_batches:.4f}",
                "gen":   f"{total_gen_loss   / n_batches:.4f}",
                "align": f"{total_align_loss / n_batches:.4f}",
            })

        return {
            "train_loss":       total_loss       / n_batches,
            "train_loss_gen":   total_gen_loss   / n_batches,
            "train_loss_align": total_align_loss / n_batches,
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        split: str = "valid",
    ) -> Dict[str, float]:
        self.model.eval()

        generated_texts:        List[str]      = []
        reference_texts:        List[str]      = []
        item_features_list:     List[Set[str]] = []
        user_pos_features_list: List[Set[str]] = []

        pbar = tqdm(loader, desc=f"Evaluating [{split}]", leave=False)

        for batch in pbar:
            for i in range(len(batch["target_items"])):
                target_item = batch["target_items"][i]
                history     = batch["histories"][i]
                ref_text    = batch["explanation_texts"][i]

                candidate_paths = build_candidate_paths(
                    {"target_item": target_item, "history": history},
                    self.kg,
                    self.config.max_hop,
                )

                gen_text = self.model.generate(
                    target_item=target_item,
                    history=history,
                    node_degrees=self.kg.degree,
                    adj=self.kg.adj,
                    id2entity=self.id2entity,
                    id2relation=self.id2relation,
                    candidate_paths=candidate_paths,
                )

                generated_texts.append(gen_text)
                reference_texts.append(ref_text)
                item_features_list.append(set(batch["item_features"][i]))
                user_pos_features_list.append(set(batch["user_pos_features"][i]))

        return self.evaluator.evaluate_explanations(
            predictions=generated_texts,
            references=reference_texts,
            item_features=item_features_list,
            user_pos_features=user_pos_features_list,
        )

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        ckpt = {
            "epoch":             epoch,
            "global_step":       self.global_step,
            "metrics":           metrics,
            "graph_transformer": self.model.graph_transformer.state_dict(),
            "projector":         self.model.projector.state_dict(),
            "path_retrieval":    self.model.path_retrieval.state_dict(),
            "optimizer":         self.optimizer.state_dict(),
            "scheduler":         self.scheduler.state_dict(),
        }

        lora_path = self.output_dir / f"lora_epoch{epoch+1}"
        self.model.llm.save_pretrained(str(lora_path))

        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch+1}.pt"
        torch.save(ckpt, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pt")
            self.model.llm.save_pretrained(str(self.output_dir / "best_lora"))
            logger.info(f"Best model updated → epoch {epoch+1}")

    def load_checkpoint(self, ckpt_path: str) -> int:
        ckpt = torch.load(ckpt_path, map_location=self.config.device, weights_only=True)
        self.model.graph_transformer.load_state_dict(ckpt["graph_transformer"])
        self.model.projector.load_state_dict(ckpt["projector"])
        self.model.path_retrieval.load_state_dict(ckpt["path_retrieval"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]
        logger.info(
            f"Resumed from epoch {ckpt['epoch']+1}, "
            f"global_step {self.global_step}"
        )
        return ckpt["epoch"] + 1

    def train(self, resume_from: Optional[str] = None) -> Dict:
        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)

        logger.info(
            f"Starting training — epochs: {self.config.epochs}, "
            f"dataset: {self.config.dataset}"
        )

        all_metrics: List[Dict] = []

        for epoch in range(start_epoch, self.config.epochs):
            train_metrics = self.train_epoch(epoch)

            if (epoch + 1) % 2 == 0 or epoch == self.config.epochs - 1:
                valid_metrics = self.evaluate(self.valid_loader, split="valid")
                combined      = {
                    **train_metrics,
                    **{f"valid_{k}": v for k, v in valid_metrics.items()},
                }

                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Loss: {train_metrics['train_loss']:.4f} | "
                    f"F-EHR: {valid_metrics['F-EHR']:.4f} | "
                    f"P-EHR: {valid_metrics['P-EHR']:.4f} | "
                    f"BLEU-4: {valid_metrics['BLEU-4']:.4f} | "
                    f"ROUGE-L: {valid_metrics['ROUGE-L']:.4f} | "
                    f"FMR: {valid_metrics['FMR']:.4f} | "
                    f"DIV: {valid_metrics['DIV']:.4f}"
                )

                is_best = valid_metrics["P-EHR"] < self.best_p_ehr
                if is_best:
                    self.best_p_ehr = valid_metrics["P-EHR"]
                    self.best_epoch = epoch

                self.save_checkpoint(epoch, combined, is_best=is_best)
                all_metrics.append(combined)

        logger.info(
            f"Training complete — "
            f"Best P-EHR: {self.best_p_ehr:.4f} at epoch {self.best_epoch+1}"
        )
        return {
            "best_p_ehr":  self.best_p_ehr,
            "best_epoch":  self.best_epoch,
            "all_metrics": all_metrics,
        }


def main():
    parser = argparse.ArgumentParser(description="Train PURE model")
    parser.add_argument("--dataset",    type=str,   default="movies",
                        choices=["books", "movies", "yelp"])
    parser.add_argument("--data_dir",   type=str,   default="./data")
    parser.add_argument("--output_dir", type=str,   default="./checkpoints")
    parser.add_argument("--resume",     type=str,   default=None)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     type=str,   default="cuda")
    args = parser.parse_args()

    config = PUREConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    set_seed(config.seed)
    logger.info(f"Config: {config}")

    data        = load_data(config)
    kg          = data["kg"]
    entity_texts = data["entity_texts"]
    id2entity   = data["id2entity"]
    id2relation = data["id2relation"]

    edge_index, edge_type = build_edge_tensors(kg, config.device)
    node_embeddings       = build_or_load_index(
        config, kg, entity_texts, edge_index, edge_type
    )
    logger.info(f"Node embeddings shape: {node_embeddings.shape}")

    specificity_scorer = NodeSpecificityScorer(
        lambda_s=config.lambda_s,
        lambda_m=config.lambda_m,
        lambda_p=config.lambda_p,
        alpha_struct=config.alpha_struct,
        n_clusters=config.n_clusters,
    )
    logger.info("Fitting semantic clusters for specificity scoring...")
    specificity_scorer.fit_clusters(
        node_embeddings.cpu().numpy(),
        all_node_ids=list(id2entity.keys()),
        adj=kg.adj,
    )

    path_encoder   = PathEncoder(device=config.device)
    entity_vocab   = set(id2entity.values())
    pure_evaluator = PUREEvaluator(
        feature_vocab=entity_vocab,
        tau=config.tau,
        eval_k=config.eval_top_k,
    )
    logger.info(
        f"PUREEvaluator ready — "
        f"vocab_size: {len(entity_vocab)}, tau: {config.tau}"
    )

    from transformers import AutoTokenizer
    tokenizer           = AutoTokenizer.from_pretrained(config.llm_name)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = RecommendationDataset(
        data["train"], kg, tokenizer,
        max_history=config.history_len, split="train",
    )
    valid_dataset = RecommendationDataset(
        data["valid"], kg, tokenizer,
        max_history=config.history_len, split="valid",
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True,  collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    logger.info(
        f"DataLoaders ready — "
        f"train: {len(train_dataset)}, valid: {len(valid_dataset)}"
    )

    model = PUREModel(
        config=config,
        kg=kg,
        node_embeddings=node_embeddings,
        specificity_scorer=specificity_scorer,
        path_encoder=path_encoder,
    ).to(config.device)

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        kg=kg,
        id2entity=id2entity,
        id2relation=id2relation,
        evaluator=pure_evaluator,
        output_dir=args.output_dir,
    )
    results = trainer.train(resume_from=args.resume)
    logger.info(f"Final results: {results}")


if __name__ == "__main__":
    main()
