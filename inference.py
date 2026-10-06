import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PUREConfig
from data.dataset import KnowledgeGraph, RecommendationDataset, collate_fn
from evaluation.metrics import PUREEvaluator
from models.path_retrieval import PathIndex
from models.pure_model import PUREModel
from models.semantic_index import NodeSpecificityScorer
from train import (
    build_edge_tensors,
    build_or_load_index,
    build_or_load_path_index,
    load_data,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inference.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_pure_model(
    config: PUREConfig,
    checkpoint_path: str,
    kg: KnowledgeGraph,
    node_embeddings: torch.Tensor,
    specificity_scorer: NodeSpecificityScorer,
    path_index: PathIndex,
) -> PUREModel:
    logger.info(f"Loading PURE model from: {checkpoint_path}")
    ckpt_dir = Path(checkpoint_path).parent
    ckpt = torch.load(
        checkpoint_path, map_location=config.device, weights_only=True,
    )
    lora_path = (
        ckpt_dir / "best_lora"
        if Path(checkpoint_path).stem == "best_model"
        else ckpt_dir / f"lora_epoch{ckpt['epoch'] + 1}"
    )
    if not lora_path.exists():
        raise FileNotFoundError(f"Fine-tuned LoRA adapter is required: {lora_path}")
    model = PUREModel(
        config=config,
        kg=kg,
        node_embeddings=node_embeddings,
        specificity_scorer=specificity_scorer,
        path_index=path_index,
        adapter_path=str(lora_path),
    )

    model.graph_transformer.load_state_dict(ckpt["graph_transformer"])
    model.projector.load_state_dict(ckpt["projector"])
    model.align_projector.load_state_dict(ckpt["align_projector"])
    model.path_retrieval.load_state_dict(ckpt["path_retrieval"])
    logger.info(
        f"Graph modules loaded — "
        f"epoch: {ckpt.get('epoch', 'N/A')}, "
        f"step: {ckpt.get('global_step', 'N/A')}"
    )

    model = model.move_graph_modules(config.device)
    model.eval()
    logger.info("Model ready for inference.")
    return model


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PUREInferencer:
    def __init__(
        self,
        model: PUREModel,
        kg: KnowledgeGraph,
        id2entity: Dict[int, str],
        id2relation: Dict[int, str],
        config: PUREConfig,
        evaluator: PUREEvaluator,
    ):
        self.model           = model
        self.kg              = kg
        self.id2entity       = id2entity
        self.id2relation     = id2relation
        self.config          = config
        self.evaluator       = evaluator

    @torch.no_grad()
    def generate_single(
        self,
        target_item: int,
        history: List[int],
        max_new_tokens: int = 128,
        return_paths: bool = False,
        return_subgraph: bool = False,
    ) -> Dict:
        _sync_cuda()
        start_time = time.perf_counter()

        explanation = self.model.generate(
            target_item=target_item,
            history=history,
            node_degrees=self.kg.degree,
            adj=self.kg.adj,
            id2entity=self.id2entity,
            id2relation=self.id2relation,
            max_new_tokens=max_new_tokens,
        )

        _sync_cuda()
        latency_ms = (time.perf_counter() - start_time) * 1000

        result: Dict = {
            "explanation": explanation,
            "latency_ms":  round(latency_ms, 2),
        }

        if return_paths or return_subgraph:
            selected_paths, _ = self.model.path_retrieval.retrieve(
                target_emb=self.model.node_embeddings[target_item],
                history_embs=self.model.node_embeddings[history],
                target_item=target_item,
                history=history,
                path_index=self.model.path_index,
                node_embeddings=self.model.node_embeddings,
                node_degrees=self.kg.degree,
                adj=self.kg.adj,
            )
            if return_paths:
                result["selected_paths"] = self._format_paths(selected_paths)
            if return_subgraph:
                result["subgraph"] = self._format_subgraph(selected_paths)

        return result

    def _format_paths(
        self,
        paths: List[List[Tuple[int, int, int]]],
    ) -> List[List[Dict]]:
        return [
            [
                {
                    "head":     self.id2entity.get(h, f"entity_{h}"),
                    "relation": self.id2relation.get(r, f"rel_{r}"),
                    "tail":     self.id2entity.get(t, f"entity_{t}"),
                }
                for (h, r, t) in path
            ]
            for path in paths
        ]

    def _format_subgraph(
        self,
        paths: List[List[Tuple[int, int, int]]],
    ) -> Dict:
        nodes: Dict[int, str] = {}
        edges: List[Dict]     = []
        for path in paths:
            for (h, r, t) in path:
                nodes[h] = self.id2entity.get(h, f"entity_{h}")
                nodes[t] = self.id2entity.get(t, f"entity_{t}")
                edges.append({
                    "source":   h,
                    "target":   t,
                    "relation": self.id2relation.get(r, f"rel_{r}"),
                })
        return {
            "nodes": [{"id": nid, "name": name} for nid, name in nodes.items()],
            "edges": edges,
        }

    @torch.no_grad()
    def evaluate_dataset(
        self,
        test_loader: DataLoader,
        output_path: Optional[str] = None,
        max_new_tokens: int = 128,
        save_every: int = 100,
        eval_k: int = 5,
    ) -> Dict[str, float]:
        generated_texts:        List[str]      = []
        reference_texts:        List[str]      = []
        latencies:              List[float]    = []
        all_results:            List[Dict]     = []
        ranking_predictions:    List[List[int]] = []
        ranking_targets:        List[int]       = []

        output_file = Path(output_path) if output_path else None
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(test_loader, desc="Evaluating [test]")

        for batch in pbar:
            for i in range(len(batch["target_items"])):
                target_item    = batch["target_items"][i]
                history        = batch["histories"][i]
                ref_text       = batch["explanation_texts"][i]
                item_feats     = set(batch["item_features"][i])
                user_pos_feats = set(batch["user_pos_features"][i])
                negative_items = batch["negative_items"][i]
                history_texts  = [self.id2entity.get(h, str(h)) for h in history]

                result     = self.generate_single(
                    target_item=target_item,
                    history=history,
                    max_new_tokens=max_new_tokens,
                    return_paths=True,
                )
                gen_text   = result["explanation"]
                latency_ms = result["latency_ms"]

                generated_texts.append(gen_text)
                reference_texts.append(ref_text)
                latencies.append(latency_ms)

                all_results.append({
                    "user_id":           batch["user_ids"][i],
                    "target_item_id":    target_item,
                    "target_item":       self.id2entity.get(target_item, str(target_item)),
                    "history":           history_texts,
                    "reference":         ref_text,
                    "generated":         gen_text,
                    "latency_ms":        latency_ms,
                    "selected_paths":    result.get("selected_paths", []),
                    "item_features":     list(item_feats),
                    "user_pos_features": list(user_pos_feats),
                })
                if negative_items:
                    ranking_predictions.append(self.model.rank_candidates(
                        history, [target_item, *negative_items],
                        self.kg.degree, self.kg.adj,
                    ))
                    ranking_targets.append(target_item)

                pbar.set_postfix({
                    "avg_lat": f"{np.mean(latencies):.0f}ms",
                    "n":       len(generated_texts),
                })

                if output_file and len(all_results) % save_every == 0:
                    self._save_partial(all_results, output_file)
                    logger.info(f"Partial results saved ({len(all_results)} samples)")

        logger.info("Computing text quality metrics...")
        explanation_metrics = self.evaluator.evaluate_text(
            predictions=generated_texts,
            references=reference_texts,
        )

        latency_stats = {
            "latency_mean_ms": round(float(np.mean(latencies)),           2),
            "latency_p50_ms":  round(float(np.percentile(latencies, 50)), 2),
            "latency_p95_ms":  round(float(np.percentile(latencies, 95)), 2),
            "latency_p99_ms":  round(float(np.percentile(latencies, 99)), 2),
        }

        metrics: Dict[str, float] = {
            "bleu4":     round(explanation_metrics["BLEU-4"],  4),
            "rouge_l":   round(explanation_metrics["ROUGE-L"], 4),
            "div":       round(explanation_metrics["DIV"],     4),
            **latency_stats,
            "n_samples": len(generated_texts),
        }
        if ranking_targets:
            metrics.update(self.evaluator.evaluate_ranking(
                ranking_predictions, ranking_targets,
            ))

        if output_file:
            self._save_full(all_results, metrics, output_file)
            logger.info(f"Full results saved to: {output_file}")

        self._print_report(metrics)
        return metrics

    @torch.no_grad()
    def profile_latency(
        self,
        test_samples: List[Dict],
        n_warmup: int = 5,
        n_repeat: int = 50,
        max_new_tokens: int = 128,
    ) -> Dict[str, float]:
        logger.info(f"Profiling latency: {n_warmup} warmup + {n_repeat} measurements")

        for i in range(min(n_warmup, len(test_samples))):
            s = test_samples[i % len(test_samples)]
            self.generate_single(
                target_item=s["target_item"],
                history=s["history"],
                max_new_tokens=max_new_tokens,
            )
        logger.info("Warmup complete.")

        latencies: List[float] = []
        for i in range(n_repeat):
            s      = test_samples[i % len(test_samples)]
            result = self.generate_single(
                target_item=s["target_item"],
                history=s["history"],
                max_new_tokens=max_new_tokens,
            )
            latencies.append(result["latency_ms"])

        stats = {
            "mean_ms":   round(float(np.mean(latencies)),           2),
            "std_ms":    round(float(np.std(latencies)),            2),
            "min_ms":    round(float(np.min(latencies)),            2),
            "max_ms":    round(float(np.max(latencies)),            2),
            "p50_ms":    round(float(np.percentile(latencies, 50)), 2),
            "p95_ms":    round(float(np.percentile(latencies, 95)), 2),
            "p99_ms":    round(float(np.percentile(latencies, 99)), 2),
            "n_samples": n_repeat,
        }
        logger.info(
            f"Latency — mean: {stats['mean_ms']}ms | "
            f"p50: {stats['p50_ms']}ms | "
            f"p95: {stats['p95_ms']}ms"
        )
        return stats

    def interactive_mode(self):
        entity2id: Dict[str, int] = {v: k for k, v in self.id2entity.items()}

        print("\n" + "=" * 60)
        print("  PURE Interactive Inference Mode")
        print("  Commands: 'quit' to exit | 'help' for usage")
        print("=" * 60)

        while True:
            print()
            history_input = input("Enter history items (comma-separated names): ").strip()

            if history_input.lower() == "quit":
                print("Exiting interactive mode.")
                break

            if history_input.lower() == "help":
                print(
                    "\nUsage:\n"
                    "  History : comma-separated item names (e.g. 'The Green Mile, Se7en')\n"
                    "  Target  : single item name to explain\n"
                    "  Fuzzy matching is supported for partial names.\n"
                )
                continue

            target_input = input("Enter target item name: ").strip()
            if target_input.lower() == "quit":
                print("Exiting interactive mode.")
                break

            history_names = [h.strip() for h in history_input.split(",")]
            history_ids: List[int] = []
            for name in history_names:
                eid = entity2id.get(name)
                if eid is None:
                    candidates = [k for k in entity2id if name.lower() in k.lower()]
                    if candidates:
                        best = candidates[0]
                        eid  = entity2id[best]
                        print(f"  [Fuzzy] '{name}' → '{best}'")
                    else:
                        print(f"  [Warning] '{name}' not found in KG, skipped.")
                        continue
                history_ids.append(eid)

            target_id = entity2id.get(target_input)
            if target_id is None:
                candidates = [k for k in entity2id if target_input.lower() in k.lower()]
                if candidates:
                    best      = candidates[0]
                    target_id = entity2id[best]
                    print(f"  [Fuzzy] '{target_input}' → '{best}'")
                else:
                    print(f"  [Error] Target '{target_input}' not found in KG.")
                    continue

            if not history_ids:
                print("  [Error] No valid history items resolved.")
                continue

            print("\nGenerating explanation...")
            result = self.generate_single(
                target_item=target_id,
                history=history_ids,
                max_new_tokens=128,
                return_paths=True,
            )

            print("\n" + "-" * 55)
            print(f"Target  : {self.id2entity.get(target_id, str(target_id))}")
            print(f"History : {[self.id2entity.get(h, str(h)) for h in history_ids]}")
            print(f"Latency : {result['latency_ms']:.1f} ms")
            print()

            paths = result.get("selected_paths", [])
            if paths:
                print("Selected reasoning paths:")
                for j, path in enumerate(paths, 1):
                    path_str = " → ".join(
                        f"{step['head']} -[{step['relation']}]-> {step['tail']}"
                        for step in path
                    )
                    print(f"  Path {j}: {path_str}")
                print()

            print("Generated explanation:")
            print(f"  {result['explanation']}")
            print("-" * 55)

    def _save_partial(self, results: List[Dict], output_file: Path):
        partial_path = output_file.with_suffix(".partial.json")
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _save_full(
        self,
        results: List[Dict],
        metrics: Dict[str, float],
        output_file: Path,
    ):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "results": results}, f, ensure_ascii=False, indent=2)

        metrics_file = output_file.with_name(output_file.stem + "_metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def _print_report(self, metrics: Dict[str, float]):
        sep = "=" * 55
        print(f"\n{sep}")
        print("  PURE Evaluation Report")
        print(sep)
        print(f"  Samples evaluated  : {metrics['n_samples']}")
        print()
        print("  [Text Quality]")
        print(f"    BLEU-4  (↑)      : {metrics['bleu4']:.4f}")
        print(f"    ROUGE-L (↑)      : {metrics['rouge_l']:.4f}")
        print()
        print("  [Diversity]")
        print(f"    DIV (↓)          : {metrics['div']:.4f}")
        print()
        print("  [Inference Efficiency]")
        print(f"    Mean latency     : {metrics['latency_mean_ms']:.1f} ms")
        print(f"    P50  latency     : {metrics['latency_p50_ms']:.1f} ms")
        print(f"    P95  latency     : {metrics['latency_p95_ms']:.1f} ms")
        print(f"    P99  latency     : {metrics['latency_p99_ms']:.1f} ms")
        print(sep + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PURE Inference & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="evaluate",
        choices=["evaluate", "interactive", "profile"],
    )
    parser.add_argument(
        "--dataset", type=str, default="movies",
        choices=["books", "movies", "yelp"],
    )
    parser.add_argument("--data_dir",       type=str, default="./data")
    parser.add_argument("--checkpoint",     type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--output_path",    type=str, default="./results/test_results.json")
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--eval_k",         type=int, default=5)
    parser.add_argument("--device",         type=str, default="cuda")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--path_score_threshold", type=float, default=0.0)
    parser.add_argument("--candidate_pool", type=int, default=40)
    parser.add_argument("--mmr_gamma", type=float, default=0.6)
    parser.add_argument("--profile_warmup", type=int, default=5)
    parser.add_argument("--profile_repeat", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    config = PUREConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        path_score_threshold=args.path_score_threshold,
        candidate_pool=args.candidate_pool,
        mmr_gamma=args.mmr_gamma,
    )
    logger.info(
        f"Mode: [{args.mode}] | Dataset: [{args.dataset}] | Device: [{args.device}]"
    )

    data         = load_data(config)
    kg           = data["kg"]
    entity_texts = data["entity_texts"]
    id2entity    = data["id2entity"]
    id2relation  = data["id2relation"]

    edge_index, edge_type = build_edge_tensors(kg, config.device)
    node_embeddings       = build_or_load_index(config, kg, entity_texts, edge_index, edge_type)
    logger.info(f"Node embeddings shape: {node_embeddings.shape}")

    specificity_scorer = NodeSpecificityScorer(
        lambda_s=config.lambda_s,
        lambda_m=config.lambda_m,
        lambda_p=config.lambda_p,
        alpha_struct=config.alpha_struct,
        n_clusters=config.n_clusters,
    )
    logger.info("Fitting specificity scorer clusters...")
    specificity_scorer.fit_clusters(
        node_embeddings.cpu().numpy(),
        all_node_ids=list(range(kg.n_entities)),
        adj=kg.adj,
    )

    path_index = build_or_load_path_index(
        config, kg, data, node_embeddings, id2entity, id2relation,
    )

    model = load_pure_model(
        config=config,
        checkpoint_path=args.checkpoint,
        kg=kg,
        node_embeddings=node_embeddings,
        specificity_scorer=specificity_scorer,
        path_index=path_index,
    )

    pure_evaluator = PUREEvaluator(eval_k=args.eval_k)

    inferencer = PUREInferencer(
        model=model,
        kg=kg,
        id2entity=id2entity,
        id2relation=id2relation,
        config=config,
        evaluator=pure_evaluator,
    )

    if args.mode == "evaluate":
        from transformers import AutoTokenizer
        tokenizer           = AutoTokenizer.from_pretrained(config.llm_name)
        tokenizer.pad_token = tokenizer.eos_token

        test_dataset = RecommendationDataset(
            data["test"], kg, tokenizer,
            max_history=config.history_len,
            split="test",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        logger.info(f"Test set size: {len(test_dataset)}")

        metrics = inferencer.evaluate_dataset(
            test_loader=test_loader,
            output_path=args.output_path,
            max_new_tokens=args.max_new_tokens,
            eval_k=args.eval_k,
        )
        logger.info(
            f"Done. BLEU-4={metrics['bleu4']:.4f} | "
            f"ROUGE-L={metrics['rouge_l']:.4f}"
        )

    elif args.mode == "interactive":
        inferencer.interactive_mode()

    elif args.mode == "profile":
        test_samples = [
            {
                "target_item": s["target_item"],
                "history":     s["history"][-config.history_len:],
            }
            for s in data["test"][: args.profile_repeat + args.profile_warmup]
        ]
        stats        = inferencer.profile_latency(
            test_samples=test_samples,
            n_warmup=args.profile_warmup,
            n_repeat=args.profile_repeat,
            max_new_tokens=args.max_new_tokens,
        )
        profile_path = Path(args.output_path).with_name(
            f"latency_profile_{args.dataset}.json"
        )
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Latency profile saved to: {profile_path}")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
