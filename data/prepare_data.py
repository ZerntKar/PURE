"""Prepare chronological PURE examples from aligned interaction records.

Input JSONL records require user_id, item_id (KG entity ID), timestamp, text,
and optionally explanation. Item feature names come from KG-derived JSON.
Sentires-Guide output supplies phrase-level sentiment quadruples.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from data.sentires import SentiresGuideExtractor


def split_interactions(records, item_features, sentires, history_len=10,
                       candidate_count=40, seed=42):
    by_user = defaultdict(list)
    catalog = {int(item_id) for item_id in item_features}
    for record in records:
        if int(record["item_id"]) not in catalog:
            raise ValueError(
                f"Item {record['item_id']} is absent from the aligned item catalog"
            )
        by_user[record["user_id"]].append(record)
    rng = random.Random(seed)
    splits = {"train": [], "valid": [], "test": []}
    for user_id, interactions in by_user.items():
        interactions.sort(key=lambda record: record["timestamp"])
        if len(interactions) < 3:
            continue
        seen = {int(record["item_id"]) for record in interactions}
        available_negatives = sorted(catalog - seen)
        for index in range(len(interactions)):
            current = interactions[index]
            past = interactions[max(0, index - history_len):index]
            item = int(current["item_id"])
            positive = sentires.extract_positive_features(
                past
            )
            negatives = rng.sample(
                available_negatives,
                min(candidate_count - 1, len(available_negatives)),
            )
            sample = {
                "user_id": user_id,
                "history": [int(record["item_id"]) for record in past],
                "target_item": item,
                "explanation": current.get("explanation", current["text"]),
                "item_features": item_features[str(item)],
                "user_positive_features": sorted(positive),
                "negative_items": negatives,
            }
            split = (
                "test" if index == len(interactions) - 1 else
                "valid" if index == len(interactions) - 2 else "train"
            )
            splits[split].append(sample)
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True)
    parser.add_argument("--item-features", required=True)
    parser.add_argument("--sentires-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with open(args.interactions, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    item_features = json.loads(
        Path(args.item_features).read_text(encoding="utf-8")
    )
    sentires = SentiresGuideExtractor(args.sentires_output)
    splits = split_interactions(
        records, item_features, sentires, args.history_len,
        args.candidate_count, args.seed,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, samples in splits.items():
        (output / f"{name}.json").write_text(
            json.dumps(samples, ensure_ascii=False), encoding="utf-8",
        )
        print(f"{name}: {len(samples)}")


if __name__ == "__main__":
    main()
