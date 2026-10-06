"""Recompute paper F-EHR/P-EHR from Sentires-Guide extraction output.

First run inference.py to save generated explanations, then process those
explanations with Sentires-Guide. Supply its quadruple export as pickle or JSON.
"""

import argparse
import json
from pathlib import Path

from data.sentires import SentiresGuideExtractor
from evaluation.metrics import PUREEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--sentires-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tau", type=float, default=0.35)
    args = parser.parse_args()

    inference_report = json.loads(Path(args.results).read_text(encoding="utf-8"))
    records = inference_report["results"]
    extractor = SentiresGuideExtractor(args.sentires_output)
    evaluator = PUREEvaluator(
        tau=args.tau, feature_extractor=extractor,
    )
    generated_features = [
        extractor.extract(
            record["generated"],
            user=record["user_id"],
            item=record["target_item_id"],
        )
        for record in records
    ]
    feature_metrics = evaluator.evaluate_explanations(
        predictions=[record["generated"] for record in records],
        references=[record["reference"] for record in records],
        item_features=[set(record["item_features"]) for record in records],
        user_pos_features=[set(record["user_pos_features"]) for record in records],
        generated_features=generated_features,
    )
    metrics = {
        **inference_report.get("metrics", {}),
        **feature_metrics,
        "feature_source": "Sentires-Guide",
        "p_ehr_evaluable": sum(
            bool(record["user_pos_features"]) or not generated
            for record, generated in zip(records, generated_features)
        ),
    }
    Path(args.output).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
