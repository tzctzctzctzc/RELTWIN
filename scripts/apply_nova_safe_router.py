#!/usr/bin/env python3
"""Apply a frozen NOVA-Safe model without consulting labels for selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from nova_safe import SCHEMA_VERSION, choose_candidate, load_jsonl, sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.features)
    model = json.loads(args.model.read_text(encoding="utf-8"))
    if model.get("feature_schema_version") != SCHEMA_VERSION:
        raise ValueError("Router model feature schema does not match NOVA-Safe v2")
    seen = set()
    events = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_hash = sha256_file(args.model)
    features_hash = sha256_file(args.features)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            key = (str(row["benchmark"]), int(row["source_index"]))
            if key in seen:
                raise ValueError(f"Duplicate feature key: {key}")
            seen.add(key)
            decision = choose_candidate(row, model)
            selected_name = decision["selected"]
            incumbent_name = row["incumbent"]
            selected = row["candidates"][selected_name]
            incumbent = row["candidates"][incumbent_name]
            event = {
                "status": "ok",
                "feature_schema_version": SCHEMA_VERSION,
                "benchmark": row["benchmark"],
                "source": row["source"],
                "source_index": row["source_index"],
                "audio": row["audio"],
                "audio_group": row["audio_group"],
                "query": row["query"],
                "duration_seconds": row["duration_seconds"],
                "ground_truth": row.get("ground_truth"),
                "prediction": selected["prediction"],
                "iou": selected.get("iou"),
                "incumbent_name": incumbent_name,
                "incumbent_prediction": incumbent["prediction"],
                "incumbent_iou": incumbent.get("iou"),
                "selected_candidate": selected_name,
                "candidate_ious": {
                    name: candidate.get("iou") for name, candidate in row["candidates"].items()
                },
                "candidate_predictions": {
                    name: candidate["prediction"] for name, candidate in row["candidates"].items()
                },
                "decision_margin": decision["decision_margin"],
                "abstain_reason": decision["abstain_reason"],
                "guarded_candidates": decision["guarded"],
                "candidate_scores": decision["scores"],
                "router_model_sha256": model_hash,
                "features_sha256": features_hash,
                "selection_is_label_free": True,
            }
            events.append(event)
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    counts = Counter(event["selected_candidate"] for event in events)
    reasons = Counter(event["abstain_reason"] for event in events if event["abstain_reason"])
    summary = {
        "rows": len(events),
        "selection_counts": dict(counts),
        "abstain_counts": dict(reasons),
        "selection_is_label_free": True,
        "feature_schema_version": SCHEMA_VERSION,
        "router_model": str(args.model.resolve()),
        "router_model_sha256": model_hash,
        "features": str(args.features.resolve()),
        "features_sha256": features_hash,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

