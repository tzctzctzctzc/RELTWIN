#!/usr/bin/env python3
"""Apply a locked NOVA router using features only; labels never affect selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def feature_vector(candidate: dict, names: list[str]) -> np.ndarray:
    return np.asarray([candidate["features"][name] for name in names], dtype=np.float64)


def main():
    args = parse_args()
    rows = load_jsonl(args.features)
    model = json.loads(args.model.read_text(encoding="utf-8"))
    official = model["official"]
    names = model["feature_names"]
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    threshold = float(model["threshold"])

    selected = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            candidates = row["candidates"]
            base = feature_vector(candidates[official], names)
            predicted_delta = {official: 0.0}
            for name, candidate in candidates.items():
                if name == official:
                    continue
                delta = (feature_vector(candidate, names) - base - mean) / scale
                predicted_delta[name] = float(np.r_[1.0, delta] @ weights)
            best = max(
                (name for name in candidates if name != official),
                key=predicted_delta.get,
            )
            chosen = best if predicted_delta[best] > threshold else official
            candidate = candidates[chosen]
            event = {
                "status": "ok",
                "index": row["index"],
                "audio": row["audio"],
                "query": row["query"],
                "ground_truth": row["ground_truth"],
                "prediction": candidate["prediction"],
                "iou": candidate.get("iou"),
                "selected_candidate": chosen,
                "predicted_delta": predicted_delta,
                "threshold": threshold,
            }
            selected.append(event)
            handle.write(json.dumps(event) + "\n")

    available_iou = [row["iou"] for row in selected if row["iou"] is not None]
    counts = {name: sum(row["selected_candidate"] == name for row in selected) for name in rows[0]["candidates"]}
    summary = {
        "rows": len(selected),
        "selection_counts": counts,
        "mIoU_percent": float(np.mean(available_iou) * 100) if available_iou else None,
        "selection_is_label_free": True,
        "router_model": str(args.model.resolve()),
        "features": str(args.features.resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
