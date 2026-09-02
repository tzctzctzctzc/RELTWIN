#!/usr/bin/env python3
"""Paired comparison and oracle analysis for interval prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from interval_metrics import event_f1_iou, temporal_set_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", nargs=2,
        metavar=("NAME", "PREDICTIONS"), required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[int, dict]:
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "ok":
                records[int(record["index"])] = record
    if not records:
        raise ValueError(f"No valid predictions: {path}")
    return records


def metrics(ious: np.ndarray, event_f1: np.ndarray) -> dict:
    return {
        "rows": int(len(ious)),
        "mIoU": float(ious.mean() * 100),
        "R1@0.3": float((ious >= 0.3).mean() * 100),
        "R1@0.5": float((ious >= 0.5).mean() * 100),
        "R1@0.7": float((ious >= 0.7).mean() * 100),
        "event_F1@0.5": float(event_f1.mean() * 100),
    }


def score_records(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ious = np.asarray(
        [temporal_set_iou(row["ground_truth"], row["prediction"]) for row in records],
        dtype=np.float64,
    )
    event_f1 = np.asarray(
        [
            event_f1_iou(row["ground_truth"], row["prediction"], 0.5)["f1"]
            for row in records
        ],
        dtype=np.float64,
    )
    return ious, event_f1


def paired_bootstrap_ci(
    differences: np.ndarray, samples: int, seed: int
) -> list[float] | None:
    if samples <= 0 or not len(differences):
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(samples, start + 1000)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[indices].mean(axis=1) * 100
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def compare(
    baseline: dict[int, dict],
    candidate: dict[int, dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())
        extra = sorted(candidate.keys() - baseline.keys())
        raise ValueError(f"Prediction indices differ; missing={missing[:5]}, extra={extra[:5]}")
    indices = sorted(baseline)
    for index in indices:
        if baseline[index]["ground_truth"] != candidate[index]["ground_truth"]:
            raise ValueError(f"Ground truth differs at index {index}")
    baseline_rows = [baseline[index] for index in indices]
    candidate_rows = [candidate[index] for index in indices]
    baseline_iou, baseline_f1 = score_records(baseline_rows)
    candidate_iou, candidate_f1 = score_records(candidate_rows)
    differences = candidate_iou - baseline_iou
    oracle_mask = candidate_iou > baseline_iou
    oracle_rows = [
        candidate_rows[i] if oracle_mask[i] else baseline_rows[i]
        for i in range(len(indices))
    ]
    oracle_iou, oracle_f1 = score_records(oracle_rows)
    return {
        "baseline": metrics(baseline_iou, baseline_f1),
        "candidate": metrics(candidate_iou, candidate_f1),
        "delta_mIoU": float(differences.mean() * 100),
        "delta_mIoU_bootstrap_95ci": paired_bootstrap_ci(
            differences, bootstrap_samples, seed
        ),
        "wins": int((differences > 1e-12).sum()),
        "ties": int((np.abs(differences) <= 1e-12).sum()),
        "losses": int((differences < -1e-12).sum()),
        "oracle": metrics(oracle_iou, oracle_f1),
    }


def main() -> None:
    args = parse_args()
    baseline = load_predictions(args.baseline)
    result = {
        name: compare(
            baseline,
            load_predictions(Path(path)),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for name, path in args.candidate
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
