#!/usr/bin/env python3
"""Fit and independently test a conservative relative-quality NOVA router."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "component_mean",
    "component_min",
    "complement_no",
    "duration_fraction",
    "interval_count",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--official", required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--feature",
        action="append",
        choices=FEATURE_NAMES,
        help="Feature subset for an ablation; defaults to all NOVA features",
    )
    parser.add_argument("--max-calibration-downside", type=float, default=0.0025)
    parser.add_argument("--min-heldout-capture", type=float, default=0.35)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def split_bucket(row: dict) -> int:
    key = f"{row.get('pair_id')}|{row.get('template')}|{row.get('variant', 0)}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 4


def vector(candidate: dict, feature_names=FEATURE_NAMES) -> np.ndarray:
    return np.asarray([candidate["features"][name] for name in feature_names], dtype=np.float64)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def predict_delta(
    row: dict,
    official: str,
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    feature_names=FEATURE_NAMES,
):
    base = vector(row["candidates"][official], feature_names)
    result = {official: 0.0}
    for name, candidate in row["candidates"].items():
        if name == official:
            continue
        delta = (vector(candidate, feature_names) - base - mean) / scale
        result[name] = float(np.r_[1.0, delta] @ weights)
    return result


def select(row: dict, predicted: dict[str, float], official: str, threshold: float) -> str:
    nonofficial = [name for name in predicted if name != official]
    if not nonofficial:
        return official
    best = max(nonofficial, key=predicted.get)
    return best if predicted[best] > threshold else official


def evaluate(rows, predictions, official, threshold):
    chosen_names = [select(row, score, official, threshold) for row, score in zip(rows, predictions)]
    base = np.asarray([row["candidates"][official]["iou"] for row in rows], dtype=np.float64)
    chosen = np.asarray(
        [row["candidates"][name]["iou"] for row, name in zip(rows, chosen_names)],
        dtype=np.float64,
    )
    all_iou = np.asarray(
        [[candidate["iou"] for candidate in row["candidates"].values()] for row in rows],
        dtype=np.float64,
    )
    oracle = all_iou.max(axis=1)
    denominator = float(oracle.mean() - base.mean())
    capture = float((chosen.mean() - base.mean()) / denominator) if denominator > 0 else 0.0
    downside = np.maximum(base - chosen, 0.0)
    upside = np.maximum(chosen - base, 0.0)
    counts = {name: chosen_names.count(name) for name in rows[0]["candidates"]}
    return {
        "rows": len(rows),
        "official_mIoU_percent": float(base.mean() * 100),
        "router_mIoU_percent": float(chosen.mean() * 100),
        "oracle_mIoU_percent": float(oracle.mean() * 100),
        "oracle_capture": capture,
        "mean_downside_points": float(downside.mean() * 100),
        "mean_upside_points": float(upside.mean() * 100),
        "regressed_rows": int((chosen < base).sum()),
        "improved_rows": int((chosen > base).sum()),
        "selection_counts": counts,
    }


def main():
    args = parse_args()
    feature_names = tuple(args.feature or FEATURE_NAMES)
    rows = load_jsonl(args.features)
    if not rows or args.official not in rows[0]["candidates"]:
        raise ValueError("Feature rows are empty or official candidate is missing")
    partitions = {
        "train": [row for row in rows if split_bucket(row) in (0, 1)],
        "calibration": [row for row in rows if split_bucket(row) == 2],
        "heldout": [row for row in rows if split_bucket(row) == 3],
    }
    if any(not value for value in partitions.values()):
        raise ValueError({name: len(value) for name, value in partitions.items()})

    x_train = []
    y_train = []
    for row in partitions["train"]:
        official_candidate = row["candidates"][args.official]
        base_x = vector(official_candidate, feature_names)
        base_y = float(official_candidate["iou"])
        for name, candidate in row["candidates"].items():
            if name == args.official:
                continue
            x_train.append(vector(candidate, feature_names) - base_x)
            y_train.append(float(candidate["iou"]) - base_y)
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x_train - mean) / scale

    best = None
    for alpha in (0.1, 1.0, 10.0, 100.0):
        weights = fit_ridge(normalized, y_train, alpha)
        calibration_predictions = [
            predict_delta(row, args.official, mean, scale, weights, feature_names)
            for row in partitions["calibration"]
        ]
        values = sorted(
            {score for prediction in calibration_predictions for name, score in prediction.items() if name != args.official}
        )
        thresholds = [float("inf"), *values]
        for threshold in thresholds:
            metrics = evaluate(
                partitions["calibration"], calibration_predictions, args.official, threshold
            )
            if metrics["mean_downside_points"] > args.max_calibration_downside * 100:
                continue
            key = (metrics["router_mIoU_percent"], threshold, -alpha)
            if best is None or key > best[0]:
                best = (key, alpha, threshold, weights, metrics)
    if best is None:
        raise RuntimeError("No threshold satisfied the calibration downside constraint")

    _, alpha, threshold, weights, calibration_metrics = best
    heldout_predictions = [
        predict_delta(row, args.official, mean, scale, weights, feature_names)
        for row in partitions["heldout"]
    ]
    heldout_metrics = evaluate(partitions["heldout"], heldout_predictions, args.official, threshold)
    candidate_means = {
        name: float(np.mean([row["candidates"][name]["iou"] for row in partitions["heldout"]]) * 100)
        for name in rows[0]["candidates"]
    }
    gate_passed = (
        heldout_metrics["oracle_capture"] >= args.min_heldout_capture
        and heldout_metrics["router_mIoU_percent"] > max(candidate_means.values())
    )
    model = {
        "method": "relative_ridge_with_official_fallback",
        "feature_names": list(feature_names),
        "official": args.official,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "alpha": alpha,
        "threshold": threshold,
        "split": "sha256(pair_id|template|variant) mod 4; train={0,1}, calibration=2, heldout=3",
    }
    report = {
        "status": "independent_development_only",
        "partition_sizes": {name: len(value) for name, value in partitions.items()},
        "calibration": calibration_metrics,
        "heldout": heldout_metrics,
        "heldout_candidate_mIoU_percent": candidate_means,
        "gate": {
            "min_heldout_capture": args.min_heldout_capture,
            "requires_router_above_best_single": True,
            "passed": gate_passed,
        },
        "model": model,
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
