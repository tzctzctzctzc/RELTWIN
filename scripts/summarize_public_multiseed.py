#!/usr/bin/env python3
"""Summarize fixed-seed SpotSound-Bench predictions against one baseline.

The script reports both a paired-record bootstrap and a conservative
hierarchical bootstrap that resamples training seeds and benchmark records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"No predictions in {path}")
    return rows


def aligned_iou(reference: list[dict], candidate: list[dict], path: Path) -> np.ndarray:
    if len(reference) != len(candidate):
        raise ValueError(f"Record count mismatch for {path}: {len(candidate)} != {len(reference)}")
    values = []
    for expected, actual in zip(reference, candidate):
        identity = ("index", "audio", "query", "ground_truth")
        if any(actual.get(key) != expected.get(key) for key in identity):
            raise ValueError(f"Record identity mismatch at index {expected.get('index')} in {path}")
        values.append(float(actual["iou"]))
    return np.asarray(values, dtype=np.float64)


def interval(values: np.ndarray) -> dict:
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "p_nonpositive": float((values <= 0).mean()),
    }


def bootstrap(
    rng: np.random.Generator,
    baseline: np.ndarray,
    methods: np.ndarray,
    samples: int,
) -> dict:
    seeds, records = methods.shape
    averaged_delta = methods.mean(axis=0) - baseline
    record_indices = rng.integers(0, records, size=(samples, records))
    paired_record = averaged_delta[record_indices].mean(axis=1)

    hierarchical = np.empty(samples, dtype=np.float64)
    chunk = 1_000
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        seed_indices = rng.integers(0, seeds, size=(size, seeds))
        record_indices = rng.integers(0, records, size=(size, records))
        for offset in range(size):
            selected = methods[seed_indices[offset]][:, record_indices[offset]]
            hierarchical[start + offset] = selected.mean() - baseline[record_indices[offset]].mean()
    return {
        "paired_record": interval(paired_record),
        "hierarchical_seed_record": interval(hierarchical),
    }


def metric_summary(baseline: np.ndarray, methods: np.ndarray, threshold: float | None) -> dict:
    if threshold is None:
        base = baseline
        candidates = methods
    else:
        base = (baseline >= threshold).astype(np.float64)
        candidates = (methods >= threshold).astype(np.float64)
    per_seed = candidates.mean(axis=1)
    return {
        "baseline": float(base.mean()),
        "method_per_seed": [float(value) for value in per_seed],
        "method_mean": float(per_seed.mean()),
        "mean_delta": float(per_seed.mean() - base.mean()),
    }


def transformed_metric(
    baseline: np.ndarray, methods: np.ndarray, threshold: float | None
) -> tuple[np.ndarray, np.ndarray]:
    if threshold is None:
        return baseline, methods
    return (baseline >= threshold).astype(np.float64), (methods >= threshold).astype(np.float64)


def main() -> None:
    args = parse_args()
    baseline_rows = read_predictions(args.baseline)
    baseline = np.asarray([float(row["iou"]) for row in baseline_rows], dtype=np.float64)
    methods = np.stack(
        [aligned_iou(baseline_rows, read_predictions(path), path) for path in args.method]
    )
    rng = np.random.default_rng(args.seed)
    metric_specs = {"mIoU": None, "R1@0.3": 0.3, "R1@0.5": 0.5}
    result = {
        "baseline_predictions": args.baseline.as_posix(),
        "method_predictions": [path.as_posix() for path in args.method],
        "records": int(baseline.size),
        "seeds": int(methods.shape[0]),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "metrics_fraction": {
            name: metric_summary(baseline, methods, threshold)
            for name, threshold in metric_specs.items()
        },
        "delta_bootstrap": {},
    }
    for name, threshold in metric_specs.items():
        metric_baseline, metric_methods = transformed_metric(baseline, methods, threshold)
        result["delta_bootstrap"][name] = bootstrap(
            rng, metric_baseline, metric_methods, args.samples
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
