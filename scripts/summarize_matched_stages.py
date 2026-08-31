#!/usr/bin/env python3
"""Compare two matched multi-seed stages on aligned SpotSound predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def aligned(left: list[dict], right: list[dict], path: Path) -> tuple[np.ndarray, np.ndarray]:
    if len(left) != len(right):
        raise ValueError(f"Record count mismatch in {path}")
    identity = ("index", "audio", "query", "ground_truth")
    for first, second in zip(left, right):
        if any(first.get(key) != second.get(key) for key in identity):
            raise ValueError(f"Identity mismatch at row {first.get('index')} in {path}")
    return (
        np.asarray([float(row["iou"]) for row in left]),
        np.asarray([float(row["iou"]) for row in right]),
    )


def interval(values: np.ndarray) -> dict:
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "p_nonpositive": float((values <= 0).mean()),
    }


def summarize_delta(rng: np.random.Generator, delta: np.ndarray, samples: int) -> dict:
    seeds, records = delta.shape
    per_seed = delta.mean(axis=1)
    record_indices = rng.integers(0, records, size=(samples, records))
    paired_record = delta.mean(axis=0)[record_indices].mean(axis=1)
    hierarchical = np.empty(samples)
    for sample in range(samples):
        selected_seeds = rng.integers(0, seeds, size=seeds)
        selected_records = rng.integers(0, records, size=records)
        hierarchical[sample] = delta[selected_seeds][:, selected_records].mean()
    return {
        "per_seed_delta": [float(value) for value in per_seed],
        "mean_delta": float(per_seed.mean()),
        "paired_record": interval(paired_record),
        "hierarchical_seed_record": interval(hierarchical),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if len(args.source) != len(args.target):
        raise ValueError("--source and --target require the same number of seeds")
    source_rows = [read(path) for path in args.source]
    target_rows = [read(path) for path in args.target]
    pairs = [aligned(left, right, path) for left, right, path in zip(source_rows, target_rows, args.target)]
    sources = np.stack([pair[0] for pair in pairs])
    targets = np.stack([pair[1] for pair in pairs])
    specs = {"mIoU": None, "R1@0.3": 0.3, "R1@0.5": 0.5}
    rng = np.random.default_rng(args.seed)
    metrics = {}
    for name, threshold in specs.items():
        if threshold is None:
            source_metric, target_metric = sources, targets
        else:
            source_metric = (sources >= threshold).astype(float)
            target_metric = (targets >= threshold).astype(float)
        metrics[name] = summarize_delta(rng, target_metric - source_metric, args.samples)
    result = {
        "source_predictions": [path.as_posix() for path in args.source],
        "target_predictions": [path.as_posix() for path in args.target],
        "records": int(sources.shape[1]),
        "seeds": int(sources.shape[0]),
        "bootstrap_samples": args.samples,
        "bootstrap_seed": args.seed,
        "delta_target_minus_source": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
