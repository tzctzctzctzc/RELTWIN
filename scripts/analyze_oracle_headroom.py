#!/usr/bin/env python3
"""Measure diagnostic Oracle headroom over aligned prediction files.

The Oracle selects the prediction with the largest ground-truth IoU for each
benchmark row. It is an upper bound, not a deployable selector or SOTA result.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


IDENTITY_KEYS = ("index", "audio", "query", "ground_truth")


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def parse_group(value: str) -> tuple[str, list[str]]:
    name, separator, members = value.partition("=")
    parsed = [member for member in members.split(",") if member]
    if not separator or not name or not parsed:
        raise argparse.ArgumentTypeError("expected NAME=MODEL_A,MODEL_B,...")
    return name, parsed


def read_predictions(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no predictions in {path}")
    return rows


def aligned_iou(reference: list[dict], candidate: list[dict], path: Path) -> np.ndarray:
    if len(reference) != len(candidate):
        raise ValueError(f"row count mismatch for {path}")
    for expected, actual in zip(reference, candidate):
        if any(expected.get(key) != actual.get(key) for key in IDENTITY_KEYS):
            raise ValueError(f"identity mismatch at row {expected.get('index')} in {path}")
    return np.asarray([float(row["iou"]) for row in candidate], dtype=np.float64)


def metric_summary(values: np.ndarray) -> dict:
    return {
        "mIoU_percent": float(100 * values.mean()),
        "R1@0.3_percent": float(100 * (values >= 0.3).mean()),
        "R1@0.5_percent": float(100 * (values >= 0.5).mean()),
        "R1@0.7_percent": float(100 * (values >= 0.7).mean()),
    }


def oracle_summary(
    names: list[str], values: np.ndarray, baseline: np.ndarray
) -> tuple[dict, np.ndarray]:
    oracle = values.max(axis=0)
    maxima = np.isclose(values, oracle[None, :], rtol=0.0, atol=1e-12)
    unique = maxima.sum(axis=0) == 1
    winner_counts = {
        name: int((maxima[index] & unique).sum()) for index, name in enumerate(names)
    }
    delta = oracle - baseline
    result = {
        "members": names,
        "metrics": metric_summary(oracle),
        "delta_vs_baseline_mIoU_points": float(100 * delta.mean()),
        "rows_improved_vs_baseline": int((delta > 1e-12).sum()),
        "rows_regressed_vs_baseline": int((delta < -1e-12).sum()),
        "rows_tied_vs_baseline": int(np.isclose(delta, 0.0, atol=1e-12).sum()),
        "unique_winner_counts": winner_counts,
        "winner_tie_rows": int((~unique).sum()),
    }
    return result, oracle


def breakdown(
    mask: np.ndarray, baseline: np.ndarray, oracle: np.ndarray
) -> dict:
    count = int(mask.sum())
    if not count:
        return {"rows": 0}
    delta = oracle[mask] - baseline[mask]
    return {
        "rows": count,
        "baseline_mIoU_percent": float(100 * baseline[mask].mean()),
        "oracle_mIoU_percent": float(100 * oracle[mask].mean()),
        "headroom_points": float(100 * delta.mean()),
        "rows_improved": int((delta > 1e-12).sum()),
    }


def error_breakdowns(
    reference: list[dict], baseline: np.ndarray, oracle: np.ndarray
) -> dict:
    durations = np.asarray([float(row["duration_seconds"]) for row in reference])
    multi = np.asarray([len(row["ground_truth"]) > 1 for row in reference])
    return {
        "target_cardinality": {
            "single_interval": breakdown(~multi, baseline, oracle),
            "multi_interval": breakdown(multi, baseline, oracle),
        },
        "duration_seconds": {
            "le_30": breakdown(durations <= 30, baseline, oracle),
            "gt_30_le_60": breakdown((durations > 30) & (durations <= 60), baseline, oracle),
            "gt_60": breakdown(durations > 60, baseline, oracle),
        },
        "baseline_iou": {
            "lt_0.3": breakdown(baseline < 0.3, baseline, oracle),
            "0.3_to_0.5": breakdown((baseline >= 0.3) & (baseline < 0.5), baseline, oracle),
            "0.5_to_0.7": breakdown((baseline >= 0.5) & (baseline < 0.7), baseline, oracle),
            "ge_0.7": breakdown(baseline >= 0.7, baseline, oracle),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=parse_named_path, required=True)
    parser.add_argument("--model", type=parse_named_path, action="append", required=True)
    parser.add_argument("--group", type=parse_group, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_name, baseline_path = args.baseline
    model_paths = dict(args.model)
    if len(model_paths) != len(args.model):
        raise ValueError("model names must be unique")
    groups = dict(args.group)
    if len(groups) != len(args.group):
        raise ValueError("group names must be unique")
    missing = {
        member for members in groups.values() for member in members if member not in model_paths
    }
    if missing:
        raise ValueError(f"unknown group members: {sorted(missing)}")

    reference = read_predictions(baseline_path)
    baseline = np.asarray([float(row["iou"]) for row in reference], dtype=np.float64)
    arrays = {
        name: aligned_iou(reference, read_predictions(path), path)
        for name, path in model_paths.items()
    }
    result = {
        "claim_scope": "diagnostic_only_ground_truth_oracle_not_deployable",
        "baseline": {
            "name": baseline_name,
            "path": baseline_path.as_posix(),
            "metrics": metric_summary(baseline),
        },
        "records": len(reference),
        "models": {
            name: {"path": model_paths[name].as_posix(), "metrics": metric_summary(values)}
            for name, values in arrays.items()
        },
        "group_oracles": {},
    }
    group_arrays: dict[str, np.ndarray] = {}
    for group_name, members in groups.items():
        values = np.stack([arrays[member] for member in members])
        summary, oracle = oracle_summary(members, values, baseline)
        result["group_oracles"][group_name] = summary
        group_arrays[group_name] = oracle

    all_names = list(arrays)
    all_summary, all_oracle = oracle_summary(
        all_names, np.stack([arrays[name] for name in all_names]), baseline
    )
    result["all_nonbaseline_oracle"] = all_summary
    baseline_plus = np.vstack([baseline, *[arrays[name] for name in all_names]])
    full_summary, full_oracle = oracle_summary(
        [baseline_name, *all_names], baseline_plus, baseline
    )
    result["baseline_plus_all_oracle"] = full_summary
    result["baseline_plus_all_oracle"]["breakdowns"] = error_breakdowns(
        reference, baseline, full_oracle
    )

    comparisons = {}
    for first, second in combinations(groups, 2):
        left, right = group_arrays[first], group_arrays[second]
        comparisons[f"{first}_vs_{second}"] = {
            f"{first}_wins": int((left > right + 1e-12).sum()),
            f"{second}_wins": int((right > left + 1e-12).sum()),
            "ties": int(np.isclose(left, right, atol=1e-12).sum()),
            f"mean_{second}_minus_{first}_points": float(100 * (right - left).mean()),
        }
    result["group_oracle_comparisons"] = comparisons

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
