#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "query_mIoU",
    "query_R1@0.5",
    "pair_acc_0.5",
    "swap_error_rate",
    "wrong_window_preference_rate",
)
BOOTSTRAP_METRICS = ("mIoU", "R1@0.5", "PairAcc@0.5", "SwapErrorReduction")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", type=Path, required=True)
    parser.add_argument("--target", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def bootstrap_delta(rng, delta, samples):
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    values = delta[indices].mean(axis=1)
    low, high = np.quantile(values, (0.025, 0.975))
    return {
        "mean": float(values.mean()),
        "ci95": [float(low), float(high)],
        "p_nonpositive": float((values <= 0).mean()),
    }


def main():
    args = parse_args()
    if len(args.source) != len(args.target):
        raise ValueError("--source and --target must contain the same number of seeds")
    source = [json.loads(path.read_text(encoding="utf-8")) for path in args.source]
    target = [json.loads(path.read_text(encoding="utf-8")) for path in args.target]
    rng = np.random.default_rng(args.seed)

    per_seed = []
    for seed, (baseline, method) in enumerate(zip(source, target)):
        if baseline["query_count"] != method["query_count"]:
            raise ValueError(f"seed {seed} query counts do not match")
        metrics = {
            key: {
                "source": float(baseline[key]),
                "target": float(method[key]),
                "delta": float(method[key] - baseline[key]),
            }
            for key in METRICS
        }
        query_iou_delta = np.asarray(
            [
                right["iou_correct"] - left["iou_correct"]
                for left, right in zip(baseline["query_rows"], method["query_rows"])
            ],
            dtype=float,
        )
        query_r1_delta = np.asarray(
            [
                float(right["iou_correct"] >= 0.5) - float(left["iou_correct"] >= 0.5)
                for left, right in zip(baseline["query_rows"], method["query_rows"])
            ],
            dtype=float,
        )
        pair_acc_delta = np.asarray(
            [
                float(right["pair_acc_0.5"]) - float(left["pair_acc_0.5"])
                for left, right in zip(baseline["pair_rows"], method["pair_rows"])
            ],
            dtype=float,
        )
        swap_reduction = np.asarray(
            [
                float(left["swap_error"]) - float(right["swap_error"])
                for left, right in zip(baseline["pair_rows"], method["pair_rows"])
            ],
            dtype=float,
        )
        arrays = {
            "mIoU": query_iou_delta,
            "R1@0.5": query_r1_delta,
            "PairAcc@0.5": pair_acc_delta,
            "SwapErrorReduction": swap_reduction,
        }
        seed_bootstrap = {}
        for name, delta in arrays.items():
            stats = bootstrap_delta(rng, delta, args.bootstrap_samples)
            seed_bootstrap[name] = stats
        per_seed.append(
            {
                "seed": seed,
                "query_count": baseline["query_count"],
                "relation_pair_count": baseline["relation_pair_count"],
                "metrics": metrics,
                "iou_wins_ties_losses": [
                    int((query_iou_delta > 0).sum()),
                    int((query_iou_delta == 0).sum()),
                    int((query_iou_delta < 0).sum()),
                ],
                "bootstrap": seed_bootstrap,
            }
        )

    aggregate = {}
    for key in METRICS:
        left = np.asarray([item[key] for item in source], dtype=float)
        right = np.asarray([item[key] for item in target], dtype=float)
        aggregate[key] = {
            "source_mean": float(left.mean()),
            "source_std": float(left.std(ddof=1)),
            "target_mean": float(right.mean()),
            "target_std": float(right.std(ddof=1)),
            "mean_delta": float((right - left).mean()),
            "per_seed_delta": [float(value) for value in right - left],
        }

    stratified = {}
    rng = np.random.default_rng(args.seed)
    deltas_by_name = {name: [] for name in BOOTSTRAP_METRICS}
    for baseline, method in zip(source, target):
        deltas_by_name["mIoU"].append(
            np.asarray([r["iou_correct"] - l["iou_correct"] for l, r in zip(baseline["query_rows"], method["query_rows"])])
        )
        deltas_by_name["R1@0.5"].append(
            np.asarray([float(r["iou_correct"] >= 0.5) - float(l["iou_correct"] >= 0.5) for l, r in zip(baseline["query_rows"], method["query_rows"])])
        )
        deltas_by_name["PairAcc@0.5"].append(
            np.asarray([float(r["pair_acc_0.5"]) - float(l["pair_acc_0.5"]) for l, r in zip(baseline["pair_rows"], method["pair_rows"])])
        )
        deltas_by_name["SwapErrorReduction"].append(
            np.asarray([float(l["swap_error"]) - float(r["swap_error"]) for l, r in zip(baseline["pair_rows"], method["pair_rows"])])
        )
    for name, seed_deltas in deltas_by_name.items():
        parts = []
        for delta in seed_deltas:
            indices = rng.integers(0, len(delta), size=(args.bootstrap_samples, len(delta)))
            parts.append(delta[indices].mean(axis=1))
        values = np.mean(parts, axis=0)
        low, high = np.quantile(values, (0.025, 0.975))
        stratified[name] = {
            "mean": float(values.mean()),
            "ci95": [float(low), float(high)],
            "p_nonpositive": float((values <= 0).mean()),
        }

    result = {
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "stratified_bootstrap": stratified,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "stratified_bootstrap": stratified}, indent=2))


if __name__ == "__main__":
    main()
