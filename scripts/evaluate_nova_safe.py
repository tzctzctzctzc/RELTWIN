#!/usr/bin/env python3
"""Score NOVA-Safe predictions, bootstrap by audio, and emit an auditable gate report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from interval_metrics import event_f1_iou, normalize_intervals, temporal_set_iou
from nova_safe import CATASTROPHIC_DROP, load_jsonl, sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--expected-incumbent-miou", type=float)
    parser.add_argument("--expected-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _bootstrap_interval(rows: list[dict], samples: int, seed: int) -> list[float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["audio_group"])].append(row["delta_iou"])
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled = rng.choice(names, size=len(names), replace=True)
        values = [value for name in sampled for value in groups[str(name)]]
        estimates.append(float(np.mean(values) * 100))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def _failure_category(row: dict) -> str | None:
    switched = row["selected_candidate"] != row["incumbent_name"]
    if row["delta_iou"] <= -CATASTROPHIC_DROP:
        return "catastrophic_wrong_window"
    if switched and row["delta_iou"] < -1e-12:
        return "false_switch"
    candidate_ious = [value for value in row.get("candidate_ious", {}).values() if value is not None]
    if not switched and candidate_ious and max(candidate_ious) > row["incumbent_iou"] + 1e-12:
        return "missed_rescue"
    if row.get("abstain_reason") == "all_challengers_guarded":
        return "structural_abstain"
    return None


def main():
    args = parse_args()
    raw_rows = load_jsonl(args.predictions)
    seen = set()
    rows = []
    event_f1 = []
    for row in raw_rows:
        key = (row["benchmark"], int(row["source_index"]))
        if key in seen:
            raise ValueError(f"Duplicate prediction row: {key}")
        seen.add(key)
        duration = float(row["duration_seconds"])
        ground_truth = normalize_intervals(row["ground_truth"], duration)
        prediction = normalize_intervals(row["prediction"], duration)
        incumbent_prediction = normalize_intervals(row["incumbent_prediction"], duration)
        selected_iou = temporal_set_iou(ground_truth, prediction)
        incumbent_iou = temporal_set_iou(ground_truth, incumbent_prediction)
        audited = dict(row)
        audited["iou"] = selected_iou
        audited["incumbent_iou"] = incumbent_iou
        audited["delta_iou"] = selected_iou - incumbent_iou
        audited["failure_category"] = None
        rows.append(audited)
        event_f1.append(event_f1_iou(ground_truth, prediction, 0.5)["f1"])
    if not rows:
        raise ValueError("No predictions to evaluate")
    for row in rows:
        row["failure_category"] = _failure_category(row)
    selected = np.asarray([row["iou"] for row in rows], dtype=np.float64)
    incumbent = np.asarray([row["incumbent_iou"] for row in rows], dtype=np.float64)
    delta = selected - incumbent
    incumbent_miou = float(incumbent.mean() * 100)
    baseline_reproduced = True
    if args.expected_incumbent_miou is not None:
        baseline_reproduced = abs(incumbent_miou - args.expected_incumbent_miou) <= args.expected_tolerance
    catastrophic = int((delta <= -CATASTROPHIC_DROP).sum())
    passed = float(delta.mean()) > 0.0 and catastrophic == 0 and baseline_reproduced
    report = {
        "benchmark": rows[0]["benchmark"],
        "rows": len(rows),
        "audio_groups": len({row["audio_group"] for row in rows}),
        "incumbent_mIoU_percent": incumbent_miou,
        "router_mIoU_percent": float(selected.mean() * 100),
        "delta_mIoU_points": float(delta.mean() * 100),
        "delta_mIoU_audio_group_bootstrap_95ci": _bootstrap_interval(rows, args.bootstrap_samples, args.seed),
        "R1@0.3_percent": float((selected >= 0.3).mean() * 100),
        "R1@0.5_percent": float((selected >= 0.5).mean() * 100),
        "R1@0.7_percent": float((selected >= 0.7).mean() * 100),
        "event_F1@0.5_percent": float(np.mean(event_f1) * 100),
        "improved_rows": int((delta > 1e-12).sum()),
        "tied_rows": int((np.abs(delta) <= 1e-12).sum()),
        "regressed_rows": int((delta < -1e-12).sum()),
        "new_catastrophic_regressions": catastrophic,
        "selection_counts": dict(Counter(row["selected_candidate"] for row in rows)),
        "failure_counts": dict(Counter(row["failure_category"] for row in rows if row["failure_category"])),
        "baseline_reproduced": baseline_reproduced,
        "expected_incumbent_mIoU_percent": args.expected_incumbent_miou,
        "promotion_gate": {
            "requires_positive_delta": True,
            "requires_zero_new_catastrophic_regressions": True,
            "passed": passed,
        },
        "predictions_sha256": sha256_file(args.predictions),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
    }
    failures = [row for row in rows if row["failure_category"]]
    failures.sort(key=lambda row: (row["delta_iou"], row["source_index"]))
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    with args.failures.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["failures_sha256"] = sha256_file(args.failures)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

