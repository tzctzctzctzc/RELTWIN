#!/usr/bin/env python3
"""Audit a rejected NOVA-Safe pilot without changing the frozen router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova_safe import load_jsonl, sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def summarize_predictions(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot audit an empty prediction file")
    benchmark = str(rows[0]["benchmark"])
    if any(str(row["benchmark"]) != benchmark for row in rows):
        raise ValueError("One prediction file must contain exactly one benchmark")

    incumbent_scores = []
    selected_scores = []
    oracle_scores = []
    margins = []
    scored_actual_gains = []
    guarded_oracle_gain = 0.0
    eligible_oracle_gain = 0.0
    false_switches = 0

    for row in rows:
        incumbent_name = row["incumbent_name"]
        incumbent_iou = float(row["candidate_ious"][incumbent_name])
        selected_iou = float(row["candidate_ious"][row["selected_candidate"]])
        candidate_ious = {
            name: float(value)
            for name, value in row["candidate_ious"].items()
            if value is not None
        }
        incumbent_scores.append(incumbent_iou)
        selected_scores.append(selected_iou)
        oracle_scores.append(max(candidate_ious.values()))
        if row["selected_candidate"] != incumbent_name and selected_iou < incumbent_iou:
            false_switches += 1

        for name, score in row.get("candidate_scores", {}).items():
            candidate_iou = candidate_ious.get(name)
            if candidate_iou is None:
                continue
            margins.append(float(score["decision_margin"]))
            actual_gain = candidate_iou - incumbent_iou
            scored_actual_gains.append(actual_gain)
            eligible_oracle_gain += max(actual_gain, 0.0)
        for name in row.get("guarded_candidates", {}):
            candidate_iou = candidate_ious.get(name)
            if candidate_iou is not None:
                guarded_oracle_gain += max(candidate_iou - incumbent_iou, 0.0)

    incumbent = np.asarray(incumbent_scores, dtype=np.float64)
    selected = np.asarray(selected_scores, dtype=np.float64)
    oracle = np.asarray(oracle_scores, dtype=np.float64)
    actual = np.asarray(scored_actual_gains, dtype=np.float64)
    margin_array = np.asarray(margins, dtype=np.float64)
    if len(actual) >= 2 and np.std(actual) > 0 and np.std(margin_array) > 0:
        correlation = float(np.corrcoef(margin_array, actual)[0, 1])
    else:
        correlation = None

    oracle_headroom = float((oracle - incumbent).mean() * 100)
    selected_gain = float((selected - incumbent).mean() * 100)
    captured = None if oracle_headroom <= 0 else selected_gain / oracle_headroom
    return {
        "benchmark": benchmark,
        "rows": len(rows),
        "incumbent_mIoU_percent": float(incumbent.mean() * 100),
        "router_mIoU_percent": float(selected.mean() * 100),
        "router_gain_points": selected_gain,
        "candidate_oracle_mIoU_percent": float(oracle.mean() * 100),
        "candidate_oracle_headroom_points": oracle_headroom,
        "oracle_improvable_rows": int((oracle > incumbent + 1e-12).sum()),
        "oracle_headroom_capture_fraction": captured,
        "scored_candidate_pairs": len(actual),
        "positive_scored_candidate_pairs": int((actual > 1e-12).sum()),
        "eligible_positive_headroom_points": float(eligible_oracle_gain / len(rows) * 100),
        "guarded_positive_headroom_points": float(guarded_oracle_gain / len(rows) * 100),
        "decision_margin_quantiles": _quantiles(margins),
        "decision_margin_actual_gain_correlation": correlation,
        "false_switches": false_switches,
    }


def build_audit(summaries: list[dict], development_report: dict) -> dict:
    weak_ranking = all(
        summary["decision_margin_actual_gain_correlation"] is not None
        and abs(summary["decision_margin_actual_gain_correlation"]) < 0.1
        for summary in summaries
    )
    unused_headroom = all(
        summary["candidate_oracle_headroom_points"] > 0.1
        and (summary["oracle_headroom_capture_fraction"] or 0.0) <= 0.0
        for summary in summaries
    )
    chosen = development_report.get("chosen", development_report.get("selected_mode", {}))
    return {
        "decision": "rollback_and_rethink",
        "baseline_or_default_pointer_modified": False,
        "full_evaluation_authorized": False,
        "development_choice": chosen,
        "benchmarks": summaries,
        "diagnosis": {
            "weak_cross_domain_ranking": weak_ranking,
            "material_unused_candidate_headroom": unused_headroom,
            "primary_cause": "candidate_family_and_domain_mismatch" if weak_ranking and unused_headroom else "inconclusive",
            "threshold_only_change_rejected": weak_ranking,
            "explanation": (
                "The frozen Keep-only margin does not rank deployment-family gains. "
                "Loosening its threshold would increase unsafe switches without targeting rescues."
            ),
        },
        "next_single_factor_change": {
            "factor": "candidate_family_matched_development_data",
            "action": (
                "Generate independent development pairs with the same incumbent/challenger families "
                "used at deployment, then refit all three feature modes."
            ),
            "additional_gate": (
                "Reject a development pass supported by only one effective switch; require positive "
                "held-out gain on at least two sources before drawing a new disjoint 1000+100 pilot."
            ),
            "pilot_reuse": "Observed pilot audio groups become development-only and cannot enter the next pilot.",
        },
    }


def main():
    args = parse_args()
    summaries = [summarize_predictions(load_jsonl(path)) for path in args.predictions]
    if len({summary["benchmark"] for summary in summaries}) != len(summaries):
        raise ValueError("Each prediction input must represent a distinct benchmark")
    development_report = json.loads(args.development_report.read_text(encoding="utf-8"))
    audit = build_audit(summaries, development_report)
    audit["inputs"] = {
        "predictions": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.predictions
        ],
        "development_report": {
            "path": str(args.development_report.resolve()),
            "sha256": sha256_file(args.development_report),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
