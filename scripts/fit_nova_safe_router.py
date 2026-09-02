#!/usr/bin/env python3
"""Fit NOVA-Safe with leave-one-source-out selection and group bootstrap uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nova_safe import (
    BOOTSTRAP_QUANTILE,
    FEATURE_SETS,
    SCHEMA_VERSION,
    ensemble_prediction,
    evaluate_selection,
    feature_vector,
    fit_bootstrap_ensemble,
    hard_guard,
    load_jsonl,
    sha256_file,
    training_pairs,
)


ALPHAS = (0.1, 1.0, 10.0, 100.0)
COMPLEXITY = {"geometry": 0, "keep": 1, "full": 2}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, action="append", required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--cv-bootstrap-models", type=int, default=64)
    parser.add_argument("--final-bootstrap-models", type=int, default=128)
    parser.add_argument("--feature-mode", choices=tuple(FEATURE_SETS), action="append")
    return parser.parse_args()


def _excluded_groups(paths: list[Path]) -> set[str]:
    groups = set()
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            source = row.get("source")
            audio = row.get("audio_group", row.get("audio", row.get("audio_path")))
            if source is not None and audio is not None:
                groups.add(f"{source}|{Path(str(audio)).name}")
    return groups


def _model_payload(ensemble: dict, feature_names, threshold: float) -> dict:
    return {
        "feature_names": list(feature_names),
        "mean": ensemble["mean"].tolist(),
        "scale": ensemble["scale"].tolist(),
        "weights": ensemble["weights"].tolist(),
        "threshold": float(threshold),
    }


def _fold_scores(rows: list[dict], ensemble: dict, feature_names) -> list[dict[str, float]]:
    result = []
    for row in rows:
        incumbent_name = row["incumbent"]
        incumbent = row["candidates"][incumbent_name]
        scores = {}
        for name, challenger in row["candidates"].items():
            if name == incumbent_name:
                continue
            if hard_guard(incumbent, challenger, float(row["duration_seconds"])):
                continue
            _, lower = ensemble_prediction(ensemble, feature_vector(row, name, feature_names))
            scores[name] = lower
        result.append(scores)
    return result


def _selections(rows: list[dict], scores: list[dict[str, float]], threshold: float) -> list[str]:
    selected = []
    for row, row_scores in zip(rows, scores):
        incumbent = row["incumbent"]
        if not row_scores:
            selected.append(incumbent)
            continue
        best = max(row_scores, key=lambda name: (row_scores[name], name))
        selected.append(best if row_scores[best] > threshold else incumbent)
    return selected


def _fit_mode(rows: list[dict], mode: str, args) -> dict:
    feature_names = FEATURE_SETS[mode]
    sources = sorted({str(row["source"]) for row in rows})
    if len(sources) < 2:
        raise ValueError("Leave-one-source-out fitting requires at least two development sources")
    best = None
    alpha_reports = []
    for alpha_index, alpha in enumerate(ALPHAS):
        folds = {}
        threshold_values = {0.0}
        for source_index, source in enumerate(sources):
            train_rows = [row for row in rows if str(row["source"]) != source]
            heldout_rows = [row for row in rows if str(row["source"]) == source]
            pairs = training_pairs(train_rows, feature_names)
            ensemble = fit_bootstrap_ensemble(
                pairs,
                alpha,
                args.seed + 10000 * alpha_index + source_index,
                args.cv_bootstrap_models,
            )
            scores = _fold_scores(heldout_rows, ensemble, feature_names)
            threshold_values.update(score for values in scores for score in values.values() if score >= 0.0)
            folds[source] = (heldout_rows, scores)
        threshold_reports = []
        for threshold in sorted(threshold_values):
            source_metrics = {}
            all_rows = []
            all_selections = []
            for source in sources:
                heldout_rows, scores = folds[source]
                selections = _selections(heldout_rows, scores, threshold)
                source_metrics[source] = evaluate_selection(heldout_rows, selections)
                all_rows.extend(heldout_rows)
                all_selections.extend(selections)
            overall = evaluate_selection(all_rows, all_selections)
            worst_delta = min(metrics["delta_mIoU_points"] for metrics in source_metrics.values())
            catastrophic = sum(
                metrics["new_catastrophic_regressions"] for metrics in source_metrics.values()
            )
            passed = worst_delta > 0.0 and catastrophic == 0
            report = {
                "threshold": threshold,
                "worst_source_delta_points": worst_delta,
                "new_catastrophic_regressions": catastrophic,
                "passed": passed,
                "sources": source_metrics,
                "overall": overall,
            }
            threshold_reports.append(report)
            key = (passed, worst_delta, overall["delta_mIoU_points"], threshold, -alpha)
            if best is None or key > best[0]:
                best = (key, alpha, threshold, report)
        alpha_reports.append(
            {
                "alpha": alpha,
                "best": max(
                    threshold_reports,
                    key=lambda item: (
                        item["passed"],
                        item["worst_source_delta_points"],
                        item["overall"]["delta_mIoU_points"],
                        item["threshold"],
                    ),
                ),
            }
        )
    _, alpha, threshold, selected_report = best
    return {
        "mode": mode,
        "feature_names": list(feature_names),
        "alpha": alpha,
        "threshold": threshold,
        "passed": selected_report["passed"],
        "worst_source_delta_points": selected_report["worst_source_delta_points"],
        "selected_cv": selected_report,
        "alpha_reports": alpha_reports,
    }


def choose_mode(reports: list[dict]) -> dict | None:
    passed = [report for report in reports if report["passed"]]
    if not passed:
        return None
    best_worst = max(report["worst_source_delta_points"] for report in passed)
    contenders = [
        report for report in passed
        if best_worst - report["worst_source_delta_points"] <= 0.05
    ]
    return max(
        contenders,
        key=lambda report: (
            -COMPLEXITY[report["mode"]],
            report["worst_source_delta_points"],
            report["selected_cv"]["overall"]["delta_mIoU_points"],
        ),
    )


def main():
    args = parse_args()
    rows = [row for path in args.features for row in load_jsonl(path)]
    if not rows:
        raise ValueError("No development feature rows")
    invalid_schema = [row.get("feature_schema_version") for row in rows if row.get("feature_schema_version") != SCHEMA_VERSION]
    if invalid_schema:
        raise ValueError(f"Unexpected feature schema: {sorted(set(map(str, invalid_schema)))}")
    groups = [f"{row['source']}|{row['audio_group']}" for row in rows]
    excluded = _excluded_groups(args.exclude_manifest)
    leaked = sorted(set(groups) & excluded)
    if leaked:
        raise ValueError(f"Pilot/development audio leakage: {leaked[:5]}")
    modes = args.feature_mode or ["geometry", "keep", "full"]
    reports = [_fit_mode(rows, mode, args) for mode in modes]
    chosen = choose_mode(reports)
    report = {
        "status": "independent_development_only",
        "feature_schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "sources": {source: sum(row["source"] == source for row in rows) for source in sorted(set(row["source"] for row in rows))},
        "audio_groups": len(set(groups)),
        "input_hashes": {str(path.resolve()): sha256_file(path) for path in args.features},
        "bootstrap": {
            "quantile": BOOTSTRAP_QUANTILE,
            "cv_models": args.cv_bootstrap_models,
            "final_models": args.final_bootstrap_models,
            "seed": args.seed,
        },
        "mode_reports": reports,
        "selection_rule": "highest worst-source delta; within 0.05 point choose cheaper mode",
        "gate": {"passed": chosen is not None, "chosen_mode": chosen["mode"] if chosen else None},
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if chosen is None:
        print(json.dumps(report, indent=2))
        raise SystemExit(2)
    feature_names = FEATURE_SETS[chosen["mode"]]
    final_pairs = training_pairs(rows, feature_names)
    ensemble = fit_bootstrap_ensemble(
        final_pairs,
        chosen["alpha"],
        args.seed + 900000,
        args.final_bootstrap_models,
    )
    model = {
        "method": "nova_safe_v2_group_bootstrap_relative_ridge",
        "feature_schema_version": SCHEMA_VERSION,
        "feature_mode": chosen["mode"],
        "feature_names": list(feature_names),
        "alpha": chosen["alpha"],
        "threshold": chosen["threshold"],
        "bootstrap_quantile": BOOTSTRAP_QUANTILE,
        "equivalence_iou": 0.995,
        "catastrophic_drop": 0.5,
        "seed": args.seed,
        "development_sources": sorted(set(row["source"] for row in rows)),
        "input_hashes": report["input_hashes"],
        **_model_payload(ensemble, feature_names, chosen["threshold"]),
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    report["model_sha256"] = sha256_file(args.output_model)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

