#!/usr/bin/env python3
"""Score TEMPO audio grounding from resumable SpotSound-format predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interval_metrics import interval_iou, normalize_intervals, parse_canonical_intervals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def symmetric_miou(reference, prediction) -> float:
    if not reference or not prediction:
        return 0.0
    recall = sum(max(interval_iou(gt, pred) for pred in prediction) for gt in reference) / len(reference)
    precision = sum(max(interval_iou(pred, gt) for gt in reference) for pred in prediction) / len(prediction)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def greedy_matches(reference, prediction):
    candidates = sorted(
        (
            (interval_iou(gt, pred), gt_index, pred_index)
            for gt_index, gt in enumerate(reference)
            for pred_index, pred in enumerate(prediction)
        ),
        reverse=True,
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    for iou, gt_index, pred_index in candidates:
        if gt_index not in used_gt and pred_index not in used_pred:
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            matches.append((iou, gt_index, pred_index))
    return matches


def main() -> int:
    args = parse_args()
    rows = []
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status", "ok") == "ok":
                rows.append(row)
    if not rows:
        raise ValueError("No successful prediction records found")

    per_example_miou = []
    true_positive = total_reference = total_prediction = 0
    boundary_errors = []
    format_failures = 0
    multi_reference = multi_prediction = 0

    for row in rows:
        duration = float(row.get("duration_seconds", row.get("duration", 0.0)))
        reference = normalize_intervals(row.get("ground_truth", []), duration or None)
        prediction = normalize_intervals(row.get("prediction", []), duration or None)
        if "prediction" not in row:
            prediction = parse_canonical_intervals(row.get("raw_answer", ""), duration or None)
        format_failures += not prediction
        multi_reference += len(reference) > 1
        multi_prediction += len(prediction) > 1
        per_example_miou.append(symmetric_miou(reference, prediction))
        matches = greedy_matches(reference, prediction)
        true_positive += sum(iou >= 0.5 for iou, _, _ in matches)
        total_reference += len(reference)
        total_prediction += len(prediction)
        for iou, gt_index, pred_index in matches:
            if iou >= 0.3:
                gt = reference[gt_index]
                pred = prediction[pred_index]
                boundary_errors.extend((abs(gt[0] - pred[0]), abs(gt[1] - pred[1])))

    precision = true_positive / total_prediction if total_prediction else 0.0
    recall = true_positive / total_reference if total_reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "protocol": "TEMPO Appendix B",
        "scale": "percent_except_seconds",
        "records": len(rows),
        "format_failures": format_failures,
        "multi_reference_records": multi_reference,
        "multi_prediction_records": multi_prediction,
        "symmetric_mIoU": 100.0 * sum(per_example_miou) / len(per_example_miou),
        "F1_IoU_0.5": 100.0 * f1,
        "precision_IoU_0.5": 100.0 * precision,
        "recall_IoU_0.5": 100.0 * recall,
        "boundary_MAE_seconds": (
            sum(boundary_errors) / len(boundary_errors) if boundary_errors else None
        ),
        "matched_boundaries": len(boundary_errors),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
