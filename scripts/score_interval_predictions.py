#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from interval_metrics import (
    THRESHOLDS,
    event_f1_iou,
    normalize_intervals,
    onset_f1_auto_code,
    parse_auto_aeg_intervals,
    parse_canonical_intervals,
    parse_spotsound_intervals,
    segment_f1,
    soft_precision_recall,
    temporal_set_iou,
    threshold_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def mean(values):
    return sum(values) / len(values) if values else None


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status", "ok") == "ok":
                rows.append(row)
    return rows


def main():
    args = parse_args()
    rows = load_rows(args.predictions)
    if not rows:
        raise ValueError("No successful prediction records found")

    spot_ious = []
    soft_recalls = []
    soft_precisions = []
    soft_f1s = []
    threshold_values = {
        threshold: {"recall": [], "precision": [], "f1": [], "all_recall": [], "event_f1": []}
        for threshold in THRESHOLDS
    }
    segment_values = []
    auto_onset_values = []
    raw_spot_nonempty = raw_auto_nonempty = raw_canonical_nonempty = 0
    ground_truth_multi = prediction_multi = 0
    present_count = absent_count = correct_rejections = false_activations = 0

    for row in rows:
        duration = float(row.get("duration_seconds", row.get("duration", 0.0)))
        ground_truth = normalize_intervals(row.get("ground_truth", row.get("gt", [])), duration or None)
        if "prediction" in row or "pred" in row:
            prediction = normalize_intervals(row.get("prediction", row.get("pred", [])), duration or None)
        else:
            prediction = parse_canonical_intervals(row.get("raw_answer", row.get("response", "")), duration or None)
        raw_answer = row.get("raw_answer", row.get("response", ""))
        raw_spot_nonempty += bool(parse_spotsound_intervals(raw_answer, duration or None))
        raw_auto_nonempty += bool(parse_auto_aeg_intervals(raw_answer))
        raw_canonical_nonempty += bool(parse_canonical_intervals(raw_answer, duration or None))

        if not ground_truth:
            absent_count += 1
            correct_rejections += not prediction
            false_activations += bool(prediction)
            continue
        present_count += 1
        ground_truth_multi += len(ground_truth) > 1
        prediction_multi += len(prediction) > 1
        set_iou = temporal_set_iou(ground_truth, prediction)
        precision, recall, soft_f1 = soft_precision_recall(ground_truth, prediction)
        spot_ious.append(set_iou)
        soft_precisions.append(precision)
        soft_recalls.append(recall)
        soft_f1s.append(soft_f1)
        for threshold in THRESHOLDS:
            values = threshold_metrics(ground_truth, prediction, threshold)
            for key in ("recall", "precision", "f1", "all_recall"):
                threshold_values[threshold][key].append(values[key])
            threshold_values[threshold]["event_f1"].append(
                event_f1_iou(ground_truth, prediction, threshold)["f1"]
            )
        if duration > 0:
            segment_values.append(segment_f1(ground_truth, prediction, duration)["f1"])
        auto_onset_values.append(onset_f1_auto_code(ground_truth, prediction)["f1"])

    percent = lambda values: 100.0 * mean(values) if values else None
    summary = {
        "predictions": str(args.predictions.resolve()),
        "scale": "percent",
        "records": len(rows),
        "present_records": present_count,
        "absent_records": absent_count,
        "multi_ground_truth_records": ground_truth_multi,
        "multi_prediction_records": prediction_multi,
        "parser_audit": {
            "spotsound_raw_nonempty": raw_spot_nonempty,
            "auto_aeg_raw_nonempty": raw_auto_nonempty,
            "canonical_raw_nonempty": raw_canonical_nonempty,
        },
        "spotsound_protocol": {
            "mIoU": percent(spot_ious),
            "R1@0.3": 100.0 * sum(value >= 0.3 for value in spot_ious) / len(spot_ious),
            "R1@0.5": 100.0 * sum(value >= 0.5 for value in spot_ious) / len(spot_ious),
        },
        "auto_aeg_core": {
            "mIoU": percent(soft_recalls),
            **{
                f"R_IoU@{threshold}": percent(threshold_values[threshold]["recall"])
                for threshold in THRESHOLDS
            },
            **{
                f"All_R_IoU@{threshold}": percent(threshold_values[threshold]["all_recall"])
                for threshold in THRESHOLDS
            },
            "seg_F1": percent(segment_values),
            "ev_F1_public_code_onset_0.5s": percent(auto_onset_values),
        },
        "paper_faithful_symmetric": {
            "soft_P_IoU": percent(soft_precisions),
            "soft_R_IoU": percent(soft_recalls),
            "soft_F1_IoU": percent(soft_f1s),
            **{
                f"P_IoU@{threshold}": percent(threshold_values[threshold]["precision"])
                for threshold in THRESHOLDS
            },
            **{
                f"F1_IoU@{threshold}": percent(threshold_values[threshold]["f1"])
                for threshold in THRESHOLDS
            },
            **{
                f"event_F1_IoU@{threshold}": percent(threshold_values[threshold]["event_f1"])
                for threshold in THRESHOLDS
            },
        },
        "rejection": {
            "correct_rejections": correct_rejections,
            "false_activations": false_activations,
            "FPR_absent": false_activations / absent_count if absent_count else None,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

