#!/usr/bin/env python3
"""Restore local-window NOVA predictions to a long recording timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interval_metrics import normalize_intervals, temporal_set_iou


def restore_record(row: dict) -> dict:
    offset = float(row["audio_window_start_seconds"])
    duration = float(row["full_duration"])

    def restore(intervals):
        return normalize_intervals(
            [(start + offset, end + offset) for start, end in intervals], duration
        )

    incumbent = restore(row["incumbent_prediction"])
    candidate = restore(row["candidate_prediction"])
    selected = restore(row["selected_prediction"])
    threshold = restore(row["fixed_threshold_prediction"])
    truth = normalize_intervals(row["global_annotations"])
    result = dict(row)
    result.update(
        {
            "duration": duration,
            "ground_truth": truth,
            "incumbent_prediction": incumbent,
            "candidate_prediction": candidate,
            "selected_prediction": selected,
            "fixed_threshold_prediction": threshold,
            "incumbent_iou": temporal_set_iou(truth, incumbent),
            "candidate_iou": temporal_set_iou(truth, candidate),
            "selected_iou": temporal_set_iou(truth, selected),
            "fixed_threshold_iou": temporal_set_iou(truth, threshold),
        }
    )
    result["delta_iou"] = result["selected_iou"] - result["incumbent_iou"]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        restore_record(json.loads(line))
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(row["source_index"]) for row in rows]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("predictions must have unique sorted source_index values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "switches": sum(bool(row["switch"]) for row in rows),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
