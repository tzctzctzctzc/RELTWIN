#!/usr/bin/env python3
"""Build a local-window Boundary Utility manifest from long-audio predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interval_metrics import normalize_intervals, temporal_set_iou


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def localize_intervals(intervals, start: float, end: float):
    return normalize_intervals(
        [
            (max(left, start) - start, min(right, end) - start)
            for left, right in intervals
            if min(right, end) > max(left, start)
        ],
        end - start,
    )


def build_manifest(
    annotations: list[dict],
    predictions: list[dict],
    *,
    benchmark_name: str,
    expected_rows: int,
    boundary_context_seconds: float = 2.0,
    max_boundary_window_seconds: float = 300.0,
) -> list[dict]:
    if boundary_context_seconds < 0:
        raise ValueError("boundary_context_seconds must be non-negative")
    if max_boundary_window_seconds <= 0:
        raise ValueError("max_boundary_window_seconds must be positive")
    if len(annotations) != expected_rows or len(predictions) != expected_rows:
        raise ValueError(
            f"row count mismatch: annotations={len(annotations)}, "
            f"predictions={len(predictions)}, expected={expected_rows}"
        )
    by_index: dict[int, dict] = {}
    for row in predictions:
        index = int(row["index"])
        if index in by_index:
            raise ValueError(f"duplicate prediction index: {index}")
        by_index[index] = row
    if set(by_index) != set(range(expected_rows)):
        raise ValueError("prediction indices are not contiguous")

    output = []
    for index, annotation in enumerate(annotations):
        prediction = by_index[index]
        if str(annotation["caption"]).strip() != str(prediction["query"]).strip():
            raise ValueError(f"query mismatch at index {index}")
        if Path(annotation["audio_path"]).name != Path(prediction["audio"]).name:
            raise ValueError(f"audio mismatch at index {index}")
        selected_chunk = int(prediction["selected_chunk"])
        bounds = prediction["chunk_bounds_seconds"]
        if selected_chunk < 0 or selected_chunk >= len(bounds):
            raise ValueError(f"invalid selected chunk at index {index}")
        window_start, window_end = map(float, bounds[selected_chunk])
        full_duration = float(prediction["duration_seconds"])
        if (
            window_start < 0
            or window_end <= window_start
            or window_end > full_duration + 1e-3
        ):
            raise ValueError(f"invalid window bounds at index {index}")
        incumbent = normalize_intervals(prediction["prediction"], full_duration)
        selected_window_incumbent = localize_intervals(
            incumbent, window_start, window_end
        )
        if len(selected_window_incumbent) != len(incumbent):
            raise ValueError(f"incumbent escapes selected window at index {index}")
        # Boundary Utility can only move an existing boundary by a fraction of
        # a second.  Re-encoding the complete long-audio selection wastes memory
        # and can exceed the GPU limit without changing the feasible actions.
        # Crop using the incumbent only; labels never influence this window.
        if incumbent:
            boundary_start = max(
                window_start,
                min(left for left, _ in incumbent) - boundary_context_seconds,
            )
            boundary_end = min(
                window_end,
                max(right for _, right in incumbent) + boundary_context_seconds,
            )
        else:
            boundary_start, boundary_end = window_start, window_end
        local_incumbent = localize_intervals(
            incumbent, boundary_start, boundary_end
        )
        if len(local_incumbent) != len(incumbent):
            raise ValueError(f"incumbent escapes boundary crop at index {index}")
        global_truth = normalize_intervals(annotation["annotations"])
        local_truth = localize_intervals(
            global_truth, boundary_start, boundary_end
        )
        boundary_duration = boundary_end - boundary_start
        force_abstain_reason = (
            "boundary_crop_exceeds_memory_budget"
            if boundary_duration > max_boundary_window_seconds
            else None
        )
        output.append(
            {
                "benchmark": benchmark_name,
                "source": benchmark_name,
                "source_index": index,
                "audio_group": annotation.get(
                    "audio_group", annotation["audio_path"]
                ),
                "audio_path": annotation["audio_path"],
                "caption": str(annotation["caption"]).strip(),
                "annotations": local_truth,
                "duration": boundary_duration,
                "audio_window_start_seconds": boundary_start,
                "audio_window_end_seconds": boundary_end,
                "selected_chunk_start_seconds": window_start,
                "selected_chunk_end_seconds": window_end,
                "boundary_context_seconds": boundary_context_seconds,
                "max_boundary_window_seconds": max_boundary_window_seconds,
                "force_abstain_reason": force_abstain_reason,
                "full_duration": full_duration,
                "global_annotations": global_truth,
                "global_incumbent_prediction": incumbent,
                "incumbent_name": "official_windowed",
                "incumbent_prediction": local_incumbent,
                "stored_incumbent_iou": temporal_set_iou(global_truth, incumbent),
                "benchmark_id": annotation.get("benchmark_id", annotation.get("qid")),
                "annotation_overshoot_seconds": annotation.get(
                    "annotation_overshoot_seconds", 0.0
                ),
                "selected_chunk": selected_chunk,
                "chunk_detection_log_odds": prediction.get(
                    "chunk_detection_log_odds"
                ),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--boundary-context-seconds", type=float, default=2.0)
    parser.add_argument("--max-boundary-window-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_manifest(
        json.loads(args.annotations.read_text(encoding="utf-8")),
        read_jsonl(args.predictions),
        benchmark_name=args.benchmark_name,
        expected_rows=args.expected_rows,
        boundary_context_seconds=args.boundary_context_seconds,
        max_boundary_window_seconds=args.max_boundary_window_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "benchmark": args.benchmark_name,
                "rows": len(rows),
                "empty_local_truth": sum(not row["annotations"] for row in rows),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
