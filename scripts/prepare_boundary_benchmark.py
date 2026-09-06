#!/usr/bin/env python3
"""Build a strict Boundary Utility manifest from cached benchmark predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def build_manifest(
    annotations: list[dict],
    predictions: list[dict],
    *,
    benchmark_name: str,
    incumbent_name: str,
    expected_rows: int | None = None,
) -> list[dict]:
    if expected_rows is not None and len(annotations) != expected_rows:
        raise ValueError(
            f"annotation row count mismatch: {len(annotations)} != {expected_rows}"
        )
    if len(predictions) != len(annotations):
        raise ValueError(
            f"prediction row count mismatch: {len(predictions)} != {len(annotations)}"
        )

    by_index: dict[int, dict] = {}
    for row in predictions:
        if "index" not in row:
            raise ValueError("prediction is missing index")
        index = int(row["index"])
        if index in by_index:
            raise ValueError(f"duplicate prediction index: {index}")
        by_index[index] = row
    expected_indices = set(range(len(annotations)))
    if set(by_index) != expected_indices:
        missing = sorted(expected_indices - set(by_index))[:10]
        extra = sorted(set(by_index) - expected_indices)[:10]
        raise ValueError(f"prediction indices are not contiguous; missing={missing}, extra={extra}")

    output = []
    for index, annotation in enumerate(annotations):
        prediction = by_index[index]
        annotation_query = normalize_text(annotation["caption"])
        prediction_query = normalize_text(prediction["query"])
        if annotation_query != prediction_query:
            raise ValueError(
                f"query mismatch at index {index}: {annotation_query!r} != {prediction_query!r}"
            )

        annotation_id = annotation.get("benchmark_id", annotation.get("qid"))
        prediction_id = prediction.get("benchmark_id", prediction.get("qid"))
        if annotation_id is not None and prediction_id is not None:
            if str(annotation_id) != str(prediction_id):
                raise ValueError(
                    f"benchmark id mismatch at index {index}: "
                    f"{annotation_id!r} != {prediction_id!r}"
                )

        annotation_audio = str(annotation["audio_path"])
        prediction_audio = str(prediction["audio"])
        if Path(annotation_audio).name != Path(prediction_audio).name:
            raise ValueError(
                f"audio mismatch at index {index}: "
                f"{annotation_audio!r} != {prediction_audio!r}"
            )

        duration = annotation.get("duration", prediction.get("duration_seconds"))
        if duration is None:
            raise ValueError(f"duration is missing at index {index}")
        duration = float(duration)
        prediction_duration = prediction.get("duration_seconds")
        if prediction_duration is not None and abs(duration - float(prediction_duration)) > 1e-3:
            raise ValueError(
                f"duration mismatch at index {index}: {duration} != {prediction_duration}"
            )

        output.append(
            {
                "benchmark": benchmark_name,
                "source": benchmark_name,
                "source_index": index,
                "audio_group": annotation_audio,
                "audio_path": annotation_audio,
                "caption": annotation_query,
                "annotations": annotation["annotations"],
                "duration": duration,
                "incumbent_name": incumbent_name,
                "incumbent_prediction": prediction["prediction"],
                "benchmark_id": annotation_id,
                "hardcase_tags": annotation.get("hardcase_tags", []),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--incumbent-name", default="official")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a JSON list")
    manifest = build_manifest(
        annotations,
        read_jsonl(args.predictions),
        benchmark_name=args.benchmark_name,
        incumbent_name=args.incumbent_name,
        expected_rows=args.expected_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "benchmark": args.benchmark_name,
                "rows": len(manifest),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
