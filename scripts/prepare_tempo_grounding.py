#!/usr/bin/env python3
"""Materialize the public TEMPO/TACOS audio-grounding evaluation split."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


_INTERVAL_RE = re.compile(
    r"<\|\s*(-?\d+(?:\.\d+)?)\s*\|>\s*to\s*"
    r"<\|\s*(-?\d+(?:\.\d+)?)\s*\|>",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    return parser.parse_args()


def extract_query(question: str) -> str:
    for left, right in (("'", "'"), ("‘", "’"), ('“', '”'), ('"', '"')):
        start = question.find(left)
        end = question.rfind(right)
        if start >= 0 and end > start:
            return question[start + len(left) : end].strip()
    # Public TEMPO prompts are heterogeneous: some quote the event, while
    # others directly ask e.g. "when does the power saw happen?".  The exact
    # source question is preserved separately for model input, so retaining
    # the full text is the lossless fallback used for logging/evaluation IDs.
    return question.strip()


def parse_intervals(answer: str) -> list[list[float]]:
    intervals = [
        [float(start), float(end)]
        for start, end in _INTERVAL_RE.findall(answer)
        if float(end) >= float(start)
    ]
    if not intervals:
        raise ValueError(f"Cannot parse any interval from: {answer}")
    return intervals


def main() -> int:
    args = parse_args()
    parquet_files = sorted(args.parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet shards in {args.parquet_dir}")

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.annotations.parent.mkdir(parents=True, exist_ok=True)
    annotations: list[dict] = []
    written_audio: set[str] = set()

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            audio_key = str(row["audio_key"])
            audio = row["audio"]
            relative_audio = Path(audio_key)
            destination = args.audio_dir / relative_audio
            if audio_key not in written_audio:
                payload = audio.get("bytes")
                if not payload:
                    raise ValueError(f"Missing embedded audio bytes for {audio_key}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                written_audio.add(audio_key)

            annotations.append(
                {
                    "audio_path": relative_audio.as_posix(),
                    "caption": extract_query(str(row["question"])),
                    "annotations": parse_intervals(str(row["answer"])),
                    "benchmark": "TEMPO-TACOS-AudioGrounding",
                    "benchmark_id": str(row["id"]),
                    "audio_id": str(row["audio_id"]),
                    "source_question": str(row["question"]),
                }
            )

    args.annotations.write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    multi = sum(len(row["annotations"]) > 1 for row in annotations)
    print(
        json.dumps(
            {
                "records": len(annotations),
                "unique_audio": len(written_audio),
                "multi_interval_records": multi,
                "annotations": str(args.annotations.resolve()),
                "audio_dir": str(args.audio_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
