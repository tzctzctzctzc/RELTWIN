#!/usr/bin/env python3
"""Audit benchmark row identity without deduplicating repeated audio queries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audio_counts = Counter(row["audio"] for row in rows)
    duplicate_audio = {name: count for name, count in sorted(audio_counts.items()) if count > 1}
    report = {
        "prediction_file": args.predictions.as_posix(),
        "rows": len(rows),
        "unique_indices": len({row["index"] for row in rows}),
        "unique_audio_files": len(audio_counts),
        "additional_query_rows_reusing_audio": len(rows) - len(audio_counts),
        "multiply_queried_audio_files": len(duplicate_audio),
        "duplicate_audio_row_counts": duplicate_audio,
        "nonempty_predictions": sum(bool(row.get("prediction")) for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
