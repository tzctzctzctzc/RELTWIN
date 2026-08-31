#!/usr/bin/env python3
"""Derive ordinary-event queries from a frozen RelTwin relation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def derive_event_spans(row: dict, duration_a: float, duration_b: float) -> dict[str, list[list[float]]]:
    """Recover A/B event spans from the known AB and BA composition windows."""
    (ab_start, ab_end), = row["window_ab"]
    (ba_start, ba_end), = row["window_ba"]
    spans = {
        "A": [
            [ab_start, ab_start + duration_a],
            [ba_end - duration_a, ba_end],
        ],
        "B": [
            [ab_end - duration_b, ab_end],
            [ba_start, ba_start + duration_b],
        ],
    }
    return {
        key: [[round(float(start), 6), round(float(end), 6)] for start, end in sorted(value)]
        for key, value in spans.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-manifest", type=Path, required=True)
    parser.add_argument("--source-audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import soundfile as sf

    args = parse_args()
    relation_rows = json.loads(args.relation_manifest.read_text(encoding="utf-8"))
    unique = {}
    for row in relation_rows:
        key = (row["audio_path"], int(row["pair_id"]), int(row["variant"]))
        unique.setdefault(key, row)

    ordinary_rows = []
    for row in unique.values():
        source_a, source_b = row["source_files"]
        durations = [
            sf.info(args.source_audio_dir / source_a).duration,
            sf.info(args.source_audio_dir / source_b).duration,
        ]
        spans = derive_event_spans(row, *durations)
        common = {
            "audio_path": row["audio_path"],
            "pair_id": row["pair_id"],
            "variant": row["variant"],
            "layout": row["layout"],
            "template": "ordinary_event",
            "source_files": row["source_files"],
            "source_dataset": "ESC-50",
        }
        ordinary_rows.extend([
            {
                **common,
                "relation": "A",
                "caption": row["event_a"],
                "annotations": spans["A"],
            },
            {
                **common,
                "relation": "B",
                "caption": row["event_b"],
                "annotations": spans["B"],
            },
        ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ordinary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"unique_audios": len(unique), "ordinary_queries": len(ordinary_rows)}))


if __name__ == "__main__":
    main()
