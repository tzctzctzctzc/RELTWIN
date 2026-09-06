#!/usr/bin/env python3
"""Normalize public temporal-grounding benchmarks to the SpotSound harness."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def clotho_audio_name(vid: str) -> str:
    # The official WebDataset removes decimal points from the JSONL ``vid`` key.
    return vid.replace(".", "") + ".wav"


def prepare_clotho(source: Path) -> list[dict]:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(
            {
                "benchmark": "Clotho-Moment",
                "qid": str(item["qid"]),
                "audio_path": clotho_audio_name(item["vid"]),
                "caption": item["query"],
                "annotations": item["relevant_windows"],
                "duration": float(item["duration"]),
                "fg_dB": item.get("fg_dB"),
            }
        )
    return rows


def prepare_amr_jsonl(source: Path, *, benchmark: str) -> list[dict]:
    """Normalize the public Lighthouse AMR evaluation JSONL files.

    UnAV100-subset and TUT Sound Events 2017 use ``<vid>.wav`` in the
    author-released WAV archives.  Clotho-Moment is deliberately kept on its
    existing converter because its WebDataset filenames remove decimal points.
    """
    rows = []
    seen_qids: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        qid = str(item["qid"])
        if qid in seen_qids:
            raise ValueError(f"Duplicate qid {qid!r} at line {line_number}")
        seen_qids.add(qid)
        duration = float(item["duration"])
        annotations = [
            [float(start), float(end)]
            for start, end in item["relevant_windows"]
        ]
        if duration <= 0:
            raise ValueError(f"Non-positive duration at line {line_number}")
        if not annotations:
            raise ValueError(f"Missing relevant window at line {line_number}")
        for start, end in annotations:
            if start < 0 or end <= start or end > duration + 1e-6:
                raise ValueError(
                    f"Invalid interval {[start, end]} for duration {duration} "
                    f"at line {line_number}"
                )
        rows.append(
            {
                "benchmark": benchmark,
                "qid": qid,
                "audio_path": f'{item["vid"]}.wav',
                "caption": item["query"],
                "annotations": annotations,
                "duration": duration,
            }
        )
    return rows


def prepare_aegbench(source: Path) -> list[dict]:
    items = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items.get("items", items)
    rows = []
    for item in items:
        clips_by_category: dict[str, list[list[float]]] = defaultdict(list)
        for clip in item.get("clips", []):
            clips_by_category[clip["category"]].append(
                [float(clip["start"]), float(clip["end"])]
            )
        for category in item.get("categories", []):
            annotations = clips_by_category.get(category, [])
            if not annotations:
                continue
            rows.append(
                {
                    "benchmark": "AEGBench",
                    "benchmark_id": item.get("benchmark_id", item.get("id")),
                    "audio_path": item.get("audio_rel", item["audio_path"]),
                    "caption": category,
                    "category": category,
                    "annotations": annotations,
                    "duration": float(item["duration"]),
                    "source": item.get("source"),
                    "hardcase_tags": item.get("hardcase_tags", []),
                }
            )
    return rows


def prepare_audiogrounding(source: Path) -> list[dict]:
    items = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for item in items:
        for phrase_index, phrase in enumerate(item.get("phrases", [])):
            rows.append(
                {
                    "benchmark": "AudioGrounding-v2",
                    "benchmark_id": f'{item["audiocap_id"]}:{phrase_index}',
                    "audio_path": item["audio_id"],
                    "caption": phrase["phrase"],
                    "annotations": [
                        [float(start), float(end)]
                        for start, end in phrase["segments"]
                    ],
                    "source_caption": item.get("tokens"),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=(
            "clotho-moment",
            "aegbench",
            "audiogrounding",
            "unav100-subset",
            "tut2017",
        ),
        required=True,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare = {
        "clotho-moment": prepare_clotho,
        "aegbench": prepare_aegbench,
        "audiogrounding": prepare_audiogrounding,
        "unav100-subset": lambda source: prepare_amr_jsonl(
            source, benchmark="UnAV100-subset-public100"
        ),
        "tut2017": lambda source: prepare_amr_jsonl(
            source, benchmark="TUT-Sound-Events-2017"
        ),
    }[args.benchmark]
    rows = prepare(args.source)
    expected = {
        "clotho-moment": 6649,
        "aegbench": 9924,
        "audiogrounding": 997,
        "unav100-subset": 100,
        "tut2017": 104,
    }[args.benchmark]
    if len(rows) != expected:
        raise ValueError(f"Unexpected {args.benchmark} row count: {len(rows)} != {expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"benchmark": args.benchmark, "rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
