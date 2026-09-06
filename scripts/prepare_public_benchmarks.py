#!/usr/bin/env python3
"""Normalize public temporal-grounding benchmarks to the SpotSound harness."""

from __future__ import annotations

import argparse
import csv
import json
import re
import wave
from collections import defaultdict
from pathlib import Path


LAT_PROMPT_PREFIX = "<audio>Please listen to the audio carefully and locate "
LAT_PROMPT_SUFFIX = (
    " Please strictly output the result in the format of "
    "[Start Time - End Time]. Do not output any extra explanatory text."
)
LAT_INTERVAL = re.compile(
    r"^\[(?P<start>\d{2}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(?P<end>\d{2}:\d{2}(?::\d{2})?)\]$"
)


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


def prepare_amr_jsonl(
    source: Path,
    *,
    benchmark: str,
    annotation_tolerance_seconds: float = 0.0,
) -> list[dict]:
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
        maximum_overshoot = max(end - duration for _, end in annotations)
        for start, end in annotations:
            if (
                start < 0
                or end <= start
                or end > duration + annotation_tolerance_seconds + 1e-6
            ):
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
                "annotation_overshoot_seconds": max(0.0, maximum_overshoot),
            }
        )
    return rows


def parse_lat_timestamp(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Unsupported LAT timestamp: {value!r}")
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid LAT timestamp: {value!r}")
    return float(hours * 3600 + minutes * 60 + seconds)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate()
        if frame_rate <= 0:
            raise ValueError(f"Invalid WAV frame rate: {path}")
        return handle.getnframes() / frame_rate


def prepare_lat_tag(
    source: Path,
    metadata: Path,
    *,
    language: str,
    audio_dir: Path | None = None,
    allow_annotation_overshoot: bool = False,
) -> list[dict]:
    """Normalize LAT-Bench temporal-audio-grounding conversations.

    The released task prompt wraps its semantic query in fixed generation
    instructions.  The SpotSound harness already supplies an output-format
    instruction, so only that fixed wrapper is removed; the complete semantic
    description is preserved verbatim.
    """
    durations: dict[str, float] = {}
    for line_number, line in enumerate(
        metadata.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        audio_id = str(item["id"])
        if audio_id in durations:
            raise ValueError(
                f"Duplicate LAT metadata id {audio_id!r} at line {line_number}"
            )
        durations[audio_id] = float(item["duration"])

    rows = []
    seen_qids: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item.get("messages", [])
        audio_ids = item.get("audios", [])
        if (
            len(messages) != 2
            or messages[0].get("role") != "user"
            or messages[1].get("role") != "assistant"
            or len(audio_ids) != 1
        ):
            raise ValueError(f"Unexpected LAT record at line {line_number}")

        audio_id = str(audio_ids[0])
        if audio_id not in durations:
            raise ValueError(
                f"Missing LAT metadata for {audio_id!r} at line {line_number}"
            )
        raw_prompt = str(messages[0]["content"])
        if not raw_prompt.startswith(LAT_PROMPT_PREFIX) or not raw_prompt.endswith(
            LAT_PROMPT_SUFFIX
        ):
            raise ValueError(f"Unexpected LAT prompt template at line {line_number}")
        caption = raw_prompt[
            len(LAT_PROMPT_PREFIX) : len(raw_prompt) - len(LAT_PROMPT_SUFFIX)
        ].strip()
        if not caption:
            raise ValueError(f"Empty LAT semantic query at line {line_number}")

        raw_interval = str(messages[1]["content"]).strip()
        match = LAT_INTERVAL.fullmatch(raw_interval)
        if match is None:
            raise ValueError(f"Unexpected LAT answer at line {line_number}: {raw_interval!r}")
        start = parse_lat_timestamp(match.group("start"))
        end = parse_lat_timestamp(match.group("end"))
        metadata_duration = durations[audio_id]
        audio_path = f"{audio_id}.wav"
        duration = (
            wav_duration(audio_dir / audio_path)
            if audio_dir is not None
            else metadata_duration
        )
        annotation_overshoot = max(0.0, end - duration)
        if start < 0 or end <= start or (
            annotation_overshoot > 1.0 and not allow_annotation_overshoot
        ):
            raise ValueError(
                f"Invalid LAT interval {[start, end]} for duration {duration} "
                f"at line {line_number}"
            )

        qid = f"{audio_id}:{line_number - 1}"
        if qid in seen_qids:
            raise ValueError(f"Duplicate LAT qid {qid!r} at line {line_number}")
        seen_qids.add(qid)
        rows.append(
            {
                "benchmark": f"LAT-Bench-{language.upper()}-TAG",
                "qid": qid,
                "audio_group": audio_id,
                "audio_path": audio_path,
                "caption": caption,
                "annotations": [[start, end]],
                "duration": duration,
                "metadata_duration": metadata_duration,
                "duration_mismatch": abs(duration - metadata_duration) > 1.0,
                "annotation_overshoot_seconds": annotation_overshoot,
                "released_prompt": raw_prompt,
            }
        )
    return rows


def prepare_desed_public(source: Path, *, audio_dir: Path) -> list[dict]:
    """Group DESED strong labels into one query-conditioned interval set."""
    grouped: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected_fields = {"filename", "onset", "offset", "event_label"}
        if set(reader.fieldnames or []) != expected_fields:
            raise ValueError(
                f"Unexpected DESED columns: {reader.fieldnames}; "
                f"expected {sorted(expected_fields)}"
            )
        for line_number, item in enumerate(reader, start=2):
            filename = str(item["filename"])
            event_label = str(item["event_label"])
            if Path(filename).name != filename or not event_label:
                raise ValueError(f"Invalid DESED identity at line {line_number}")
            start, end = float(item["onset"]), float(item["offset"])
            if start < 0 or end <= start:
                raise ValueError(f"Invalid DESED interval at line {line_number}")
            grouped[(filename, event_label)].append([start, end])

    durations: dict[str, float] = {}
    rows = []
    for (filename, event_label), annotations in grouped.items():
        if filename not in durations:
            audio_path = audio_dir / filename
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing DESED audio: {audio_path}")
            durations[filename] = wav_duration(audio_path)
        duration = durations[filename]
        annotations.sort()
        for previous, current in zip(annotations, annotations[1:]):
            if current[0] < previous[1]:
                raise ValueError(
                    f"Overlapping DESED intervals for {filename!r}, {event_label!r}"
                )
        if annotations[-1][1] > duration + 0.1:
            raise ValueError(
                f"DESED annotation exceeds audio duration for {filename!r}, "
                f"{event_label!r}: {annotations[-1][1]} > {duration}"
            )
        rows.append(
            {
                "benchmark": "DESED-public-eval",
                "qid": f"{filename}:{event_label}",
                "audio_group": filename,
                "audio_path": filename,
                "caption": event_label.replace("_", " "),
                "event_label": event_label,
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
            "lat-bench-en-tag",
            "desed-public-eval",
        ),
        required=True,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Required metadata JSONL for LAT-Bench.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="Required WAV directory for LAT-Bench duration validation.",
    )
    parser.add_argument(
        "--allow-annotation-overshoot",
        action="store_true",
        help="Preserve and flag released intervals beyond the actual audio duration.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare = {
        "clotho-moment": prepare_clotho,
        "aegbench": prepare_aegbench,
        "audiogrounding": prepare_audiogrounding,
        "unav100-subset": lambda source: prepare_amr_jsonl(
            source,
            benchmark="UnAV100-subset-public100",
            annotation_tolerance_seconds=2.0,
        ),
        "tut2017": lambda source: prepare_amr_jsonl(
            source, benchmark="TUT-Sound-Events-2017"
        ),
    }.get(args.benchmark)
    if args.benchmark == "lat-bench-en-tag":
        if args.metadata is None:
            raise ValueError("--metadata is required for LAT-Bench")
        if args.audio_dir is None:
            raise ValueError("--audio-dir is required for LAT-Bench")
        rows = prepare_lat_tag(
            args.source,
            args.metadata,
            language="en",
            audio_dir=args.audio_dir,
            allow_annotation_overshoot=args.allow_annotation_overshoot,
        )
    elif args.benchmark == "desed-public-eval":
        if args.audio_dir is None:
            raise ValueError("--audio-dir is required for DESED")
        rows = prepare_desed_public(args.source, audio_dir=args.audio_dir)
    else:
        rows = prepare(args.source)
    expected = {
        "clotho-moment": 6649,
        "aegbench": 9924,
        "audiogrounding": 997,
        "unav100-subset": 100,
        "tut2017": 104,
        "lat-bench-en-tag": 426,
        "desed-public-eval": 1112,
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
