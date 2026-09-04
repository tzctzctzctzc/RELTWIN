#!/usr/bin/env python3
"""Convert official AudioGrounding training records into SpanTool rows."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--max-audio-seconds", type=float, default=120.0)
    parser.add_argument("--repair-dir", type=Path)
    parser.add_argument("--default-audio-seconds", type=float, default=10.0)
    return parser.parse_args()


def audio_id(value: str) -> str:
    return Path(value).stem.removeprefix("Y")


def convert_record(record: dict, audio_filename: str) -> list[dict]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for event in record.get("events", []):
        caption = str(event.get("caption", "")).strip()
        start = float(event.get("start_time", 0.0))
        end = float(event.get("end_time", 0.0))
        if caption and end > start:
            grouped[caption].append([start, end])
    rows = []
    for caption, annotations in sorted(grouped.items()):
        annotations.sort()
        rows.append(
            {
                "audio_path": audio_filename,
                "caption": caption,
                "annotations": annotations,
                "qid": f"{Path(audio_filename).stem}:{caption}",
                "source_dataset": "AudioGrounding-train",
            }
        )
    return rows


def repair_unknown_length_flac(
    source: Path, destination: Path, duration_seconds: float
) -> None:
    """Copy a FLAC and fill a missing STREAMINFO total-samples field."""
    with source.open("rb") as handle:
        header = handle.read(42)
    if (
        len(header) < 42
        or header[:4] != b"fLaC"
        or header[4] & 0x7F
        or int.from_bytes(header[5:8], "big") != 34
    ):
        raise ValueError(f"Unsupported FLAC header: {source}")
    stream_info = int.from_bytes(header[18:26], "big")
    sample_rate = stream_info >> 44
    total_samples = stream_info & ((1 << 36) - 1)
    if sample_rate <= 0 or total_samples != 0:
        raise ValueError(f"FLAC does not have a repairable unknown length: {source}")
    expected_samples = round(duration_seconds * sample_rate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    repaired = (stream_info & ~((1 << 36) - 1)) | expected_samples
    with destination.open("r+b") as handle:
        handle.seek(18)
        handle.write(repaired.to_bytes(8, "big"))


def main() -> None:
    args = parse_args()
    excluded = set()
    for path in args.exclude_manifest:
        records = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(audio_id(row["audio_path"]) for row in records)

    rows = []
    audio_count = 0
    invalid_audio = 0
    repaired_audio = 0
    import soundfile as sf

    for metadata_path in sorted(args.input_dir.glob("*.json")):
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if record.get("split") not in (None, "train"):
            continue
        if audio_id(metadata_path.name) in excluded:
            continue
        audio_path = metadata_path.with_suffix(".flac")
        if not audio_path.is_file():
            continue
        selected_audio = audio_path
        try:
            info = sf.info(selected_audio)
            duration = info.frames / info.samplerate
            if not 0 < duration <= args.max_audio_seconds:
                raise ValueError(f"invalid duration {duration}")
        except (RuntimeError, ValueError):
            if args.repair_dir is None:
                invalid_audio += 1
                continue
            try:
                selected_audio = args.repair_dir / audio_path.name
                if not selected_audio.is_file():
                    repair_unknown_length_flac(
                        audio_path, selected_audio, args.default_audio_seconds
                    )
                info = sf.info(selected_audio)
                duration = info.frames / info.samplerate
                if not 0 < duration <= args.max_audio_seconds:
                    raise ValueError(f"invalid repaired duration {duration}")
                repaired_audio += 1
            except (RuntimeError, ValueError):
                invalid_audio += 1
                continue
        converted = convert_record(record, str(selected_audio.resolve()))
        if converted:
            for row in converted:
                row["duration"] = duration
            rows.extend(converted)
            audio_count += 1

    if not rows:
        raise ValueError("No AudioGrounding training rows were produced")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "audio": audio_count,
                "excluded_audio_ids": len(excluded),
                "invalid_audio": invalid_audio,
                "repaired_audio": repaired_audio,
            }
        )
    )


if __name__ == "__main__":
    main()
