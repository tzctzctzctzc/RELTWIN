#!/usr/bin/env python3
"""Build balanced non-held-out ESC-50 localization rehearsal mixtures."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from build_esc50_longneedle import HELDOUT_CLASSES, choose_starts
from build_esc50_reltwin import load_clip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=432)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    excluded = set()
    for path in args.exclude_manifest:
        for row in json.loads(path.read_text(encoding="utf-8")):
            excluded.update(row.get("source_files", []))

    by_class = defaultdict(list)
    with args.metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["filename"] not in excluded and row["category"] not in HELDOUT_CLASSES:
                by_class[row["category"]].append(row["filename"])
    classes = sorted(name for name, files in by_class.items() if files)
    if len(classes) < 2:
        raise ValueError("Too few non-held-out ESC-50 classes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    profiles = [
        (duration, cardinality, scale_bucket)
        for duration in (12.0, 24.0, 60.0)
        for cardinality in (1, 2, 3, 4)
        for scale_bucket in (0, 1, 2)
    ]
    target_fractions = (0.07, 0.20, 0.42)
    for index in range(args.count):
        duration, cardinality, scale_bucket = profiles[index % len(profiles)]
        target_class = classes[index % len(classes)]
        target_source = str(rng.choice(by_class[target_class]))
        target = load_clip(args.audio_dir / target_source, args.sample_rate)
        target_seconds = duration * target_fractions[scale_bucket] / cardinality
        target_samples = min(len(target), max(round(0.55 * args.sample_rate), round(target_seconds * args.sample_rate)))
        crop_start = int(rng.integers(0, max(1, len(target) - target_samples + 1)))
        target = target[crop_start : crop_start + target_samples]

        total_samples = round(duration * args.sample_rate)
        background = np.zeros(total_samples, dtype=np.float32)
        background_sources = []
        cursor = 0
        distractor_classes = [name for name in classes if name != target_class]
        while cursor < total_samples:
            distractor_class = str(rng.choice(distractor_classes))
            source = str(rng.choice(by_class[distractor_class]))
            clip = load_clip(args.audio_dir / source, args.sample_rate)
            take = min(len(clip), total_samples - cursor)
            background[cursor : cursor + take] = clip[:take] * 0.55
            background_sources.append(source)
            cursor += take

        starts = choose_starts(
            rng,
            cardinality,
            total_samples,
            len(target),
            round(0.20 * args.sample_rate),
        )
        annotations = []
        for start in starts:
            end = start + len(target)
            background[start:end] += target * 1.25
            annotations.append(
                [round(start / args.sample_rate, 6), round(end / args.sample_rate, 6)]
            )
        peak = float(np.max(np.abs(background)))
        if peak > 0.98:
            background *= 0.98 / peak
        filename = f"sc_{index:04d}.wav"
        sf.write(args.output_dir / filename, background, args.sample_rate, subtype="PCM_16")
        rows.append(
            {
                "audio_path": f"audio/{filename}",
                "caption": target_class.replace("_", " "),
                "annotations": annotations,
                "duration": duration,
                "cardinality": cardinality,
                "scale_bucket": scale_bucket,
                "target_density": sum(end - start for start, end in annotations) / duration,
                "source_files": [target_source, *background_sources],
                "source_dataset": "ESC-50-train-classes",
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "classes": len(classes),
                "excluded_source_files": len(excluded),
                "cardinalities": sorted({row["cardinality"] for row in rows}),
                "durations": sorted({row["duration"] for row in rows}),
            }
        )
    )


if __name__ == "__main__":
    main()
