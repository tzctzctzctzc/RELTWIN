#!/usr/bin/env python3
"""Build long, low-density ESC-50 mixtures for external router development."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HELDOUT_CLASSES = {
    "dog",
    "rooster",
    "sea_waves",
    "thunderstorm",
    "crying_baby",
    "clapping",
    "keyboard_typing",
    "vacuum_cleaner",
    "chainsaw",
    "fireworks",
}


def choose_starts(rng, count: int, clip_samples: int, target_samples: int, gap_samples: int):
    starts = []
    for _ in range(1000):
        candidate = int(rng.integers(gap_samples, clip_samples - target_samples - gap_samples))
        if all(abs(candidate - start) >= target_samples + gap_samples for start in starts):
            starts.append(candidate)
            if len(starts) == count:
                return sorted(starts)
    raise RuntimeError("Could not place non-overlapping target events")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main():
    import soundfile as sf

    from build_esc50_reltwin import load_clip

    args = parse_args()
    rng = np.random.default_rng(args.seed)
    excluded = set()
    for path in args.exclude_manifest:
        for row in json.loads(path.read_text(encoding="utf-8")):
            excluded.update(row.get("source_files", []))

    by_class = defaultdict(list)
    with args.metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["filename"] not in excluded:
                by_class[row["category"]].append(row["filename"])
    target_classes = sorted(HELDOUT_CLASSES)
    distractor_classes = sorted(set(by_class) - HELDOUT_CLASSES)
    if any(len(by_class[name]) < 16 for name in target_classes):
        raise ValueError("Too few source-disjoint target clips")

    target_orders = {name: list(rng.permutation(by_class[name])) for name in target_classes}
    class_order = target_classes * ((args.count + len(target_classes) - 1) // len(target_classes))
    rng.shuffle(class_order)
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_samples = round(args.duration * args.sample_rate)
    for index, target_class in enumerate(class_order[: args.count]):
        target_source = target_orders[target_class].pop()
        background = np.zeros(total_samples, dtype=np.float32)
        background_sources = []
        cursor = 0
        while cursor < total_samples:
            distractor_class = str(rng.choice(distractor_classes))
            source = str(rng.choice(by_class[distractor_class]))
            clip = load_clip(args.audio_dir / source, args.sample_rate)
            take = min(len(clip), total_samples - cursor)
            background[cursor : cursor + take] = clip[:take] * 0.65
            background_sources.append(source)
            cursor += take

        target = load_clip(args.audio_dir / target_source, args.sample_rate)
        target_seconds = float(rng.uniform(1.2, 2.5))
        target_samples = min(len(target), round(target_seconds * args.sample_rate))
        crop_start = int(rng.integers(0, max(1, len(target) - target_samples + 1)))
        target = target[crop_start : crop_start + target_samples]
        occurrence_count = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        starts = choose_starts(
            rng,
            occurrence_count,
            total_samples,
            target_samples,
            round(0.5 * args.sample_rate),
        )
        annotations = []
        for start in starts:
            end = start + target_samples
            background[start:end] += target * 1.35
            annotations.append([
                round(start / args.sample_rate, 6),
                round(end / args.sample_rate, 6),
            ])
        peak = float(np.max(np.abs(background)))
        if peak > 0.98:
            background *= 0.98 / peak
        filename = f"longneedle_{index:04d}.wav"
        sf.write(args.output_dir / filename, background, args.sample_rate, subtype="PCM_16")
        rows.append({
            "audio_path": str((args.output_dir / filename).resolve()),
            "pair_id": 20000 + index,
            "variant": 0,
            "template": "longneedle_ordinary",
            "relation": "target",
            "caption": target_class.replace("_", " "),
            "annotations": annotations,
            "target_density": sum(end - start for start, end in annotations) / args.duration,
            "source_files": [target_source, *background_sources],
            "target_source": target_source,
            "source_dataset": "ESC-50",
        })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "duration_seconds": args.duration,
        "mean_target_density": float(np.mean([row["target_density"] for row in rows])),
        "excluded_source_files": len(excluded),
    }))


if __name__ == "__main__":
    main()
