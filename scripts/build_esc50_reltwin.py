#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


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


def seconds(samples, sample_rate):
    return round(samples / sample_rate, 6)


def load_clip(path, sample_rate):
    wave, source_rate = sf.read(path, always_2d=True, dtype="float32")
    wave = wave.mean(axis=1)
    if source_rate != sample_rate:
        wave = librosa.resample(wave, orig_sr=source_rate, target_sr=sample_rate)
    wave = np.asarray(wave, dtype=np.float32)
    wave -= wave.mean()
    rms = float(np.sqrt(np.mean(wave**2) + 1e-12))
    wave *= 0.08 / max(rms, 1e-5)
    peak = float(np.max(np.abs(wave)))
    if peak > 0.95:
        wave *= 0.95 / peak
    fade = min(round(0.02 * sample_rate), len(wave) // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return wave


def compose(wave_a, wave_b, label_a, label_b, variant, sample_rate):
    lead = np.zeros(round((1.5 + 0.5 * (variant % 3)) * sample_rate), dtype=np.float32)
    gap = np.zeros(round((0.25 if variant % 2 == 0 else 0.65) * sample_rate), dtype=np.float32)
    between = np.zeros(round((3.0 if variant % 2 == 0 else 5.0) * sample_rate), dtype=np.float32)
    tail = np.zeros(round(2.0 * sample_rate), dtype=np.float32)
    pieces = [lead]
    cursor = len(lead)
    event_spans = {label_a: [], label_b: []}

    def add_piece(piece):
        nonlocal cursor
        pieces.append(piece)
        cursor += len(piece)

    def add_event(label, wave):
        start = cursor
        add_piece(wave)
        event_spans[label].append([seconds(start, sample_rate), seconds(cursor, sample_rate)])

    def add_sequence(order):
        start = cursor
        first_label, first_wave, second_label, second_wave = order
        add_event(first_label, first_wave)
        add_piece(gap)
        add_event(second_label, second_wave)
        return [seconds(start, sample_rate), seconds(cursor, sample_rate)]

    ab = (label_a, wave_a, label_b, wave_b)
    ba = (label_b, wave_b, label_a, wave_a)
    if variant % 2 == 0:
        window_ab = add_sequence(ab)
        add_piece(between)
        window_ba = add_sequence(ba)
        layout = "AB_first"
    else:
        window_ba = add_sequence(ba)
        add_piece(between)
        window_ab = add_sequence(ab)
        layout = "BA_first"
    add_piece(tail)
    return np.concatenate(pieces), {"AB": [window_ab], "BA": [window_ba]}, event_spans, layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--rehearsal-manifest", required=True)
    parser.add_argument("--train-audios", type=int, default=256)
    parser.add_argument("--test-pairs", type=int, default=40)
    parser.add_argument("--test-replicates", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    by_class = defaultdict(list)
    with Path(args.metadata).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_class[row["category"]].append(row["filename"])
    missing = sorted(HELDOUT_CLASSES - set(by_class))
    if missing:
        raise ValueError(f"Missing held-out ESC-50 classes: {missing}")

    train_classes = sorted(set(by_class) - HELDOUT_CLASSES)
    train_pairs = list(itertools.combinations(train_classes, 2))
    test_pairs = list(itertools.combinations(sorted(HELDOUT_CLASSES), 2))
    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    train_pairs = train_pairs[: args.train_audios]
    test_pairs = test_pairs[: args.test_pairs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir)
    manifests = {"train": [], "test": []}
    rehearsal = []

    def build(split, pair_id, label_a, label_b, variant):
        source_a = str(rng.choice(by_class[label_a]))
        source_b = str(rng.choice(by_class[label_b]))
        wave_a = load_clip(audio_dir / source_a, args.sample_rate)
        wave_b = load_clip(audio_dir / source_b, args.sample_rate)
        audio, windows, event_spans, layout = compose(
            wave_a, wave_b, label_a, label_b, variant, args.sample_rate
        )
        filename = f"esc_{split}_{pair_id:04d}_v{variant}.wav"
        sf.write(output_dir / filename, audio, args.sample_rate, subtype="PCM_16")
        a, b = label_a.replace("_", " "), label_b.replace("_", " ")
        templates = [
            ("followed_by", f"{a} followed by {b}", f"{b} followed by {a}"),
            ("and_then", f"{a}, and then {b}", f"{b}, and then {a}"),
        ]
        for template, query_ab, query_ba in templates:
            common = {
                "audio_path": str((output_dir / filename).resolve()),
                "pair_id": pair_id,
                "variant": variant,
                "layout": layout,
                "template": template,
                "event_a": a,
                "event_b": b,
                "window_ab": windows["AB"],
                "window_ba": windows["BA"],
                "source_files": [source_a, source_b],
                "source_dataset": "ESC-50",
            }
            manifests[split].append(
                {**common, "relation": "AB", "caption": query_ab, "annotations": windows["AB"]}
            )
            manifests[split].append(
                {**common, "relation": "BA", "caption": query_ba, "annotations": windows["BA"]}
            )
        if split == "train":
            rehearsal.extend([
                {
                    "audio_path": str((output_dir / filename).resolve()),
                    "caption": a,
                    "annotations": event_spans[label_a],
                    "source_dataset": "ESC-50",
                },
                {
                    "audio_path": str((output_dir / filename).resolve()),
                    "caption": b,
                    "annotations": event_spans[label_b],
                    "source_dataset": "ESC-50",
                },
            ])

    for pair_id, (label_a, label_b) in enumerate(train_pairs):
        build("train", pair_id, label_a, label_b, pair_id % 2)
    for offset, (label_a, label_b) in enumerate(test_pairs):
        for replicate in range(args.test_replicates):
            build("test", 10000 + offset, label_a, label_b, replicate)

    outputs = [
        (args.train_manifest, manifests["train"]),
        (args.test_manifest, manifests["test"]),
        (args.rehearsal_manifest, rehearsal),
    ]
    for path, rows in outputs:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "train_classes": len(train_classes),
        "heldout_classes": sorted(HELDOUT_CLASSES),
        "train_relation_queries": len(manifests["train"]),
        "test_relation_queries": len(manifests["test"]),
        "rehearsal_queries": len(rehearsal),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
