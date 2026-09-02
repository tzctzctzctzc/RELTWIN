#!/usr/bin/env python3
"""Create rendered-audio-disjoint dev splits and stratified evaluation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from spantool_runtime import load_manifest, validate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    split = subparsers.add_parser("train-dev")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--train-output", type=Path, required=True)
    split.add_argument("--dev-output", type=Path, required=True)
    split.add_argument("--dev-fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=0)

    gate = subparsers.add_parser("gate")
    gate.add_argument("--input", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    gate.add_argument("--size", type=int, default=1000)
    gate.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_stratum(row: dict) -> tuple[str, str, str]:
    count = len(row.get("annotations", []))
    count_bucket = "zero" if count == 0 else "single" if count == 1 else "multi"
    duration = float(row.get("duration", 0.0) or 0.0)
    if duration <= 0:
        duration = max((float(end) for _, end in row.get("annotations", [])), default=1.0)
    coverage = sum(max(0.0, float(end) - float(start)) for start, end in row.get("annotations", []))
    ratio = coverage / max(duration, 1e-6)
    coverage_bucket = "sparse" if ratio <= 0.1 else "medium" if ratio <= 0.3 else "dense"
    duration_bucket = "short" if duration <= 15 else "medium" if duration <= 45 else "long"
    return count_bucket, coverage_bucket, duration_bucket


def stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def train_dev_split(rows: list[dict], fraction: float, seed: int):
    if not 0 < fraction < 1:
        raise ValueError("dev-fraction must be between zero and one")
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["audio_path"])].append(row)
    train, dev = [], []
    for audio_path, group in sorted(groups.items()):
        destination = dev if stable_fraction(seed, audio_path) < fraction else train
        destination.extend(group)
    if not train or not dev:
        raise ValueError("Split produced an empty train or dev partition")
    if {row["audio_path"] for row in train} & {row["audio_path"] for row in dev}:
        raise AssertionError("Audio leakage between train and dev")
    return train, dev


def proportional_allocation(groups: dict, size: int) -> dict:
    total = sum(len(rows) for rows in groups.values())
    exact = {key: size * len(rows) / total for key, rows in groups.items()}
    allocation = {key: min(len(groups[key]), math.floor(value)) for key, value in exact.items()}
    remaining = size - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (exact[key] - allocation[key], len(groups[key]), str(key)),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate requested gate size")
    return allocation


def stratified_gate(rows: list[dict], size: int, seed: int) -> list[dict]:
    if not 0 < size <= len(rows):
        raise ValueError("gate size must be within the manifest size")
    groups = defaultdict(list)
    for row in rows:
        groups[row_stratum(row)].append(row)
    allocation = proportional_allocation(groups, size)
    rng = random.Random(seed)
    selected = []
    for key in sorted(groups):
        candidates = list(groups[key])
        rng.shuffle(candidates)
        selected.extend(candidates[: allocation[key]])
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}|{row.get('qid', row.get('benchmark_id', row['audio_path']))}".encode()
        ).hexdigest()
    )
    return selected


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.input)
    validate_rows(rows, str(args.input))
    if args.mode == "train-dev":
        train, dev = train_dev_split(rows, args.dev_fraction, args.seed)
        write_rows(args.train_output, train)
        write_rows(args.dev_output, dev)
        result = {
            "mode": args.mode,
            "train_rows": len(train),
            "dev_rows": len(dev),
            "train_audio": len({row["audio_path"] for row in train}),
            "dev_audio": len({row["audio_path"] for row in dev}),
        }
    else:
        gate = stratified_gate(rows, args.size, args.seed)
        write_rows(args.output, gate)
        result = {
            "mode": args.mode,
            "input_rows": len(rows),
            "gate_rows": len(gate),
            "strata": {
                "/".join(key): sum(row_stratum(row) == key for row in gate)
                for key in sorted({row_stratum(row) for row in gate})
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
