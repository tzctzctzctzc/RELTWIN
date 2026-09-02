#!/usr/bin/env python3
"""Create an exact-size deterministic, audio-grouped and stratified pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from interval_metrics import normalize_intervals
from nova_safe import canonical_key, index_rows, load_jsonl, sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _bucket(value: float, first: float, second: float) -> str:
    if value <= first:
        return "low"
    if value <= second:
        return "medium"
    return "high"


def stratum(row: dict, prediction: dict) -> str:
    duration = float(
        prediction.get(
            "duration_seconds",
            prediction.get("duration", row.get("duration_seconds", row.get("duration"))),
        )
    )
    ground_truth = normalize_intervals(
        prediction.get("ground_truth", row.get("ground_truth", row.get("annotations", []))),
        duration,
    )
    count = "single" if len(ground_truth) == 1 else ("multi" if ground_truth else "empty")
    coverage = sum(end - start for start, end in ground_truth) / duration if duration else 0.0
    if ground_truth:
        midpoint = sum((start + end) / 2 for start, end in ground_truth) / len(ground_truth) / duration
    else:
        midpoint = 0.5
    query = prediction.get("query", row.get("query", row.get("caption", "")))
    query_bucket = _bucket(len(str(query).split()), 5, 10)
    return "|".join((count, _bucket(coverage, 0.1, 0.3), _bucket(midpoint, 1 / 3, 2 / 3), query_bucket))


def choose_groups(groups: dict[str, list[dict]], size: int, seed: int) -> list[dict]:
    if size <= 0:
        raise ValueError("Pilot size must be positive")
    if size > sum(len(rows) for rows in groups.values()):
        raise ValueError("Pilot is larger than the available row count")
    buckets: dict[str, list[str]] = defaultdict(list)
    for audio_group, rows in groups.items():
        majority = Counter(row["stratum"] for row in rows).most_common(1)[0][0]
        buckets[majority].append(audio_group)
    for names in buckets.values():
        names.sort(key=lambda name: _hash(seed, name))
    selected = []
    selected_count = 0
    active = sorted(buckets)
    while active and selected_count < size:
        next_active = []
        for bucket in active:
            names = buckets[bucket]
            while names:
                audio_group = names.pop(0)
                group_size = len(groups[audio_group])
                if selected_count + group_size <= size:
                    selected.append(audio_group)
                    selected_count += group_size
                    break
            if names:
                next_active.append(bucket)
            if selected_count == size:
                break
        active = next_active
    if selected_count != size:
        remaining = sorted(
            (name for name in groups if name not in selected),
            key=lambda name: _hash(seed + 1, name),
        )
        for audio_group in remaining:
            group_size = len(groups[audio_group])
            if selected_count + group_size <= size:
                selected.append(audio_group)
                selected_count += group_size
            if selected_count == size:
                break
    if selected_count != size:
        raise RuntimeError(f"Could not preserve audio groups while selecting exactly {size} rows")
    return [row for audio_group in selected for row in groups[audio_group]]


def main():
    args = parse_args()
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = {
        canonical_key(row, args.benchmark, index): row
        for index, row in enumerate(manifest_rows)
    }
    predictions = index_rows(load_jsonl(args.incumbent), args.benchmark)
    if set(predictions) - set(manifest):
        raise ValueError("Incumbent predictions contain rows absent from the manifest")
    excluded_audio = set()
    for path in args.exclude:
        for row in json.loads(path.read_text(encoding="utf-8")):
            excluded_audio.add(Path(str(row["audio_group"])).name)
    groups: dict[str, list[dict]] = defaultdict(list)
    for key in sorted(predictions):
        prediction = predictions[key]
        item = manifest[key]
        audio = prediction.get("audio", item.get("audio", item.get("audio_path")))
        audio_group = Path(str(audio)).name
        if audio_group in excluded_audio:
            continue
        output = {
            "benchmark": args.benchmark,
            "source": args.source,
            "source_index": key[1],
            "audio_group": audio_group,
            "stratum": stratum(item, prediction),
        }
        groups[audio_group].append(output)
    selected = choose_groups(groups, args.size, args.seed)
    selected.sort(key=lambda row: row["source_index"])
    payload = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "benchmark": args.benchmark,
        "rows": len(selected),
        "audio_groups": len({row["audio_group"] for row in selected}),
        "strata": dict(Counter(row["stratum"] for row in selected)),
        "seed": args.seed,
        "manifest_sha256": sha256_file(args.manifest),
        "incumbent_sha256": sha256_file(args.incumbent),
        "output_sha256": sha256_file(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

