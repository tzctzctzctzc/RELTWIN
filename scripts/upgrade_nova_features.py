#!/usr/bin/env python3
"""Upgrade an aligned legacy NOVA feature JSONL into the NOVA-Safe v2 schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova_safe import SCHEMA_VERSION, load_jsonl, pair_feature_map, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    upgraded = []
    for fallback_index, legacy in enumerate(rows):
        if args.incumbent not in legacy["candidates"]:
            raise ValueError(f"Missing incumbent {args.incumbent}")
        row = {
            "feature_schema_version": SCHEMA_VERSION,
            "benchmark": args.benchmark,
            "source": args.source,
            "source_index": int(legacy.get("source_index", legacy.get("index", fallback_index))),
            "audio": Path(str(legacy["audio"])).name,
            "audio_group": Path(str(legacy["audio"])).name,
            "query": legacy["query"],
            "duration_seconds": args.duration,
            "ground_truth": legacy.get("ground_truth", []),
            "incumbent": args.incumbent,
            "candidates": legacy["candidates"],
            "legacy_feature_sha256": sha256_file(args.input),
        }
        row["pair_features"] = {
            name: pair_feature_map(row, name)
            for name in row["candidates"]
            if name != args.incumbent
        }
        upgraded.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in upgraded:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(upgraded), "output_sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()

