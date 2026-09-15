#!/usr/bin/env python3
"""Materialize CompA-Order and reconstruct its official paired manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--official-csv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = pq.read_table(args.parquet)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    parquet_rows = table.to_pylist()
    available_pairs = {
        (str(row["file_name"]), str(row["answer"])) for row in parquet_rows
    }
    for row in parquet_rows:
        file_name = str(row["file_name"])
        payload = row["audio"].get("bytes")
        if not payload:
            raise ValueError(f"Missing audio bytes for {file_name}")
        destination = args.audio_dir / file_name
        if not destination.is_file():
            destination.write_bytes(payload)

    manifest = []
    with args.official_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for group_index, row in enumerate(csv.DictReader(handle), 1):
            item = {
                "group_id": str(group_index),
                "pair_file": row["pair_file"],
                "pair_caption": row["pair_caption"],
                "reversed_pair_file": row["reversed_pair_file"],
                "reversed_pair_caption": row["reversed_pair_caption"],
            }
            for file_key, caption_key in (
                ("pair_file", "pair_caption"),
                ("reversed_pair_file", "reversed_pair_caption"),
            ):
                if (item[file_key], item[caption_key]) not in available_pairs:
                    raise ValueError(
                        f"HF mirror disagrees with official CSV at group {group_index}: "
                        f"{item[file_key]} / {item[caption_key]}"
                    )
            if row["triplet_file"] != "-":
                item["triplet_file"] = row["triplet_file"]
                item["triplet_caption"] = row["triplet_caption"]
                if (item["triplet_file"], item["triplet_caption"]) not in available_pairs:
                    raise ValueError(
                        f"HF mirror disagrees with official triplet at group {group_index}"
                    )
            manifest.append(item)

    triplets = sum("triplet_file" in item for item in manifest)
    if len(manifest) != 400 or triplets != 100:
        raise ValueError(
            f"Expected the official 400 groups and 100 triplets; got {len(manifest)} and {triplets}"
        )
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": table.num_rows,
                "groups": len(manifest),
                "triplets": triplets,
                "manifest": str(args.manifest.resolve()),
                "audio_dir": str(args.audio_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
