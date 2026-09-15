#!/usr/bin/env python3
"""Materialize CompA-Order and reconstruct its official paired manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = pq.read_table(args.parquet)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = table.to_pylist()
    for row in rows:
        file_name = str(row["file_name"])
        payload = row["audio"].get("bytes")
        if not payload:
            raise ValueError(f"Missing audio bytes for {file_name}")
        destination = args.audio_dir / file_name
        if not destination.is_file():
            destination.write_bytes(payload)

    manifest = []
    index = 0
    while index < len(rows):
        pair = rows[index]
        if Path(str(pair["file_name"])).stem.endswith(("_rev", "_trip")):
            raise ValueError(f"Expected pair row at index {index}: {pair['file_name']}")
        if index + 1 >= len(rows):
            raise ValueError("CompA rows end with an incomplete pair")
        reversed_pair = rows[index + 1]
        if not Path(str(reversed_pair["file_name"])).stem.endswith("_rev"):
            raise ValueError(
                f"Expected reversed row at index {index + 1}: {reversed_pair['file_name']}"
            )
        item = {
            "group_id": str(len(manifest) + 1),
            "pair_file": str(pair["file_name"]),
            "pair_caption": str(pair["answer"]),
            "reversed_pair_file": str(reversed_pair["file_name"]),
            "reversed_pair_caption": str(reversed_pair["answer"]),
        }
        index += 2
        if index < len(rows) and Path(str(rows[index]["file_name"])).stem.endswith("_trip"):
            item["triplet_file"] = str(rows[index]["file_name"])
            item["triplet_caption"] = str(rows[index]["answer"])
            index += 1
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
