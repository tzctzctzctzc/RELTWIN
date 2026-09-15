#!/usr/bin/env python3
"""Materialize CompA-Order and reconstruct its official paired manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq


_VARIANT_RE = re.compile(r"^(?P<group>.+?)(?P<variant>_rev|_trip)?$")


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
    groups: dict[str, dict[str, dict]] = {}

    for row in table.to_pylist():
        file_name = str(row["file_name"])
        stem = Path(file_name).stem
        match = _VARIANT_RE.match(stem)
        if match is None:
            raise ValueError(f"Unexpected file name: {file_name}")
        group = match.group("group")
        variant = match.group("variant") or "base"
        variant = variant.removeprefix("_")
        if variant in groups.setdefault(group, {}):
            raise ValueError(f"Duplicate {group}/{variant}")

        payload = row["audio"].get("bytes")
        if not payload:
            raise ValueError(f"Missing audio bytes for {file_name}")
        (args.audio_dir / file_name).write_bytes(payload)
        groups[group][variant] = {
            "file": file_name,
            "caption": str(row["answer"]),
        }

    manifest = []
    for group in sorted(groups, key=lambda value: int(value)):
        variants = groups[group]
        if "base" not in variants or "rev" not in variants:
            raise ValueError(f"Incomplete base/rev pair for group {group}")
        item = {
            "group_id": group,
            "pair_file": variants["base"]["file"],
            "pair_caption": variants["base"]["caption"],
            "reversed_pair_file": variants["rev"]["file"],
            "reversed_pair_caption": variants["rev"]["caption"],
        }
        if "trip" in variants:
            item["triplet_file"] = variants["trip"]["file"]
            item["triplet_caption"] = variants["trip"]["caption"]
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
