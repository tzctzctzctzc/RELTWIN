#!/usr/bin/env python3
"""Write a deterministic SHA-256 manifest for committed result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/manifest.json"))
    args = parser.parse_args()
    output = args.output.resolve()
    files = [path for path in args.root.rglob("*") if path.is_file() and path.resolve() != output]
    manifest = {
        "algorithm": "sha256",
        "root": args.root.as_posix(),
        "files": [
            {
                "path": path.relative_to(args.root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
