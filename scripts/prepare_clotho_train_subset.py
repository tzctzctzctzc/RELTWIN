#!/usr/bin/env python3
"""Materialize a pinned Clotho-Moment train-shard subset for rehearsal."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in sorted(args.shard_dir.glob("train-*.tar")):
        with tarfile.open(shard) as archive:
            members = {Path(member.name).name: member for member in archive if member.isfile()}
            for json_name in sorted(name for name in members if name.endswith(".json")):
                stem = Path(json_name).stem
                wav_name = stem + ".wav"
                if wav_name not in members:
                    continue
                with archive.extractfile(members[json_name]) as handle:
                    metadata = json.load(handle)
                output_audio = audio_dir / wav_name
                if not output_audio.is_file():
                    with archive.extractfile(members[wav_name]) as source, output_audio.open("wb") as target:
                        shutil.copyfileobj(source, target)
                for foreground in metadata.get("fg", []):
                    start = float(foreground["start_time"])
                    end = start + float(foreground["duration"])
                    rows.append(
                        {
                            "audio_path": f"audio/{wav_name}",
                            "caption": foreground["caption"],
                            "annotations": [[start, end]],
                            "duration": 60.0,
                            "qid": foreground.get("qid"),
                            "source_dataset": "Clotho-Moment-train",
                            "source_shard": shard.name,
                        }
                    )
    if not rows:
        raise ValueError("No complete Clotho-Moment train samples found")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "audio_files": len(list(audio_dir.glob('*.wav'))) }))


if __name__ == "__main__":
    main()
