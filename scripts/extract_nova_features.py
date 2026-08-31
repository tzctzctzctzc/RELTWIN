#!/usr/bin/env python3
"""Extract label-free NOVA features for aligned candidate prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from peft import PeftModel

from nova import counterfactual_features_fast
from spotsound import AudioFlamingo3ForTemporalConditionalGeneration, AudioFlamingo3TemporalProcessor


IDENTITY_KEYS = ("index", "audio", "query", "ground_truth")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--verifier-adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_candidates(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Candidate must be NAME=JSONL: {value}")
        if name in result:
            raise ValueError(f"Duplicate candidate name: {name}")
        result[name] = Path(path)
    return result


def assert_aligned(reference: list[dict], candidate: list[dict], path: Path):
    if len(candidate) != len(reference):
        raise ValueError(f"Row count mismatch for {path}: {len(candidate)} != {len(reference)}")
    for expected, actual in zip(reference, candidate):
        for key in IDENTITY_KEYS:
            if expected.get(key) != actual.get(key):
                raise ValueError(f"Identity mismatch at index {expected.get('index')} for {path}: {key}")


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidate_paths = parse_candidates(args.candidate)
    candidates = {name: load_jsonl(path) for name, path in candidate_paths.items()}
    reference = next(iter(candidates.values()))
    for name, rows in candidates.items():
        assert_aligned(reference, rows, candidate_paths[name])
    if len(manifest) != len(reference):
        raise ValueError(f"Manifest/prediction count mismatch: {len(manifest)} != {len(reference)}")

    processor = AudioFlamingo3TemporalProcessor.from_pretrained(args.base)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.verifier_adapter, torch_device="cpu")
    model = model.eval().to("cuda")

    stop = len(reference) if args.max_rows is None else min(
        len(reference), args.start_index + args.max_rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.is_file() and not args.overwrite:
        completed = {int(row["index"]) for row in load_jsonl(args.output)}
    mode = "w" if args.overwrite else "a"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index in range(args.start_index, stop):
            if index in completed:
                continue
            item = manifest[index]
            audio_path = args.audio_dir / Path(item["audio_path"]).name
            wave, _ = librosa.load(audio_path, sr=16000, mono=True)
            wave = np.asarray(wave, dtype=np.float32)
            row = {
                "index": index,
                "audio": reference[index]["audio"],
                "query": reference[index]["query"],
                "ground_truth": reference[index]["ground_truth"],
                "pair_id": item.get("pair_id"),
                "template": item.get("template"),
                "variant": item.get("variant", 0),
                "relation": item.get("relation"),
                "candidates": {},
            }
            for name, rows in candidates.items():
                prediction = rows[index]["prediction"]
                row["candidates"][name] = {
                    "prediction": prediction,
                    "iou": float(rows[index]["iou"]),
                    "features": counterfactual_features_fast(
                        model, processor, wave, row["query"], prediction
                    ),
                }
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps({"completed": index + 1, "total": stop}), flush=True)


if __name__ == "__main__":
    main()
