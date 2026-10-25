#!/usr/bin/env python3
"""Build strictly aligned geometry and counterfactual features for NOVA-Safe v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from interval_metrics import normalize_intervals, temporal_set_iou
from nova_safe import (
    SCHEMA_VERSION,
    canonical_key,
    hard_guard,
    index_rows,
    load_jsonl,
    pair_feature_map,
    sha256_file,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="NAME=JSONL")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--verifier-adapter", type=Path)
    parser.add_argument("--feature-mode", choices=("geometry", "keep", "full"), default="full")
    parser.add_argument("--drop-replacement", choices=("silence", "neighbor", "matched_noise"), default="matched_noise")
    parser.add_argument("--replacement-seed", type=int, default=20260902)
    parser.add_argument("--context-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    paths = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"Candidate must be NAME=JSONL: {value}")
        if name in paths:
            raise ValueError(f"Duplicate candidate name: {name}")
        paths[name] = Path(raw_path)
    return paths


def manifest_index(rows: list[dict], benchmark: str) -> dict[tuple[str, int], dict]:
    indexed = {}
    for fallback_index, row in enumerate(rows):
        key = canonical_key(row, benchmark, fallback_index)
        if key in indexed:
            raise ValueError(f"Duplicate manifest key: {key}")
        indexed[key] = row
    return indexed


def _duration(candidate_rows: list[dict], manifest_row: dict) -> float:
    values = []
    for row in candidate_rows:
        value = row.get("duration_seconds", row.get("duration"))
        if value is not None:
            values.append(float(value))
    manifest_value = manifest_row.get("duration_seconds", manifest_row.get("duration"))
    if manifest_value is not None:
        values.append(float(manifest_value))
    if not values:
        raise ValueError("Duration is absent from predictions and manifest")
    if max(values) - min(values) > 1e-6:
        raise ValueError(f"Duration mismatch: {values}")
    return values[0]


def _identity(candidate_rows: list[dict], manifest_row: dict, duration: float) -> tuple[str, str, list]:
    audios = [row.get("audio", row.get("audio_path")) for row in candidate_rows]
    manifest_audio = manifest_row.get("audio", manifest_row.get("audio_path"))
    if manifest_audio:
        audios.append(manifest_audio)
    audio_names = {Path(str(value)).name for value in audios if value is not None}
    if len(audio_names) != 1:
        raise ValueError(f"Audio identity mismatch: {sorted(audio_names)}")
    queries = [row.get("query", row.get("caption")) for row in candidate_rows]
    manifest_query = manifest_row.get("query", manifest_row.get("caption"))
    if manifest_query is not None:
        queries.append(manifest_query)
    query_values = {str(value) for value in queries if value is not None}
    if len(query_values) != 1:
        raise ValueError("Query identity mismatch")
    ground_truths = []
    for row in [*candidate_rows, manifest_row]:
        value = row.get("ground_truth", row.get("annotations"))
        if value is not None:
            ground_truths.append(normalize_intervals(value, duration))
    if not ground_truths:
        ground_truth = []
    else:
        serialized = {json.dumps(value) for value in ground_truths}
        if len(serialized) != 1:
            raise ValueError("Ground-truth identity mismatch")
        ground_truth = ground_truths[0]
    return next(iter(audio_names)), next(iter(query_values)), ground_truth


def validate_alignment(
    candidates: dict[str, dict[tuple[str, int], dict]],
    manifest: dict[tuple[str, int], dict],
) -> list[tuple[str, int]]:
    key_sets = {name: set(rows) for name, rows in candidates.items()}
    reference_name = next(iter(candidates))
    reference = key_sets[reference_name]
    for name, keys in key_sets.items():
        if keys != reference:
            missing = sorted(reference - keys)[:5]
            extra = sorted(keys - reference)[:5]
            raise ValueError(f"Candidate key mismatch for {name}; missing={missing}, extra={extra}")
    missing_manifest = reference - set(manifest)
    if missing_manifest:
        raise ValueError(f"Manifest is missing prediction keys: {sorted(missing_manifest)[:5]}")
    return sorted(reference)


def resolve_audio(audio_dir: Path, audio_name: str, manifest_row: dict) -> Path:
    raw = manifest_row.get("audio_path", manifest_row.get("audio", audio_name))
    candidates = [Path(str(raw)), audio_dir / str(raw), audio_dir / Path(str(raw)).name, audio_dir / audio_name]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Audio not found for {audio_name}; tried {[str(path) for path in candidates]}")


def _load_verifier(args):
    if args.feature_mode == "geometry":
        return None, None
    if args.base is None or args.verifier_adapter is None or args.audio_dir is None:
        raise ValueError("full mode requires --base, --verifier-adapter, and --audio-dir")
    import torch
    from peft import PeftModel
    from spotsound import AudioFlamingo3ForTemporalConditionalGeneration, AudioFlamingo3TemporalProcessor

    processor = AudioFlamingo3TemporalProcessor.from_pretrained(args.base)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.verifier_adapter, torch_device="cpu")
    return model.eval().to("cuda"), processor


def main():
    args = parse_args()
    candidate_paths = parse_named_paths(args.candidate)
    if args.incumbent not in candidate_paths:
        raise ValueError("The incumbent must also be supplied as a candidate")
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest_rows, list):
        raise ValueError("Manifest must contain a JSON list")
    manifest = manifest_index(manifest_rows, args.benchmark)
    candidates = {
        name: index_rows(load_jsonl(path), args.benchmark)
        for name, path in candidate_paths.items()
    }
    all_keys = validate_alignment(candidates, manifest)
    if args.selection_manifest:
        selection_rows = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        selection_keys = {
            canonical_key(row, args.benchmark, fallback_index)
            for fallback_index, row in enumerate(selection_rows)
        }
        unknown = selection_keys - set(all_keys)
        if unknown:
            raise ValueError(f"Selection manifest contains unknown keys: {sorted(unknown)[:5]}")
        keys = [key for key in all_keys if key in selection_keys]
    else:
        keys = all_keys
    stop = len(keys) if args.max_rows is None else min(len(keys), args.start_index + args.max_rows)
    keys = keys[args.start_index:stop]

    completed = set()
    if args.output.is_file() and not args.overwrite:
        completed = {canonical_key(row, args.benchmark) for row in load_jsonl(args.output)}
    mode = "w" if args.overwrite else "a"
    model, processor = _load_verifier(args)
    candidate_hashes = {name: sha256_file(path) for name, path in candidate_paths.items()}
    artifact_hashes = {
        "manifest": sha256_file(args.manifest),
        **{f"candidate:{name}": digest for name, digest in candidate_hashes.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open(mode, encoding="utf-8") as handle:
        for offset, key in enumerate(keys):
            if key in completed:
                continue
            prediction_rows = [candidates[name][key] for name in candidate_paths]
            manifest_row = manifest[key]
            duration = _duration(prediction_rows, manifest_row)
            audio, query, ground_truth = _identity(prediction_rows, manifest_row, duration)
            row = {
                "feature_schema_version": SCHEMA_VERSION,
                "benchmark": args.benchmark,
                "source": args.source,
                "source_index": key[1],
                "audio": audio,
                "audio_group": audio,
                "query": query,
                "duration_seconds": duration,
                "ground_truth": ground_truth,
                "incumbent": args.incumbent,
                "artifact_hashes": artifact_hashes,
                "intervention": {
                    "feature_mode": args.feature_mode,
                    "drop_replacement": args.drop_replacement,
                    "replacement_seed": args.replacement_seed + key[1],
                    "context_seconds": args.context_seconds,
                },
                "candidates": {},
            }
            for name, prediction_row in zip(candidate_paths, prediction_rows):
                prediction = prediction_row.get("prediction", [])
                iou = prediction_row.get("iou")
                if iou is None and ground_truth:
                    iou = temporal_set_iou(ground_truth, prediction)
                row["candidates"][name] = {
                    "prediction": prediction,
                    "iou": float(iou) if iou is not None else None,
                    "features": {},
                }
            incumbent = row["candidates"][args.incumbent]
            eligible = [
                name
                for name, candidate in row["candidates"].items()
                if name != args.incumbent and hard_guard(incumbent, candidate, duration) is None
            ]
            if args.feature_mode in ("keep", "full") and eligible:
                import librosa
                from nova import counterfactual_features_fast, counterfactual_keep_features_fast

                audio_path = resolve_audio(args.audio_dir, audio, manifest_row)
                wave, _ = librosa.load(audio_path, sr=16000, mono=True)
                wave = np.asarray(wave, dtype=np.float32)
                for name in [args.incumbent, *eligible]:
                    extractor = (
                        counterfactual_keep_features_fast
                        if args.feature_mode == "keep"
                        else counterfactual_features_fast
                    )
                    keyword_arguments = {}
                    if args.feature_mode == "full":
                        keyword_arguments = {
                            "drop_replacement": args.drop_replacement,
                            "replacement_seed": args.replacement_seed + key[1],
                            "context_seconds": args.context_seconds,
                        }
                    row["candidates"][name]["features"] = extractor(
                        model,
                        processor,
                        wave,
                        query,
                        row["candidates"][name]["prediction"],
                        **keyword_arguments,
                    )
            row["pair_features"] = {
                name: pair_feature_map(row, name)
                for name in eligible
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"completed": offset + 1, "total": len(keys), "key": key}), flush=True)


if __name__ == "__main__":
    main()
