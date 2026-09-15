#!/usr/bin/env python3
"""Resumable same-harness evaluation for the public TimeAudio checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from interval_metrics import temporal_set_iou
from timeaudio_reltwin import (
    build_timeaudio_sample,
    load_timeaudio_model,
    move_sample,
    parse_timeaudio_intervals,
    prepare_audio_features,
    resolve_audio_path,
    sha256,
    timeaudio_question,
    write_json,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeaudio-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--whisper", type=Path, required=True)
    parser.add_argument("--beats", type=Path, required=True)
    parser.add_argument("--bert", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--delta", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-label", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def read_completed(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    if content and not content.endswith("\n"):
        raise ValueError(f"Interrupted JSONL tail: {path}")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    completed = {int(row["index"]): row for row in rows if row.get("status") == "ok"}
    if len(completed) != len(rows):
        raise ValueError("Predictions contain duplicate indices or non-ok rows")
    return completed


def main() -> int:
    args = arguments()
    torch.manual_seed(0)
    np.random.seed(0)
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    stop = len(annotations) if args.limit <= 0 else min(len(annotations), args.start + args.limit)
    indices = list(range(args.start, stop))
    paths = {index: resolve_audio_path(args.audio_dir, annotations[index]) for index in indices}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} audio files; first five: {missing[:5]}")
    if args.predictions.exists() and not args.predictions.is_file():
        raise ValueError("Predictions path must be a JSONL file")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.predictions)

    from transformers import WhisperFeatureExtractor

    load_started = time.perf_counter()
    model, cfg, load_report = load_timeaudio_model(
        args.timeaudio_repo,
        args.config,
        args.llama,
        args.whisper,
        args.beats,
        args.bert,
        args.checkpoint,
        args.delta,
    )
    model = model.eval().to("cuda")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper)
    generation = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": int(cfg.generate.get("num_beams", 4)),
        "do_sample": False,
        "min_length": int(cfg.generate.get("min_length", 1)),
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": float(cfg.generate.get("repetition_penalty", 1.0)),
        "length_penalty": float(cfg.generate.get("length_penalty", 1.0)),
    }
    load_seconds = time.perf_counter() - load_started

    previous_audio = None
    previous_features = None
    with args.predictions.open("a", encoding="utf-8") as handle:
        for index in indices:
            if index in completed:
                continue
            row, audio_path = annotations[index], paths[index]
            if audio_path != previous_audio:
                previous_features = prepare_audio_features(audio_path, feature_extractor)
                previous_audio = audio_path
            features = previous_features
            duration = float(features["duration"].item())
            sample = move_sample(build_timeaudio_sample(features, row["caption"]))
            prompt = model.prompt_template.format(timeaudio_question(row["caption"]))
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                answers = model.generate(sample, generation, prompts=[prompt])
            elapsed = time.perf_counter() - started
            answer = answers[0].strip()
            prediction = parse_timeaudio_intervals(answer, duration)
            iou = temporal_set_iou(row["annotations"], prediction)
            record = {
                "status": "ok",
                "index": index,
                "audio": audio_path.name,
                "query": row["caption"],
                "ground_truth": row["annotations"],
                "raw_answer": answer,
                "prediction": prediction,
                "iou": iou,
                "duration_seconds": duration,
                "inference_seconds": elapsed,
                "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "benchmark": row.get("benchmark"),
                "benchmark_id": row.get("benchmark_id", row.get("qid")),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed[index] = record
            print(json.dumps(record, ensure_ascii=False), flush=True)

    records = [completed[index] for index in indices if index in completed]
    ious = np.asarray([record["iou"] for record in records], dtype=np.float64)
    valid = np.asarray([bool(record["prediction"]) for record in records])
    summary = {
        "model": args.model_label or ("TimeAudio adapted" if args.delta else "TimeAudio official"),
        "official_checkpoint_sha256": sha256(args.checkpoint),
        "delta_sha256": sha256(args.delta) if args.delta else None,
        "manifest_sha256": sha256(args.annotations),
        "load_report": load_report,
        "device": torch.cuda.get_device_name(0),
        "expected_count": len(indices),
        "completed_count": len(records),
        "mIoU": float(100 * ious.mean()) if len(ious) else None,
        "R1@0.3": float(100 * (ious >= 0.3).mean()) if len(ious) else None,
        "R1@0.5": float(100 * (ious >= 0.5).mean()) if len(ious) else None,
        "valid_output_rate": float(100 * valid.mean()) if len(valid) else None,
        "parse_failure_count": int((~valid).sum()) if len(valid) else None,
        "load_seconds": load_seconds,
        "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in records])) if records else None,
        "predictions_sha256": sha256(args.predictions),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(records) == len(indices) else 3


if __name__ == "__main__":
    raise SystemExit(main())
