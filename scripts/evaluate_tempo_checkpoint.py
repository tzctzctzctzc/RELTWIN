#!/usr/bin/env python3
"""Evaluate a released TEMPO checkpoint, optionally with a RelTwin LoRA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from peft import PeftModel
from transformers import AudioFlamingo3Processor

from tempo_reltwin import (
    GROUNDING_TAG,
    TempoAudioFlamingo3ForConditionalGeneration,
    grounding_conversation,
    load_time_projector,
    parse_atomic_intervals,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--time-projector", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def audio_path(root: Path, row: dict):
    path = root / row["audio_path"]
    return path if path.is_file() else root / Path(row["audio_path"]).name


def main():
    args = arguments()
    rows = json.loads(args.annotations.read_text(encoding="utf-8"))
    stop = len(rows) if args.limit <= 0 else min(len(rows), args.start + args.limit)
    selected = list(range(args.start, stop))
    completed = {}
    if args.predictions.is_file():
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[int(item["index"])] = item

    processor = AudioFlamingo3Processor.from_pretrained(args.base)
    model = TempoAudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_time_projector(model, args.time_projector)
    if args.adapter:
        model = PeftModel.from_pretrained(
            model, args.adapter, torch_device="cpu", is_trainable=False
        ).merge_and_unload()
    model = model.eval().to("cuda")

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("a", encoding="utf-8") as handle:
        for index in selected:
            if index in completed:
                continue
            row = rows[index]
            path = audio_path(args.audio_dir, row)
            wave, _ = librosa.load(path, sr=16000, mono=True)
            wave = np.asarray(wave, dtype=np.float32)
            duration = len(wave) / 16000.0
            question = row.get("source_question") or row["caption"]
            if not question.startswith(GROUNDING_TAG):
                question = f"{GROUNDING_TAG} {question}"
            inputs = processor.apply_chat_template(
                grounding_conversation(wave, question),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to("cuda").to(model.dtype)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            elapsed = time.perf_counter() - started
            answer = processor.batch_decode(
                output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=False
            )[0].strip()
            prediction = parse_atomic_intervals(answer, duration)
            record = {
                "status": "ok",
                "index": index,
                "audio": path.name,
                "query": row["caption"],
                "ground_truth": row["annotations"],
                "raw_answer": answer,
                "prediction": prediction,
                "duration_seconds": duration,
                "inference_seconds": elapsed,
                "benchmark": row.get("benchmark"),
                "benchmark_id": row.get("benchmark_id"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed[index] = record
            print(json.dumps(record, ensure_ascii=False), flush=True)

    measured = [completed[index] for index in selected if index in completed]
    summary = {
        "base": str(args.base.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "expected_count": len(selected),
        "completed_count": len(measured),
        "mean_inference_seconds": float(
            np.mean([row["inference_seconds"] for row in measured])
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
