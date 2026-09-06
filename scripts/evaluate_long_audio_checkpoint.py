#!/usr/bin/env python3
"""Evaluate SpotSound beyond the AF3 600-second processor limit.

Long recordings are split into fixed overlapping windows.  The frozen
SpotSound existence head ranks windows without labels; grounding is performed
only in the highest-scoring window and local timestamps are mapped back to the
recording timeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from peft import PeftModel

from evaluate_checkpoint import (
    load_completed,
    make_conversation,
    parse_intervals,
    resolve_audio_path,
    temporal_set_iou,
)
from long_audio import chunk_bounds, offset_intervals
from nova import binary_first_token_log_odds
from spotsound import (
    AudioFlamingo3ForTemporalConditionalGeneration,
    AudioFlamingo3TemporalProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--chunk-seconds", type=float, default=600.0)
    parser.add_argument("--stride-seconds", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_seconds > 600.0:
        raise ValueError("AF3 processor hard limit requires chunk_seconds <= 600")
    torch.manual_seed(0)
    np.random.seed(0)

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    stop = (
        len(annotations)
        if args.limit <= 0
        else min(len(annotations), args.start + args.limit)
    )
    selected_indices = list(range(args.start, stop))
    missing = [
        annotations[index]["audio_path"]
        for index in selected_indices
        if not resolve_audio_path(args.audio_dir, annotations[index]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} audio files; first five: {missing[:5]}"
        )

    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.predictions)

    load_started = time.perf_counter()
    processor = AudioFlamingo3TemporalProcessor.from_pretrained(args.base)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        torch_device="cpu",
        is_trainable=False,
    ).merge_and_unload()
    model = model.eval().to("cuda")
    load_seconds = time.perf_counter() - load_started

    with args.predictions.open("a", encoding="utf-8") as output_handle:
        for index in selected_indices:
            if index in completed:
                continue
            item = annotations[index]
            audio_path = resolve_audio_path(args.audio_dir, item)
            wav, _ = librosa.load(audio_path, sr=16000, mono=True)
            wav = np.asarray(wav, dtype=np.float32)
            duration = len(wav) / 16000.0
            bounds = chunk_bounds(
                len(wav),
                16000,
                chunk_seconds=args.chunk_seconds,
                stride_seconds=args.stride_seconds,
            )

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            if len(bounds) == 1:
                detection_scores: list[float | None] = [None]
                selected_chunk = 0
            else:
                detection_scores = [
                    binary_first_token_log_odds(
                        model,
                        processor,
                        wav[left:right],
                        item["caption"],
                    )
                    for left, right in bounds
                ]
                selected_chunk = int(np.argmax(detection_scores))

            left, right = bounds[selected_chunk]
            chunk = wav[left:right]
            conversation = make_conversation(chunk, item["caption"], "spotsound")
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to("cuda").to(model.dtype)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            answer = processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0].strip()
            local_prediction = parse_intervals(answer, len(chunk) / 16000.0)
            offset_seconds = left / 16000.0
            prediction = offset_intervals(local_prediction, offset_seconds, duration)
            elapsed = time.perf_counter() - started
            record = {
                "status": "ok",
                "index": index,
                "audio": audio_path.name,
                "query": item["caption"],
                "ground_truth": item["annotations"],
                "raw_answer": answer,
                "local_prediction": local_prediction,
                "prediction": prediction,
                "iou": temporal_set_iou(item["annotations"], prediction),
                "duration_seconds": duration,
                "chunk_seconds": args.chunk_seconds,
                "stride_seconds": args.stride_seconds,
                "chunk_count": len(bounds),
                "chunk_bounds_seconds": [
                    [chunk_left / 16000.0, chunk_right / 16000.0]
                    for chunk_left, chunk_right in bounds
                ],
                "chunk_detection_log_odds": detection_scores,
                "selected_chunk": selected_chunk,
                "selected_chunk_start_seconds": offset_seconds,
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "inference_seconds": elapsed,
                "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "benchmark": item.get("benchmark"),
                "benchmark_id": item.get("benchmark_id", item.get("qid")),
                "annotation_overshoot_seconds": item.get(
                    "annotation_overshoot_seconds", 0.0
                ),
            }
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            completed[index] = record
            print(json.dumps(record, ensure_ascii=False), flush=True)

    records = [completed[index] for index in selected_indices if index in completed]
    ious = np.asarray([record["iou"] for record in records], dtype=np.float64)
    valid_ious = np.asarray(
        [
            record["iou"]
            for record in records
            if float(record.get("annotation_overshoot_seconds", 0.0)) <= 1.0
        ],
        dtype=np.float64,
    )
    summary = {
        "base": str(args.base.resolve()),
        "adapter": str(args.adapter.resolve()),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "method": "max_existence_evidence_over_overlapping_windows",
        "chunk_seconds": args.chunk_seconds,
        "stride_seconds": args.stride_seconds,
        "load_seconds": load_seconds,
        "expected_count": len(selected_indices),
        "completed_count": len(records),
        "released_rows_mIoU": float(ious.mean() * 100) if len(ious) else None,
        "valid_rows": int(len(valid_ious)),
        "valid_rows_mIoU": (
            float(valid_ious.mean() * 100) if len(valid_ious) else None
        ),
        "R1@0.3": float((ious >= 0.3).mean() * 100) if len(ious) else None,
        "R1@0.5": float((ious >= 0.5).mean() * 100) if len(ious) else None,
        "mean_inference_seconds": (
            float(np.mean([record["inference_seconds"] for record in records]))
            if records
            else None
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(records) == len(selected_indices) else 3


if __name__ == "__main__":
    raise SystemExit(main())
