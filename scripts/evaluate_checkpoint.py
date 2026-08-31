#!/usr/bin/env python3
"""Evaluate the official SpotSound adapter on SpotSound-Bench with resumable output."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from peft import PeftModel

from spotsound import (
    AudioFlamingo3ForTemporalConditionalGeneration,
    AudioFlamingo3TemporalProcessor,
    GROUNDING_PROMPT,
    build_conversation,
)


_INTERVAL_RE = re.compile(
    r"from\s*(-?\d+(?:\.\d+)?)\s*s?\s*(?:econds)?\s*to\s*(-?\d+(?:\.\d+)?)\s*s?",
    re.I,
)
_PAIR_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*s\s*(?:-|to|–)\s*(-?\d+(?:\.\d+)?)\s*s",
    re.I,
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
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all remaining rows")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def parse_intervals(answer: str, duration: float) -> list[tuple[float, float]]:
    matches = _INTERVAL_RE.findall(answer) or _PAIR_RE.findall(answer)
    intervals = []
    for start, end in matches:
        start_value, end_value = float(start), float(end)
        if end_value < start_value:
            start_value, end_value = end_value, start_value
        start_value = max(0.0, min(start_value, duration))
        end_value = max(0.0, min(end_value, duration))
        if end_value - start_value > 1e-3:
            intervals.append((start_value, end_value))
    return intervals


def merge_intervals(intervals: list[list[float]] | list[tuple[float, float]]):
    cleaned = sorted((float(start), float(end)) for start, end in intervals if end > start)
    merged: list[list[float]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def temporal_set_iou(ground_truth, prediction) -> float:
    gt = merge_intervals(ground_truth)
    pred = merge_intervals(prediction)
    if not gt or not pred:
        return 0.0

    intersection = 0.0
    gt_index = pred_index = 0
    while gt_index < len(gt) and pred_index < len(pred):
        intersection += max(
            0.0,
            min(gt[gt_index][1], pred[pred_index][1])
            - max(gt[gt_index][0], pred[pred_index][0]),
        )
        if gt[gt_index][1] <= pred[pred_index][1]:
            gt_index += 1
        else:
            pred_index += 1

    gt_length = sum(end - start for start, end in gt)
    pred_length = sum(end - start for start, end in pred)
    union = gt_length + pred_length - intersection
    return intersection / union if union > 0 else 0.0


def load_completed(path: Path) -> dict[int, dict]:
    completed = {}
    if not path.is_file():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            completed[int(record["index"])] = record
    return completed


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    stop = len(annotations) if args.limit <= 0 else min(len(annotations), args.start + args.limit)
    selected_indices = list(range(args.start, stop))
    missing = [
        Path(annotations[index]["audio_path"]).name
        for index in selected_indices
        if not (args.audio_dir / Path(annotations[index]["audio_path"]).name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} audio files; first five: {missing[:5]}")

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
            audio_path = args.audio_dir / Path(item["audio_path"]).name
            wav, _ = librosa.load(audio_path, sr=16000, mono=True)
            wav = np.asarray(wav, dtype=np.float32)
            duration = len(wav) / 16000.0
            conversation = build_conversation(wav, item["caption"], prompt=GROUNDING_PROMPT)
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to("cuda").to(model.dtype)

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            elapsed = time.perf_counter() - started
            answer = processor.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0].strip()
            prediction = parse_intervals(answer, duration)
            iou = temporal_set_iou(item["annotations"], prediction)
            record = {
                "status": "ok",
                "index": index,
                "audio": audio_path.name,
                "query": item["caption"],
                "ground_truth": item["annotations"],
                "raw_answer": answer,
                "prediction": prediction,
                "iou": iou,
                "duration_seconds": duration,
                "input_tokens": int(inputs["input_ids"].shape[1]),
                "inference_seconds": elapsed,
                "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            }
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            completed[index] = record
            print(json.dumps(record, ensure_ascii=False), flush=True)

    records = [completed[index] for index in selected_indices if index in completed]
    ious = np.asarray([record["iou"] for record in records], dtype=np.float64)
    summary = {
        "base": str(args.base.resolve()),
        "adapter": str(args.adapter.resolve()),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "load_seconds": load_seconds,
        "expected_count": len(selected_indices),
        "completed_count": len(records),
        "R1@0.3": float((ious >= 0.3).mean() * 100) if len(ious) else None,
        "R1@0.5": float((ious >= 0.5).mean() * 100) if len(ious) else None,
        "mIoU": float(ious.mean() * 100) if len(ious) else None,
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
