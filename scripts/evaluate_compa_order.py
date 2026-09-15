#!/usr/bin/env python3
"""Evaluate an Audio Flamingo checkpoint on CompA-Order by caption likelihood."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from peft import PeftModel

from spotsound import (
    AudioFlamingo3ForTemporalConditionalGeneration,
    AudioFlamingo3TemporalProcessor,
)


DESCRIPTION_PROMPT = (
    "Write one concise sentence that accurately describes the order and overlap "
    "of the sound events in this audio."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def conversation(wav: np.ndarray, caption: str | None = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav},
                {"type": "text", "text": DESCRIPTION_PROMPT},
            ],
        }
    ]
    if caption is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": caption}]}
        )
    return messages


def caption_score(model, processor, wav: np.ndarray, caption: str) -> tuple[float, int]:
    prefix = processor.apply_chat_template(
        conversation(wav),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to("cuda").to(model.dtype)
    full = processor.apply_chat_template(
        conversation(wav, caption),
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    ).to("cuda").to(model.dtype)
    prefix_length = int(prefix["input_ids"].shape[1])
    full_length = int(full["input_ids"].shape[1])
    if full_length <= prefix_length:
        raise ValueError("Assistant caption did not add any tokens")
    if not torch.equal(prefix["input_ids"], full["input_ids"][:, :prefix_length]):
        raise ValueError("Chat-template prefix is not stable with the assistant caption")
    labels = full["input_ids"].clone()
    labels[:, :prefix_length] = -100
    with torch.inference_mode():
        output = model(**full, labels=labels, use_cache=False)
    return -float(output.loss), full_length - prefix_length


def load_completed(path: Path) -> dict[int, dict]:
    completed = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    completed[int(row["index"])] = row
    return completed


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    stop = len(manifest) if args.limit <= 0 else min(len(manifest), args.start + args.limit)
    selected = list(range(args.start, stop))
    completed = load_completed(args.predictions)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    processor = AudioFlamingo3TemporalProcessor.from_pretrained(args.base)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model, args.adapter, torch_device="cpu", is_trainable=False
    ).merge_and_unload()
    model = model.eval().to("cuda")

    with args.predictions.open("a", encoding="utf-8") as output_handle:
        for index in selected:
            if index in completed:
                continue
            item = manifest[index]
            audio0, _ = librosa.load(args.audio_dir / item["pair_file"], sr=16000, mono=True)
            audio1, _ = librosa.load(
                args.audio_dir / item["reversed_pair_file"], sr=16000, mono=True
            )
            audio0 = np.asarray(audio0, dtype=np.float32)
            audio1 = np.asarray(audio1, dtype=np.float32)
            started = time.perf_counter()
            s00, n00 = caption_score(model, processor, audio0, item["pair_caption"])
            s10, n10 = caption_score(model, processor, audio0, item["reversed_pair_caption"])
            s01, n01 = caption_score(model, processor, audio1, item["pair_caption"])
            s11, n11 = caption_score(model, processor, audio1, item["reversed_pair_caption"])
            text_correct = s00 > s10 and s11 > s01
            audio_correct = s00 > s01 and s11 > s10
            row = {
                "status": "ok",
                "index": index,
                "group_id": item["group_id"],
                "scores": {"s00": s00, "s10": s10, "s01": s01, "s11": s11},
                "caption_tokens": {"s00": n00, "s10": n10, "s01": n01, "s11": n11},
                "text_correct": text_correct,
                "audio_correct": audio_correct,
                "group_correct": text_correct and audio_correct,
                "elapsed_seconds": time.perf_counter() - started,
            }
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()
            completed[index] = row
            print(json.dumps(row, ensure_ascii=False), flush=True)

    rows = [completed[index] for index in selected if index in completed]
    summary = {
        "protocol": "CompA-Order official Text/Audio/Group inequalities",
        "similarity": "length-normalized conditional caption log-likelihood",
        "records": len(rows),
        "expected_records": len(selected),
        "Text": 100.0 * sum(row["text_correct"] for row in rows) / len(rows),
        "Audio": 100.0 * sum(row["audio_correct"] for row in rows) / len(rows),
        "Group": 100.0 * sum(row["group_correct"] for row in rows) / len(rows),
        "mean_seconds_per_group": sum(row["elapsed_seconds"] for row in rows) / len(rows),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(rows) == len(selected) else 3


if __name__ == "__main__":
    raise SystemExit(main())
