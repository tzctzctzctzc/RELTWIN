#!/usr/bin/env python3
"""Evaluate one SpanTool checkpoint with the existing interval-metric harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

from interval_metrics import event_f1_iou, normalize_intervals, temporal_set_iou
from spantool import decode_spantool_output, refine_proposal_intervals
from spantool_runtime import (
    artifact_sha256,
    file_sha256,
    load_manifest,
    load_spantool_checkpoint,
    load_wave,
    move_model_inputs,
    prepare_spantool_input,
    select_audio_states,
    validate_rows,
)
from spotsound import (
    AudioFlamingo3ForTemporalConditionalGeneration,
    AudioFlamingo3SpanToolProcessor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--spantool-checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refinement-radius-seconds", type=float, default=0.8)
    parser.add_argument("--proposal-predictions", type=Path)
    parser.add_argument(
        "--decode-mode", choices=("standalone", "refine", "guided"),
        default="standalone",
    )
    parser.add_argument("--proposal-weight", type=float, default=1.0)
    return parser.parse_args()


def checkpoint_id(args: argparse.Namespace) -> str:
    digest = hashlib.sha256()
    digest.update(file_sha256(args.spantool_checkpoint / "spantool_head.pt").encode())
    digest.update(file_sha256(args.spantool_checkpoint / "spantool_config.json").encode())
    digest.update(artifact_sha256(args.adapter).encode())
    digest.update(file_sha256(args.annotations).encode())
    digest.update(str(args.base.resolve()).encode())
    digest.update(str(args.refinement_radius_seconds).encode())
    digest.update(args.decode_mode.encode())
    digest.update(str(args.proposal_weight).encode())
    if args.proposal_predictions:
        digest.update(file_sha256(args.proposal_predictions).encode())
    return digest.hexdigest()[:16]


def load_proposals(path: Path | None) -> dict[int, list[list[float]]]:
    if path is None:
        return {}
    proposals = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "ok":
                proposals[int(record["index"])] = record["prediction"]
    return proposals


def load_completed(path: Path, expected_id: str) -> dict[int, dict]:
    if not path.is_file():
        return {}
    completed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("checkpoint_id") != expected_id:
            raise ValueError(
                f"Prediction file belongs to checkpoint {record.get('checkpoint_id')}, "
                f"expected {expected_id}: {path}"
            )
        if record.get("status") == "ok":
            completed[int(record["index"])] = record
    return completed


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    rows = load_manifest(args.annotations)
    validate_rows(rows, str(args.annotations))
    stop = len(rows) if args.limit <= 0 else min(len(rows), args.start + args.limit)
    indices = list(range(args.start, stop))
    proposals = load_proposals(args.proposal_predictions)
    if args.decode_mode != "standalone":
        missing = [index for index in indices if index not in proposals]
        if missing:
            raise ValueError(
                f"{args.decode_mode} decoding needs proposals for every row; "
                f"missing {len(missing)}"
            )
    run_id = checkpoint_id(args)
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.predictions, run_id)

    device = torch.device("cuda")
    load_started = time.perf_counter()
    processor = AudioFlamingo3SpanToolProcessor.from_pretrained(args.base)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model, args.adapter, torch_device="cpu", is_trainable=False
    ).merge_and_unload()
    model = model.eval().to(device)
    head, head_payload = load_spantool_checkpoint(args.spantool_checkpoint, device)
    head = head.eval()
    audio_layers = [
        int(layer)
        for layer in head_payload.get("audio_layers", [head_payload["audio_layer"]])
    ]
    load_seconds = time.perf_counter() - load_started

    with args.predictions.open("a", encoding="utf-8") as output_handle:
        for index in indices:
            if index in completed:
                continue
            row = rows[index]
            wave = load_wave(args.audio_dir, row)
            duration = len(wave) / 16000.0
            inputs = move_model_inputs(
                prepare_spantool_input(processor, wave, row["caption"]),
                device,
                model.dtype,
            )
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                frame_states, frame_mask = select_audio_states(
                    outputs,
                    audio_layers,
                    inputs["input_ids"],
                    model.config.audio_token_id,
                )
                head_output = head(
                    frame_states.float(),
                    frame_mask,
                    torch.tensor([duration], device=device),
                )
                if args.decode_mode == "refine":
                    prediction = refine_proposal_intervals(
                        proposals[index],
                        head_output["fine_onset_logits"][0, : int(frame_mask.sum())],
                        head_output["fine_offset_logits"][0, : int(frame_mask.sum())],
                        duration,
                        radius_seconds=args.refinement_radius_seconds,
                    )
                else:
                    prediction = decode_spantool_output(
                        head,
                        head_output,
                        0,
                        duration,
                        refinement_radius_seconds=args.refinement_radius_seconds,
                        proposal_intervals=(
                            proposals[index] if args.decode_mode == "guided" else None
                        ),
                        proposal_weight=args.proposal_weight,
                    )
            elapsed = time.perf_counter() - started
            ground_truth = normalize_intervals(row["annotations"], duration)
            iou = temporal_set_iou(ground_truth, prediction)
            record = {
                "status": "ok",
                "checkpoint_id": run_id,
                "index": index,
                "audio": row["audio_path"],
                "query": row["caption"],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "iou": iou,
                "duration_seconds": duration,
                "predicted_count": len(prediction),
                "ground_truth_count": len(ground_truth),
                "inference_seconds": elapsed,
                "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                "benchmark": row.get("benchmark"),
                "benchmark_id": row.get("benchmark_id", row.get("qid")),
            }
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            completed[index] = record
            print(json.dumps(record, ensure_ascii=False), flush=True)

    records = [completed[index] for index in indices if index in completed]
    ious = np.asarray([record["iou"] for record in records], dtype=np.float64)
    event_f1 = [
        event_f1_iou(record["ground_truth"], record["prediction"], 0.5)["f1"]
        for record in records
    ]
    summary = {
        "checkpoint_id": run_id,
        "base": str(args.base.resolve()),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": artifact_sha256(args.adapter),
        "spantool_checkpoint": str(args.spantool_checkpoint.resolve()),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": file_sha256(args.annotations),
        "decode_mode": args.decode_mode,
        "proposal_predictions": (
            str(args.proposal_predictions.resolve()) if args.proposal_predictions else None
        ),
        "proposal_weight": args.proposal_weight,
        "audio_layer": audio_layers[0],
        "audio_layers": audio_layers,
        "head_config": head_payload["head_config"],
        "layer_weights": (
            torch.softmax(head.layer_logits.detach(), dim=0).cpu().tolist()
            if head.layer_logits is not None else [1.0]
        ),
        "device": torch.cuda.get_device_name(0),
        "load_seconds": load_seconds,
        "expected_count": len(indices),
        "completed_count": len(records),
        "R1@0.3": float((ious >= 0.3).mean() * 100) if len(ious) else None,
        "R1@0.5": float((ious >= 0.5).mean() * 100) if len(ious) else None,
        "R1@0.7": float((ious >= 0.7).mean() * 100) if len(ious) else None,
        "mIoU": float(ious.mean() * 100) if len(ious) else None,
        "event_F1@0.5": float(np.mean(event_f1) * 100) if event_f1 else None,
        "mean_inference_seconds": (
            float(np.mean([record["inference_seconds"] for record in records]))
            if records
            else None
        ),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(records) == len(indices) else 3


if __name__ == "__main__":
    raise SystemExit(main())
