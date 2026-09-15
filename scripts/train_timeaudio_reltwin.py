#!/usr/bin/env python3
"""Matched continuation of the public TimeAudio LoRA with SFT or RelTwin."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from timeaudio_reltwin import (
    build_timeaudio_sample,
    freeze_to_existing_lora,
    load_timeaudio_model,
    move_sample,
    prepare_audio_features,
    rbee_objective,
    resolve_audio_path,
    save_trainable_delta,
    set_lora_training_mode,
    sha256,
    timeaudio_answer,
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--rehearsal-manifest", type=Path)
    parser.add_argument("--rehearsal-audio-dir", type=Path)
    parser.add_argument("--output-delta", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--mode", choices=("sft", "reltwin"), required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--method-weight", type=float, default=1.0)
    parser.add_argument("--exchange-weight", type=float, default=1.0)
    parser.add_argument("--rehearsal-weight", type=float, default=0.5)
    parser.add_argument("--cache-groups", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def sequence_score(model, sample):
    output = model(move_sample(sample))
    loss = output["loss"]
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite TimeAudio sequence loss: {loss}")
    return -loss


def main() -> int:
    args = arguments()
    if args.output_delta.exists() or args.summary.exists():
        raise FileExistsError("Refusing to overwrite an existing TimeAudio run")
    if args.steps <= 0 or args.learning_rate <= 0:
        raise ValueError("Steps and learning rate must be positive")
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive")
    if min(args.method_weight, args.exchange_weight, args.rehearsal_weight) < 0:
        raise ValueError("Objective weights and temperature must be nonnegative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in manifest:
        key = (row["pair_id"], row["template"], row.get("variant", 0))
        grouped[key][row["relation"]] = row
    groups = [value for _, value in sorted(grouped.items())]
    if not groups or any(set(group) != {"AB", "BA"} for group in groups):
        raise ValueError("Every RelTwin group must contain exactly AB and BA")
    rehearsal = (
        json.loads(args.rehearsal_manifest.read_text(encoding="utf-8"))
        if args.rehearsal_manifest
        else []
    )
    if rehearsal and args.rehearsal_audio_dir is None:
        raise ValueError("Rehearsal audio directory is required")
    missing = []
    for group in groups:
        if Path(group["AB"]["audio_path"]).name != Path(group["BA"]["audio_path"]).name:
            raise ValueError("AB and BA rows must share the same audio")
        path = resolve_audio_path(args.audio_dir, group["AB"])
        if not path.is_file():
            missing.append(str(path))
    for row in rehearsal:
        path = resolve_audio_path(args.rehearsal_audio_dir, row)
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} training audio files; first five: {missing[:5]}")

    from transformers import WhisperFeatureExtractor

    model, _, load_report = load_timeaudio_model(
        args.timeaudio_repo,
        args.config,
        args.llama,
        args.whisper,
        args.beats,
        args.bert,
        args.checkpoint,
    )
    trainable = freeze_to_existing_lora(model)
    model = model.to("cuda")
    set_lora_training_mode(model)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)

    relation_cache, rehearsal_cache = {}, {}

    def relation_examples(index):
        if index in relation_cache:
            return relation_cache[index]
        group = groups[index]
        ab, ba = group["AB"], group["BA"]
        audio_path = resolve_audio_path(args.audio_dir, ab)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        features = prepare_audio_features(audio_path, feature_extractor)
        y_ab, y_ba = timeaudio_answer(ab["annotations"]), timeaudio_answer(ba["annotations"])
        examples = [
            build_timeaudio_sample(features, ab["caption"], y_ab),
            build_timeaudio_sample(features, ab["caption"], y_ba),
            build_timeaudio_sample(features, ba["caption"], y_ab),
            build_timeaudio_sample(features, ba["caption"], y_ba),
        ]
        if args.cache_groups > 0:
            if len(relation_cache) >= args.cache_groups:
                relation_cache.pop(next(iter(relation_cache)))
            relation_cache[index] = examples
        return examples

    def rehearsal_example(index):
        if index in rehearsal_cache:
            return rehearsal_cache[index]
        row = rehearsal[index]
        audio_path = resolve_audio_path(args.rehearsal_audio_dir, row)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        features = prepare_audio_features(audio_path, feature_extractor)
        example = build_timeaudio_sample(
            features, row["caption"], timeaudio_answer(row["annotations"])
        )
        if args.cache_groups > 0:
            if len(rehearsal_cache) >= args.cache_groups:
                rehearsal_cache.pop(next(iter(rehearsal_cache)))
            rehearsal_cache[index] = example
        return example

    rng = np.random.default_rng(args.seed)
    relation_order = rng.permutation(len(groups)).tolist()
    rehearsal_order = rng.permutation(len(rehearsal)).tolist() if rehearsal else []
    history = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.steps):
        if step and step % len(groups) == 0:
            relation_order = rng.permutation(len(groups)).tolist()
        examples = relation_examples(relation_order[step % len(groups)])
        optimizer.zero_grad(set_to_none=True)
        if args.mode == "sft":
            losses = []
            for index in (0, 3):
                loss = -sequence_score(model, examples[index]) / 2
                loss.backward()
                losses.append(float((loss * 2).detach()))
            diagnostics = {"sft": sum(losses) / 2}
            total_value = diagnostics["sft"]
        else:
            model.eval()
            rng_states = []
            with torch.no_grad():
                values = []
                set_lora_training_mode(model)
                for example in examples:
                    rng_states.append(torch.cuda.get_rng_state())
                    values.append(float(sequence_score(model, example).detach()))
                final_rng_state = torch.cuda.get_rng_state()
            variables = torch.tensor(values, device="cuda", dtype=torch.float32, requires_grad=True)
            outer, diagnostics = rbee_objective(
                variables,
                args.temperature,
                args.method_weight,
                args.exchange_weight,
            )
            coefficients = torch.autograd.grad(outer, variables)[0].detach()
            total_value = float(outer.detach())
            for example, coefficient, rng_state in zip(examples, coefficients, rng_states):
                torch.cuda.set_rng_state(rng_state)
                sequence_score(model, example).backward(gradient=coefficient)
            torch.cuda.set_rng_state(final_rng_state)
        if rehearsal:
            if step and step % len(rehearsal) == 0:
                rehearsal_order = rng.permutation(len(rehearsal)).tolist()
            replay = -sequence_score(
                model, rehearsal_example(rehearsal_order[step % len(rehearsal)])
            )
            (args.rehearsal_weight * replay).backward()
            diagnostics["rehearsal"] = float(replay.detach())
            total_value += args.rehearsal_weight * float(replay.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step + 1}")
        optimizer.step()
        event = {"step": step + 1, "loss": total_value, "grad_norm": grad_norm, **diagnostics}
        history.append(event)
        if step == 0 or (step + 1) % 4 == 0:
            print(json.dumps(event), flush=True)

    metadata = {
        "mode": args.mode,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "method_weight": args.method_weight,
        "exchange_weight": args.exchange_weight,
        "rehearsal_weight": args.rehearsal_weight,
        "seed": args.seed,
        "manifest_sha256": sha256(args.manifest),
        "official_checkpoint_sha256": sha256(args.checkpoint),
    }
    save_trainable_delta(model, args.output_delta, metadata)
    summary = {
        **metadata,
        "load_report": load_report,
        "relation_groups": len(groups),
        "rehearsal_examples": len(rehearsal),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "final_event": history[-1],
        "history": history,
        "output_delta": str(args.output_delta.resolve()),
        "delta_sha256": sha256(args.output_delta),
    }
    write_json(args.summary, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
