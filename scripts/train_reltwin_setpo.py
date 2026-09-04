#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel

from setpo_candidates import (
    RELATION_EXCHANGE_PERMUTATION,
    candidate_quality_axes,
    candidate_qualities,
    ordinary_candidates,
    relation_candidates,
    scale_cardinality_candidates,
)
from setpo_objective import relation_objective, rehearsal_objective, token_kl_from_logits
from spotsound import AudioFlamingo3ForTemporalConditionalGeneration, AudioFlamingo3TemporalProcessor
from train_reltwin_micro import answer_for, move_example, prepare_example, sequence_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rehearsal-manifest", type=Path, required=True)
    parser.add_argument("--extra-rehearsal-manifest", type=Path, action="append", default=[])
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--prediction-temperature", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=0.15)
    parser.add_argument("--method-weight", type=float, default=1.0)
    parser.add_argument("--exchange-weight", type=float, default=1.0)
    parser.add_argument("--reference-kl-weight", type=float, default=0.0)
    parser.add_argument("--reference-temperature", type=float, default=1.0)
    parser.add_argument("--token-kl-weight", type=float, default=0.0)
    parser.add_argument("--token-temperature", type=float, default=1.0)
    parser.add_argument("--rehearsal-weight", type=float, default=0.5)
    parser.add_argument("--rehearsal-setpo-every", type=int, default=4)
    parser.add_argument("--jitter-ratio", type=float, default=0.15)
    parser.add_argument("--candidate-mode", choices=("legacy", "scale-cardinality"), default="legacy")
    parser.add_argument("--quality-mode", choices=("scalar", "pareto"), default="scalar")
    parser.add_argument("--pareto-weight", type=float, default=0.5)
    parser.add_argument("--rehearsal-sampling", choices=("uniform", "balanced"), default="uniform")
    parser.add_argument("--rehearsal-reference-kl-weight", type=float, default=0.0)
    parser.add_argument("--long-scale-threshold", type=float, default=0.3)
    parser.add_argument("--cache-groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def candidate_answer(intervals):
    return answer_for(intervals) if intervals else "No matching interval."


def _load_rehearsal_rows(path: Path, audio_root: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["_audio_root"] = str(audio_root)
    return rows


def rehearsal_stratum(row: dict) -> tuple[int, int, int]:
    annotations = row["annotations"]
    annotated_end = max((float(end) for _, end in annotations), default=1.0)
    duration = max(float(row.get("duration", annotated_end)), annotated_end, 1e-3)
    coverage = sum(float(end) - float(start) for start, end in annotations) / duration
    scale_bucket = 0 if coverage <= 0.1 else 1 if coverage <= 0.3 else 2
    cardinality_bucket = min(len(annotations), 4)
    mean_start = sum(float(start) for start, _ in annotations) / max(len(annotations), 1)
    start_bucket = min(3, int(4 * mean_start / duration))
    return scale_bucket, cardinality_bucket, start_bucket


def make_rehearsal_schedule(rows: list[dict], steps: int, rng, sampling: str) -> list[int]:
    if sampling == "uniform":
        order = rng.permutation(len(rows)).tolist()
        schedule = []
        for step in range(steps):
            position = step % len(rows)
            if position == 0 and step > 0:
                order = rng.permutation(len(rows)).tolist()
            schedule.append(order[position])
        return schedule

    strata = defaultdict(list)
    for index, row in enumerate(rows):
        strata[rehearsal_stratum(row)].append(index)
    keys = sorted(strata)
    orders = {key: rng.permutation(strata[key]).tolist() for key in keys}
    positions = {key: 0 for key in keys}
    key_order = rng.permutation(len(keys)).tolist()
    schedule = []
    for step in range(steps):
        key_position = step % len(keys)
        if key_position == 0 and step > 0:
            key_order = rng.permutation(len(keys)).tolist()
        key = keys[key_order[key_position]]
        position = positions[key]
        if position >= len(orders[key]):
            orders[key] = rng.permutation(strata[key]).tolist()
            position = 0
        schedule.append(orders[key][position])
        positions[key] = position + 1
    return schedule


def backward_outer(model, examples, objective_builder, scale=1.0):
    model.eval()
    with torch.no_grad():
        values = [float(sequence_score(model, example).detach()) for example in examples]
    model.train()
    score_variables = torch.tensor(values, device="cuda", dtype=torch.float32, requires_grad=True)
    outer, diagnostics = objective_builder(score_variables)
    scaled_outer = outer * scale
    coefficients = torch.autograd.grad(scaled_outer, score_variables)[0].detach()
    for example, coefficient in zip(examples, coefficients):
        if float(coefficient) == 0.0:
            continue
        score = sequence_score(model, example)
        score.backward(gradient=coefficient.to(score.dtype))
    return float(scaled_outer.detach()), diagnostics


def answer_token_logits(model, example):
    batch = move_example(example, model.dtype)
    output = model(**batch, use_cache=False)
    shifted_labels = batch["labels"][:, 1:]
    answer_mask = shifted_labels != -100
    logits = output.logits[:, :-1, :][answer_mask]
    if logits.numel() == 0:
        raise RuntimeError("Answer sequence has no supervised tokens")
    return logits


def backward_token_kl(model, example, teacher_logits, temperature, scale):
    was_training = model.training
    model.eval()
    try:
        student_logits = answer_token_logits(model, example)
        if student_logits.shape != teacher_logits.shape:
            raise RuntimeError(
                f"Teacher/student token-logit shape mismatch: "
                f"{tuple(teacher_logits.shape)} vs {tuple(student_logits.shape)}"
            )
        loss = token_kl_from_logits(
            student_logits, teacher_logits.to(student_logits.device), temperature
        )
        (loss * scale).backward()
        return float(loss.detach())
    finally:
        model.train(was_training)


def main():
    args = parse_args()
    if args.rehearsal_setpo_every < 1:
        raise ValueError("--rehearsal-setpo-every must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rehearsal = _load_rehearsal_rows(args.rehearsal_manifest, args.audio_dir)
    for extra_manifest in args.extra_rehearsal_manifest:
        rehearsal.extend(_load_rehearsal_rows(extra_manifest, extra_manifest.parent))
    grouped = defaultdict(dict)
    for row in manifest:
        row["_audio_root"] = str(args.audio_dir)
        grouped[(row["pair_id"], row["template"], row.get("variant", 0))][row["relation"]] = row
    groups = [value for _, value in sorted(grouped.items())]
    if not groups or any(set(group) != {"AB", "BA"} for group in groups):
        raise ValueError("Manifest must contain AB/BA rows for every pair/template")
    if not rehearsal:
        raise ValueError("SetPO requires ordinary localization rehearsal data")

    processor = AudioFlamingo3TemporalProcessor.from_pretrained(args.base)
    relation_cache = {}
    rehearsal_cache = {}

    def cache_put(cache, key, value):
        if args.cache_groups > 0:
            if len(cache) >= args.cache_groups:
                cache.pop(next(iter(cache)))
            cache[key] = value

    def load_wave(row):
        path = Path(row["audio_path"])
        if not path.is_absolute():
            root = Path(row.get("_audio_root", args.audio_dir))
            path = root / path
            if not path.is_file():
                path = root / Path(row["audio_path"]).name
        wave, _ = librosa.load(path, sr=16000, mono=True)
        return np.asarray(wave, dtype=np.float32)

    def relation_bundle(index):
        if index in relation_cache:
            return relation_cache[index]
        group = groups[index]
        ab, ba = group["AB"], group["BA"]
        wave = load_wave(ab)
        duration = len(wave) / 16000.0
        candidates = relation_candidates(
            ab["annotations"], ba["annotations"], duration, args.jitter_ratio
        )
        answers = [candidate_answer(candidate) for candidate in candidates]
        examples = [prepare_example(processor, wave, ab["caption"], answer) for answer in answers]
        examples.extend(prepare_example(processor, wave, ba["caption"], answer) for answer in answers)
        quality_builder = candidate_quality_axes if args.quality_mode == "pareto" else candidate_qualities
        bundle = {
            "examples": examples,
            "qualities_ab": quality_builder(ab["annotations"], candidates, duration)
            if args.quality_mode == "pareto"
            else quality_builder(ab["annotations"], candidates),
            "qualities_ba": quality_builder(ba["annotations"], candidates, duration)
            if args.quality_mode == "pareto"
            else quality_builder(ba["annotations"], candidates),
        }
        cache_put(relation_cache, index, bundle)
        return bundle

    def rehearsal_bundle(index):
        if index in rehearsal_cache:
            return rehearsal_cache[index]
        row = rehearsal[index]
        wave = load_wave(row)
        duration = len(wave) / 16000.0
        candidate_builder = (
            scale_cardinality_candidates
            if args.candidate_mode == "scale-cardinality"
            else ordinary_candidates
        )
        candidates = candidate_builder(row["annotations"], duration, args.jitter_ratio)
        qualities = (
            candidate_quality_axes(row["annotations"], candidates, duration)
            if args.quality_mode == "pareto"
            else candidate_qualities(row["annotations"], candidates)
        )
        bundle = {
            "examples": [
                prepare_example(processor, wave, row["caption"], candidate_answer(candidate))
                for candidate in candidates
            ],
            "qualities": qualities,
            "duration": duration,
        }
        cache_put(rehearsal_cache, index, bundle)
        return bundle

    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter, torch_device="cpu", is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = model.train().to("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)
    relation_order = rng.permutation(len(groups)).tolist()
    relation_schedule = []
    for step in range(args.steps):
        relation_position = step % len(groups)
        if relation_position == 0 and step > 0:
            relation_order = rng.permutation(len(groups)).tolist()
        relation_schedule.append(relation_order[relation_position])
    rehearsal_schedule = make_rehearsal_schedule(
        rehearsal, args.steps, rng, args.rehearsal_sampling
    )

    reference_distributions = {}
    token_references = {}
    rehearsal_references = {}
    if (
        args.reference_kl_weight > 0
        or args.token_kl_weight > 0
        or args.rehearsal_reference_kl_weight > 0
    ):
        model.eval()
        if args.reference_kl_weight > 0 or args.token_kl_weight > 0:
            for completed, index in enumerate(sorted(set(relation_schedule)), start=1):
                relation = relation_bundle(index)
                with torch.no_grad():
                    candidate_count = len(RELATION_EXCHANGE_PERMUTATION)
                    if args.reference_kl_weight > 0:
                        scores = torch.stack(
                            [
                                sequence_score(model, example).float()
                                for example in relation["examples"]
                            ]
                        )
                        reference_distributions[index] = (
                            F.softmax(
                                scores[:candidate_count] / args.reference_temperature, dim=0
                            ).cpu(),
                            F.softmax(
                                scores[candidate_count:] / args.reference_temperature, dim=0
                            ).cpu(),
                        )
                    if args.token_kl_weight > 0:
                        token_references[index] = (
                            answer_token_logits(model, relation["examples"][0])
                            .to(torch.bfloat16)
                            .cpu(),
                            answer_token_logits(
                                model, relation["examples"][candidate_count + 1]
                            )
                            .to(torch.bfloat16)
                            .cpu(),
                        )
                if completed == 1 or completed % 8 == 0:
                    print(
                        json.dumps(
                            {
                                "reference_precompute": completed,
                                "reference_total": len(set(relation_schedule)),
                            }
                        ),
                        flush=True,
                    )
        if args.rehearsal_reference_kl_weight > 0:
            setpo_indices = {
                rehearsal_schedule[step]
                for step in range(args.steps)
                if step % args.rehearsal_setpo_every == 0
            }
            long_single_indices = []
            for index in sorted(setpo_indices):
                row = rehearsal[index]
                bundle = rehearsal_bundle(index)
                coverage = sum(end - start for start, end in row["annotations"])
                if len(row["annotations"]) == 1 and coverage / bundle["duration"] >= args.long_scale_threshold:
                    long_single_indices.append(index)
            for completed, index in enumerate(long_single_indices, start=1):
                replay = rehearsal_bundle(index)
                with torch.no_grad():
                    scores = torch.stack(
                        [sequence_score(model, example).float() for example in replay["examples"]]
                    )
                rehearsal_references[index] = F.softmax(
                    scores / args.reference_temperature, dim=0
                ).cpu()
                if completed == 1 or completed % 8 == 0:
                    print(
                        json.dumps(
                            {
                                "rehearsal_reference_precompute": completed,
                                "rehearsal_reference_total": len(long_single_indices),
                            }
                        ),
                        flush=True,
                    )
        model.train()
    history = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for step in range(args.steps):
        relation_index = relation_schedule[step]
        rehearsal_index = rehearsal_schedule[step]
        relation = relation_bundle(relation_index)
        optimizer.zero_grad(set_to_none=True)
        relation_quality_ab = torch.tensor(relation["qualities_ab"], device="cuda")
        relation_quality_ba = torch.tensor(relation["qualities_ba"], device="cuda")
        reference_ab, reference_ba = reference_distributions.get(
            relation_index, (None, None)
        )
        relation_value, diagnostics = backward_outer(
            model,
            relation["examples"],
            lambda scores: relation_objective(
                scores,
                relation_quality_ab,
                relation_quality_ba,
                args.prediction_temperature,
                args.target_temperature,
                args.method_weight,
                args.exchange_weight,
                reference_ab,
                reference_ba,
                args.reference_kl_weight,
                args.quality_mode,
                args.pareto_weight,
            ),
        )
        total_value = relation_value

        if args.token_kl_weight > 0:
            candidate_count = len(RELATION_EXCHANGE_PERMUTATION)
            teacher_ab, teacher_ba = token_references[relation_index]
            token_kl_ab = backward_token_kl(
                model,
                relation["examples"][0],
                teacher_ab,
                args.token_temperature,
                args.token_kl_weight / 2,
            )
            token_kl_ba = backward_token_kl(
                model,
                relation["examples"][candidate_count + 1],
                teacher_ba,
                args.token_temperature,
                args.token_kl_weight / 2,
            )
            diagnostics["token_kl"] = (token_kl_ab + token_kl_ba) / 2
            total_value += args.token_kl_weight * diagnostics["token_kl"]

        replay = rehearsal_bundle(rehearsal_index)
        if step % args.rehearsal_setpo_every == 0:
            replay_qualities = torch.tensor(replay["qualities"], device="cuda")
            replay_value, replay_diagnostics = backward_outer(
                model,
                replay["examples"],
                lambda scores: rehearsal_objective(
                    scores,
                    replay_qualities,
                    args.prediction_temperature,
                    args.target_temperature,
                    args.method_weight,
                    rehearsal_references.get(rehearsal_index),
                    args.rehearsal_reference_kl_weight,
                    args.quality_mode,
                    args.pareto_weight,
                ),
                scale=args.rehearsal_weight,
            )
            diagnostics.update(replay_diagnostics)
        else:
            replay_score = sequence_score(model, replay["examples"][0])
            replay_loss = -replay_score * args.rehearsal_weight
            replay_loss.backward()
            replay_value = float(replay_loss.detach())
            diagnostics["rehearsal_sft"] = float((-replay_score).detach())
        total_value += replay_value

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach().cpu())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step + 1}")
        optimizer.step()
        event = {"step": step + 1, "loss": total_value, "grad_norm": grad_norm, **diagnostics}
        history.append(event)
        if step == 0 or (step + 1) % 4 == 0:
            print(json.dumps(event), flush=True)

    args.output_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_adapter, safe_serialization=True)
    elapsed = time.perf_counter() - started
    summary = {
        "mode": "setpo",
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "prediction_temperature": args.prediction_temperature,
        "target_temperature": args.target_temperature,
        "method_weight": args.method_weight,
        "exchange_weight": args.exchange_weight,
        "reference_kl_weight": args.reference_kl_weight,
        "reference_temperature": args.reference_temperature,
        "token_kl_weight": args.token_kl_weight,
        "token_temperature": args.token_temperature,
        "rehearsal_weight": args.rehearsal_weight,
        "rehearsal_setpo_every": args.rehearsal_setpo_every,
        "rehearsal_sampling": args.rehearsal_sampling,
        "rehearsal_reference_kl_weight": args.rehearsal_reference_kl_weight,
        "long_scale_threshold": args.long_scale_threshold,
        "jitter_ratio": args.jitter_ratio,
        "candidate_mode": args.candidate_mode,
        "quality_mode": args.quality_mode,
        "pareto_weight": args.pareto_weight,
        "seed": args.seed,
        "relation_groups": len(groups),
        "rehearsal_examples": len(rehearsal),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "final_event": history[-1],
        "history": history,
        "output_adapter": str(args.output_adapter.resolve()),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
