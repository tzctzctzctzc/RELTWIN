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
    candidate_qualities,
    ordinary_candidates,
    relation_candidates,
)
from spotsound import AudioFlamingo3ForTemporalConditionalGeneration, AudioFlamingo3TemporalProcessor
from train_reltwin_micro import answer_for, move_example, prepare_example, sequence_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rehearsal-manifest", type=Path, required=True)
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
    parser.add_argument("--cache-groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def candidate_answer(intervals):
    return answer_for(intervals) if intervals else "No matching interval."


def listwise_cross_entropy(scores, qualities, prediction_temperature, target_temperature):
    target = F.softmax(qualities / target_temperature, dim=0)
    log_prediction = F.log_softmax(scores / prediction_temperature, dim=0)
    loss = -(target * log_prediction).sum()
    return loss, target


def relation_objective(
    scores,
    qualities_ab,
    qualities_ba,
    prediction_temperature,
    target_temperature,
    method_weight,
    exchange_weight,
    reference_ab=None,
    reference_ba=None,
    reference_kl_weight=0.0,
):
    candidate_count = len(RELATION_EXCHANGE_PERMUTATION)
    scores_ab = scores[:candidate_count]
    scores_ba = scores[candidate_count:]
    sft = -(scores_ab[0] + scores_ba[1]) / 2
    listwise_ab, target_ab = listwise_cross_entropy(
        scores_ab, qualities_ab, prediction_temperature, target_temperature
    )
    listwise_ba, target_ba = listwise_cross_entropy(
        scores_ba, qualities_ba, prediction_temperature, target_temperature
    )
    listwise = (listwise_ab + listwise_ba) / 2
    prediction_ab = F.softmax(scores_ab / prediction_temperature, dim=0)
    prediction_ba = F.softmax(scores_ba / prediction_temperature, dim=0)
    permutation = torch.tensor(RELATION_EXCHANGE_PERMUTATION, device=scores.device)
    exchanged_ba = prediction_ba[permutation]
    midpoint = (prediction_ab + exchanged_ba) / 2
    exchange_js = (
        F.kl_div(midpoint.clamp_min(1e-8).log(), prediction_ab, reduction="sum")
        + F.kl_div(midpoint.clamp_min(1e-8).log(), exchanged_ba, reduction="sum")
    ) / 2
    reference_kl = scores.new_zeros(())
    if reference_ab is not None and reference_ba is not None:
        reference_ab = reference_ab.to(device=scores.device, dtype=prediction_ab.dtype)
        reference_ba = reference_ba.to(device=scores.device, dtype=prediction_ba.dtype)
        reference_kl_ab = (
            reference_ab
            * (reference_ab.clamp_min(1e-8).log() - prediction_ab.clamp_min(1e-8).log())
        ).sum()
        reference_kl_ba = (
            reference_ba
            * (reference_ba.clamp_min(1e-8).log() - prediction_ba.clamp_min(1e-8).log())
        ).sum()
        reference_kl = (reference_kl_ab + reference_kl_ba) / 2
    total = (
        sft
        + method_weight * (listwise + exchange_weight * exchange_js)
        + reference_kl_weight * reference_kl
    )
    diagnostics = {
        "sft": float(sft.detach()),
        "relation_listwise": float(listwise.detach()),
        "exchange_js": float(exchange_js.detach()),
        "reference_kl": float(reference_kl.detach()),
        "p_ab_exact": float(prediction_ab[0].detach()),
        "p_ba_exact": float(prediction_ba[1].detach()),
        "target_ab_exact": float(target_ab[0].detach()),
        "target_ba_exact": float(target_ba[1].detach()),
    }
    return total, diagnostics


def rehearsal_objective(
    scores,
    qualities,
    prediction_temperature,
    target_temperature,
    method_weight,
):
    sft = -scores[0]
    listwise, target = listwise_cross_entropy(
        scores, qualities, prediction_temperature, target_temperature
    )
    total = sft + method_weight * listwise
    prediction = F.softmax(scores / prediction_temperature, dim=0)
    return total, {
        "rehearsal_sft": float(sft.detach()),
        "rehearsal_listwise": float(listwise.detach()),
        "rehearsal_p_exact": float(prediction[0].detach()),
        "rehearsal_target_exact": float(target[0].detach()),
    }


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


def token_kl_from_logits(student_logits, teacher_logits, temperature):
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    return (
        F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        * temperature**2
    )


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
    rehearsal = json.loads(args.rehearsal_manifest.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in manifest:
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
        path = args.audio_dir / Path(row["audio_path"]).name
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
        bundle = {
            "examples": examples,
            "qualities_ab": candidate_qualities(ab["annotations"], candidates),
            "qualities_ba": candidate_qualities(ba["annotations"], candidates),
        }
        cache_put(relation_cache, index, bundle)
        return bundle

    def rehearsal_bundle(index):
        if index in rehearsal_cache:
            return rehearsal_cache[index]
        row = rehearsal[index]
        wave = load_wave(row)
        duration = len(wave) / 16000.0
        candidates = ordinary_candidates(row["annotations"], duration, args.jitter_ratio)
        bundle = {
            "examples": [
                prepare_example(processor, wave, row["caption"], candidate_answer(candidate))
                for candidate in candidates
            ],
            "qualities": candidate_qualities(row["annotations"], candidates),
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
    rehearsal_order = rng.permutation(len(rehearsal)).tolist()
    relation_schedule = []
    rehearsal_schedule = []
    for step in range(args.steps):
        relation_position = step % len(groups)
        rehearsal_position = step % len(rehearsal)
        if relation_position == 0 and step > 0:
            relation_order = rng.permutation(len(groups)).tolist()
        if rehearsal_position == 0 and step > 0:
            rehearsal_order = rng.permutation(len(rehearsal)).tolist()
        relation_schedule.append(relation_order[relation_position])
        rehearsal_schedule.append(rehearsal_order[rehearsal_position])

    reference_distributions = {}
    token_references = {}
    if args.reference_kl_weight > 0 or args.token_kl_weight > 0:
        model.eval()
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
        "jitter_ratio": args.jitter_ratio,
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
