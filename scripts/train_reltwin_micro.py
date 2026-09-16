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

from spotsound import (
    AudioFlamingo3ForTemporalConditionalGeneration,
    AudioFlamingo3TemporalProcessor,
    GROUNDING_PROMPT,
)
from spotsound_reltwin import (
    capture_rng_state,
    restore_rng_state,
    set_spotsound_training_mode,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--rehearsal-manifest", type=Path)
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--output-adapter", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--mode", choices=("sft", "ranking", "rbee"), required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--method-weight", type=float, default=1.0)
    p.add_argument("--exchange-weight", type=float, default=1.0,
                   help="RBEE JS weight only; 0 retains the identical candidate CE control.")
    p.add_argument("--rehearsal-weight", type=float, default=0.0)
    p.add_argument("--cache-groups", type=int, default=32)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def answer_for(intervals):
    return ", ".join(f"from {float(s):.3f}s to {float(e):.3f}s" for s, e in intervals)


def prepare_example(processor, wave, query, answer):
    user_content = [
        {"type": "audio", "audio": wave},
        {"type": "text", "text": GROUNDING_PROMPT + query + " Answer: "},
    ]
    prompt = [{"role": "user", "content": user_content}]
    full = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    prompt_inputs = processor.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=True)
    inputs = processor.apply_chat_template(full, tokenize=True, add_generation_prompt=False, return_dict=True)
    labels = inputs["input_ids"].clone()
    labels[:, : int(prompt_inputs["input_ids"].shape[1])] = -100
    inputs["labels"] = labels
    return {key: value.cpu() if torch.is_tensor(value) else value for key, value in inputs.items()}


def move_example(example, dtype):
    result = {}
    for key, value in example.items():
        if not torch.is_tensor(value):
            result[key] = value
            continue
        value = value.to("cuda", non_blocking=True)
        if value.is_floating_point():
            value = value.to(dtype)
        result[key] = value
    return result


def sequence_score(model, example):
    batch = move_example(example, model.dtype)
    output = model(**batch, use_cache=False)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError(f"Non-finite sequence loss: {output.loss}")
    return -output.loss


def outer_objective(mode, scores, temperature, weight, margin, exchange_weight=1.0):
    # score order: qAB/yAB, qAB/yBA, qBA/yAB, qBA/yBA
    sft = -(scores[0] + scores[3]) / 2
    if mode == "sft":
        return sft, {"sft": float(sft.detach())}
    logits_ab = scores[:2] / temperature
    logits_ba = scores[2:] / temperature
    if mode == "ranking":
        rank = (
            F.relu(margin - scores[0] + scores[1])
            + F.relu(margin - scores[3] + scores[2])
        ) / 2
        total = sft + weight * rank
        return total, {"sft": float(sft.detach()), "ranking": float(rank.detach())}
    candidate = (
        F.cross_entropy(logits_ab.unsqueeze(0), torch.tensor([0], device=scores.device))
        + F.cross_entropy(logits_ba.unsqueeze(0), torch.tensor([1], device=scores.device))
    ) / 2
    p_ab = F.softmax(logits_ab, dim=0)
    p_ba_exchanged = torch.flip(F.softmax(logits_ba, dim=0), dims=[0])
    midpoint = (p_ab + p_ba_exchanged) / 2
    js = (
        F.kl_div(midpoint.log(), p_ab, reduction="sum")
        + F.kl_div(midpoint.log(), p_ba_exchanged, reduction="sum")
    ) / 2
    total = sft + weight * (candidate + exchange_weight * js)
    return total, {
        "sft": float(sft.detach()),
        "candidate_ce": float(candidate.detach()),
        "exchange_js": float(js.detach()),
        "p_ab_correct": float(p_ab[0].detach()),
        "p_ba_correct": float(torch.flip(p_ba_exchanged, dims=[0])[1].detach()),
    }


def main():
    args = parse_args()
    if not math.isfinite(args.exchange_weight) or args.exchange_weight < 0:
        raise ValueError("--exchange-weight must be finite and nonnegative")
    if args.output_adapter.exists() or args.summary.exists():
        raise FileExistsError("Refusing to overwrite an existing adapter or training summary")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in manifest:
        grouped[(row["pair_id"], row["template"], row.get("variant", 0))][row["relation"]] = row
    groups = [value for _, value in sorted(grouped.items())]
    if not groups or any(set(x) != {"AB", "BA"} for x in groups):
        raise ValueError("Manifest must contain AB/BA rows for every pair/template")
    rehearsal = []
    if args.rehearsal_manifest:
        rehearsal = json.loads(args.rehearsal_manifest.read_text(encoding="utf-8"))
    if args.rehearsal_weight > 0 and not rehearsal:
        raise ValueError("Positive rehearsal weight requires a non-empty rehearsal manifest")

    processor = AudioFlamingo3TemporalProcessor.from_pretrained(args.base)
    relation_cache = {}
    rehearsal_cache = {}

    def relation_examples(index):
        if index in relation_cache:
            return relation_cache[index]
        group = groups[index]
        ab, ba = group["AB"], group["BA"]
        path = args.audio_dir / Path(ab["audio_path"]).name
        wave, _ = librosa.load(path, sr=16000, mono=True)
        wave = np.asarray(wave, dtype=np.float32)
        y_ab = answer_for(ab["annotations"])
        y_ba = answer_for(ba["annotations"])
        examples = [
            prepare_example(processor, wave, ab["caption"], y_ab),
            prepare_example(processor, wave, ab["caption"], y_ba),
            prepare_example(processor, wave, ba["caption"], y_ab),
            prepare_example(processor, wave, ba["caption"], y_ba),
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
        path = args.audio_dir / Path(row["audio_path"]).name
        wave, _ = librosa.load(path, sr=16000, mono=True)
        example = prepare_example(processor, wave, row["caption"], answer_for(row["annotations"]))
        if args.cache_groups > 0:
            if len(rehearsal_cache) >= args.cache_groups:
                rehearsal_cache.pop(next(iter(rehearsal_cache)))
            rehearsal_cache[index] = example
        return example

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
    model = model.to("cuda")
    set_spotsound_training_mode(model, scope="lora")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    history = []
    rng = np.random.default_rng(args.seed)
    relation_order = rng.permutation(len(groups)).tolist()
    rehearsal_order = rng.permutation(len(rehearsal)).tolist() if rehearsal else []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    for step in range(args.steps):
        relation_position = step % len(groups)
        if relation_position == 0 and step > 0:
            relation_order = rng.permutation(len(groups)).tolist()
        examples = relation_examples(relation_order[relation_position])
        optimizer.zero_grad(set_to_none=True)
        if args.mode == "sft":
            losses = []
            for index in (0, 3):
                score = sequence_score(model, examples[index])
                loss = -score / 2
                loss.backward()
                losses.append(float((-score).detach()))
            diagnostics = {"sft": sum(losses) / len(losses)}
            total_value = diagnostics["sft"]
        else:
            set_spotsound_training_mode(model, scope="lora")
            rng_states = []
            with torch.no_grad():
                values = []
                for example in examples:
                    rng_states.append(capture_rng_state())
                    values.append(float(sequence_score(model, example).detach()))
                final_rng_state = capture_rng_state()
            score_variables = torch.tensor(values, device="cuda", dtype=torch.float32, requires_grad=True)
            outer, diagnostics = outer_objective(
                args.mode,
                score_variables,
                args.temperature,
                args.method_weight,
                args.margin,
                args.exchange_weight,
            )
            coefficients = torch.autograd.grad(outer, score_variables)[0].detach()
            total_value = float(outer.detach())
            for index, (coefficient, rng_state) in enumerate(zip(coefficients, rng_states)):
                restore_rng_state(rng_state)
                score = sequence_score(model, examples[index])
                score.backward(gradient=coefficient.to(score.dtype))
            restore_rng_state(final_rng_state)
        if rehearsal:
            rehearsal_position = step % len(rehearsal)
            if rehearsal_position == 0 and step > 0:
                rehearsal_order = rng.permutation(len(rehearsal)).tolist()
            replay_score = sequence_score(model, rehearsal_example(rehearsal_order[rehearsal_position]))
            replay_loss = -replay_score * args.rehearsal_weight
            replay_loss.backward()
            diagnostics["rehearsal"] = float((-replay_score).detach())
            total_value += float(replay_loss.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach().cpu())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        event = {"step": step + 1, "loss": total_value, "grad_norm": grad_norm, **diagnostics}
        history.append(event)
        if step == 0 or (step + 1) % 4 == 0:
            print(json.dumps(event), flush=True)

    args.output_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_adapter, safe_serialization=True)
    elapsed = time.perf_counter() - started
    summary = {
        "mode": args.mode,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "method_weight": args.method_weight,
        "exchange_weight": args.exchange_weight,
        "rehearsal_weight": args.rehearsal_weight,
        "margin": args.margin,
        "seed": args.seed,
        "relation_pairs": len(groups),
        "rehearsal_examples": len(rehearsal),
        "trainable_parameters": sum(p.numel() for p in trainable),
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
