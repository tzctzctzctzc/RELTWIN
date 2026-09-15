#!/usr/bin/env python3
"""Parameter-efficient RelTwin post-training on a released TEMPO checkpoint."""

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
from peft import LoraConfig, get_peft_model
from transformers import AudioFlamingo3Processor

from tempo_reltwin import (
    TempoAudioFlamingo3ForConditionalGeneration,
    atomic_answer,
    grounding_conversation,
    grounding_question,
    load_time_projector,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--time-projector", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--rehearsal-manifest", type=Path)
    parser.add_argument("--rehearsal-audio-dir", type=Path)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--mode", choices=("sft", "rbee"), default="rbee")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--method-weight", type=float, default=1.0)
    parser.add_argument("--exchange-weight", type=float, default=1.0)
    parser.add_argument("--rehearsal-weight", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--cache-groups", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def prepare_example(processor, wave, question, answer):
    prompt = grounding_conversation(wave, question)
    full = grounding_conversation(wave, question, answer)
    prefix = processor.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    inputs = processor.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False, return_dict=True
    )
    labels = inputs["input_ids"].clone()
    labels[:, : prefix["input_ids"].shape[1]] = -100
    inputs["labels"] = labels
    return {
        key: value.cpu() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def move(example, dtype):
    result = {}
    for key, value in example.items():
        if torch.is_tensor(value):
            value = value.to("cuda", non_blocking=True)
            if value.is_floating_point():
                value = value.to(dtype)
        result[key] = value
    return result


def score(model, example):
    output = model(**move(example, model.dtype), use_cache=False)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("Non-finite sequence loss")
    return -output.loss


def rbee_objective(scores, temperature, method_weight, exchange_weight):
    sft = -(scores[0] + scores[3]) / 2
    if method_weight == 0:
        return sft, {"sft": float(sft.detach())}
    ab = scores[:2] / temperature
    ba = scores[2:] / temperature
    candidate = (
        F.cross_entropy(ab.unsqueeze(0), torch.tensor([0], device=scores.device))
        + F.cross_entropy(ba.unsqueeze(0), torch.tensor([1], device=scores.device))
    ) / 2
    p_ab = F.softmax(ab, dim=0)
    p_ba = torch.flip(F.softmax(ba, dim=0), dims=[0])
    midpoint = (p_ab + p_ba) / 2
    js = (
        F.kl_div(midpoint.log(), p_ab, reduction="sum")
        + F.kl_div(midpoint.log(), p_ba, reduction="sum")
    ) / 2
    total = sft + method_weight * (candidate + exchange_weight * js)
    return total, {
        "sft": float(sft.detach()),
        "candidate_ce": float(candidate.detach()),
        "exchange_js": float(js.detach()),
    }


def main():
    args = arguments()
    if args.output_adapter.exists() or args.summary.exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in manifest:
        key = (row["pair_id"], row["template"], row.get("variant", 0))
        grouped[key][row["relation"]] = row
    groups = [value for _, value in sorted(grouped.items())]
    if not groups or any(set(group) != {"AB", "BA"} for group in groups):
        raise ValueError("Every RelTwin group must contain AB and BA")
    rehearsal = (
        json.loads(args.rehearsal_manifest.read_text(encoding="utf-8"))
        if args.rehearsal_manifest
        else []
    )
    if rehearsal and not args.rehearsal_audio_dir:
        raise ValueError("Rehearsal audio directory is required")

    processor = AudioFlamingo3Processor.from_pretrained(args.base)
    model = TempoAudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    load_time_projector(model, args.time_projector)
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith("language_model.model.layers.")
        and isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1]
        in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    ]
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, config)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = model.train().to("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)

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
        y_ab, y_ba = atomic_answer(ab["annotations"]), atomic_answer(ba["annotations"])
        examples = [
            prepare_example(processor, wave, grounding_question(ab["caption"]), y_ab),
            prepare_example(processor, wave, grounding_question(ab["caption"]), y_ba),
            prepare_example(processor, wave, grounding_question(ba["caption"]), y_ab),
            prepare_example(processor, wave, grounding_question(ba["caption"]), y_ba),
        ]
        if len(relation_cache) >= args.cache_groups:
            relation_cache.pop(next(iter(relation_cache)))
        relation_cache[index] = examples
        return examples

    def rehearsal_example(index):
        if index in rehearsal_cache:
            return rehearsal_cache[index]
        row = rehearsal[index]
        path = args.rehearsal_audio_dir / row["audio_path"]
        if not path.is_file():
            path = args.rehearsal_audio_dir / Path(row["audio_path"]).name
        wave, _ = librosa.load(path, sr=16000, mono=True)
        wave = np.asarray(wave, dtype=np.float32)
        question = row.get("source_question") or grounding_question(row["caption"])
        if not question.startswith("[audio:ground]"):
            question = f"[audio:ground] {question}"
        example = prepare_example(
            processor, wave, question, atomic_answer(row["annotations"])
        )
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
                value = -score(model, examples[index]) / 2
                value.backward()
                losses.append(float((value * 2).detach()))
            diagnostics = {"sft": sum(losses) / len(losses)}
            total_value = diagnostics["sft"]
        else:
            model.eval()
            with torch.no_grad():
                values = [float(score(model, example).detach()) for example in examples]
            model.train()
            variables = torch.tensor(values, device="cuda", requires_grad=True)
            outer, diagnostics = rbee_objective(
                variables,
                args.temperature,
                args.method_weight,
                args.exchange_weight,
            )
            coefficients = torch.autograd.grad(outer, variables)[0].detach()
            total_value = float(outer.detach())
            for example, coefficient in zip(examples, coefficients):
                score(model, example).backward(gradient=coefficient.to(model.dtype))
        if rehearsal:
            if step and step % len(rehearsal) == 0:
                rehearsal_order = rng.permutation(len(rehearsal)).tolist()
            replay = -score(model, rehearsal_example(rehearsal_order[step % len(rehearsal)]))
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

    args.output_adapter.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output_adapter, safe_serialization=True)
    summary = {
        "mode": args.mode,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "method_weight": args.method_weight,
        "exchange_weight": args.exchange_weight,
        "rehearsal_weight": args.rehearsal_weight,
        "seed": args.seed,
        "relation_groups": len(groups),
        "rehearsal_examples": len(rehearsal),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "history": history,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
