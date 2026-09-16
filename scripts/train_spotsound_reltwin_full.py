#!/usr/bin/env python3
"""Two-stage full RelTwin adaptation for the official SpotSound-A checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel

from setpo_candidates import (
    candidate_quality_axes,
    relation_candidates,
    scale_cardinality_candidates,
)
from setpo_objective import relation_objective, rehearsal_objective
from spotsound import (
    AudioFlamingo3ForTemporalConditionalGeneration,
    AudioFlamingo3TemporalProcessor,
)
from spotsound_reltwin import (
    capture_rng_state,
    configure_spotsound_parameters,
    group_twin_rows,
    parameter_component,
    restore_rng_state,
    save_bridge_delta,
    set_spotsound_training_mode,
    sha256,
    weighted_schedule,
    write_json,
)
from train_reltwin_micro import answer_for, outer_objective, prepare_example, sequence_score


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--twin-source",
        nargs=4,
        action="append",
        required=True,
        metavar=("NAME", "MANIFEST", "AUDIO_ROOT", "WEIGHT"),
    )
    parser.add_argument(
        "--replay-source",
        nargs=4,
        action="append",
        required=True,
        metavar=("NAME", "MANIFEST", "AUDIO_ROOT", "WEIGHT"),
    )
    parser.add_argument("--eval-manifest", type=Path, action="append", default=[])
    parser.add_argument("--stage1-adapter", type=Path, required=True)
    parser.add_argument("--stage1-delta", type=Path, required=True)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--output-delta", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rbee-steps", type=int, default=384)
    parser.add_argument("--setpo-steps", type=int, default=128)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-6)
    parser.add_argument("--bridge-learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--prediction-temperature", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=0.15)
    parser.add_argument("--method-weight", type=float, default=1.0)
    parser.add_argument("--exchange-weight", type=float, default=1.0)
    parser.add_argument("--rbee-replay-weight", type=float, default=0.75)
    parser.add_argument("--setpo-replay-weight", type=float, default=0.75)
    parser.add_argument("--relation-reference-kl-weight", type=float, default=0.2)
    parser.add_argument("--replay-reference-kl-weight", type=float, default=0.2)
    parser.add_argument("--reference-temperature", type=float, default=1.0)
    parser.add_argument("--jitter-ratio", type=float, default=0.15)
    parser.add_argument("--pareto-weight", type=float, default=0.5)
    parser.add_argument("--cache-audios", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    return rows


def resolve_audio_path(audio_root: Path, row: dict) -> Path:
    raw = Path(row["audio_path"])
    candidates = [raw, audio_root / raw, audio_root / raw.name]
    for candidate in list(candidates):
        candidates.extend(candidate.with_suffix(ext) for ext in (".wav", ".flac", ".mp3"))
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def parse_sources(raw_sources, kind: str) -> list[dict]:
    sources = []
    names = set()
    for name, manifest, audio_root, weight in raw_sources:
        if name in names:
            raise ValueError(f"Duplicate {kind} source name: {name}")
        names.add(name)
        numeric_weight = float(weight)
        if numeric_weight <= 0:
            raise ValueError(f"{kind} source weight must be positive: {name}")
        manifest = Path(manifest).resolve()
        sources.append(
            {
                "name": name,
                "manifest": manifest,
                "audio_root": Path(audio_root).resolve(),
                "weight": numeric_weight,
                "rows": read_json(manifest),
            }
        )
    return sources


def prepare_source_pools(twin_sources: list[dict], replay_sources: list[dict]):
    for source in twin_sources:
        source["items"], source["schema"] = group_twin_rows(source.pop("rows"))
    for source in replay_sources:
        rows = source.pop("rows")
        for row in rows:
            if not all(key in row for key in ("audio_path", "caption", "annotations")):
                raise ValueError(f"Malformed replay row in {source['name']}")
        source["items"] = rows
        source["schema"] = "ordinary_localization"
    return twin_sources, replay_sources


def identity(rows: list[dict]):
    audio = {Path(row["audio_path"]).stem for row in rows}
    queries = {
        (Path(row["audio_path"]).stem, " ".join(str(row["caption"]).lower().split()))
        for row in rows
    }
    return audio, queries


def audit_no_evaluation_overlap(twin_sources, replay_sources, eval_manifests):
    training_rows = []
    for source in twin_sources:
        for group in source["items"]:
            training_rows.extend((group["AB"], group["BA"]))
    for source in replay_sources:
        training_rows.extend(source["items"])
    train_audio, train_queries = identity(training_rows)
    reports = []
    for manifest in eval_manifests:
        rows = read_json(manifest)
        eval_audio, eval_queries = identity(rows)
        audio_overlap = sorted(train_audio & eval_audio)
        query_overlap = sorted(train_queries & eval_queries)
        report = {
            "manifest": str(manifest.resolve()),
            "sha256": sha256(manifest),
            "rows": len(rows),
            "audio_overlap": len(audio_overlap),
            "query_overlap": len(query_overlap),
            "first_audio_overlap": audio_overlap[:5],
        }
        reports.append(report)
        if audio_overlap or query_overlap:
            raise RuntimeError(f"Training/evaluation overlap detected: {report}")
    return {
        "training_rows": len(training_rows),
        "training_audio_ids": len(train_audio),
        "evaluation_manifests": reports,
        "status": "pass",
    }


def validate_audio_files(sources):
    checked = set()
    missing = []
    for source in sources:
        rows = source["items"]
        if source["schema"] != "ordinary_localization":
            rows = [row for group in rows for row in (group["AB"], group["BA"])]
        for row in rows:
            path = resolve_audio_path(source["audio_root"], row)
            checked.add(str(path))
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} training audios: {missing[:5]}")
    return {"checked_unique_audio_paths": len(checked), "missing": 0, "status": "pass"}


def source_metadata(sources):
    return [
        {
            "name": source["name"],
            "manifest": str(source["manifest"]),
            "manifest_sha256": sha256(source["manifest"]),
            "audio_root": str(source["audio_root"]),
            "weight": source["weight"],
            "schema": source["schema"],
            "items": len(source["items"]),
        }
        for source in sources
    ]


def candidate_answer(intervals):
    return answer_for(intervals) if intervals else "No matching interval."


def score_examples(model, examples, temperature):
    model.eval()
    with torch.no_grad():
        scores = torch.stack([sequence_score(model, example).float() for example in examples])
    return F.softmax(scores / temperature, dim=0).cpu()


def score_relation_reference(model, examples, temperature):
    candidate_count = len(examples) // 2
    if not candidate_count or len(examples) != 2 * candidate_count:
        raise ValueError("Relation reference requires equal AB and BA candidate sets")
    model.eval()
    with torch.no_grad():
        scores = torch.stack([sequence_score(model, example).float() for example in examples])
    return (
        F.softmax(scores[:candidate_count] / temperature, dim=0).cpu(),
        F.softmax(scores[candidate_count:] / temperature, dim=0).cpu(),
    )


def backward_outer(model, examples, objective_builder, scale=1.0):
    set_spotsound_training_mode(model, scope="full")
    rng_states = []
    with torch.no_grad():
        values = []
        for example in examples:
            rng_states.append(capture_rng_state())
            values.append(float(sequence_score(model, example).detach()))
        final_rng_state = capture_rng_state()
    device = next(model.parameters()).device
    variables = torch.tensor(values, device=device, dtype=torch.float32, requires_grad=True)
    outer, diagnostics = objective_builder(variables)
    scaled_outer = outer * scale
    coefficients = torch.autograd.grad(scaled_outer, variables)[0].detach()
    for example, coefficient, rng_state in zip(examples, coefficients, rng_states):
        if float(coefficient) == 0.0:
            continue
        restore_rng_state(rng_state)
        score = sequence_score(model, example)
        score.backward(gradient=coefficient.to(score.dtype))
    restore_rng_state(final_rng_state)
    return float(scaled_outer.detach()), diagnostics


def cosine_multiplier(step, total_steps, warmup, minimum):
    if warmup and step < warmup:
        return max((step + 1) / warmup, 1 / warmup)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_stage(model, adapter_path: Path, delta_path: Path, metadata: dict):
    adapter_path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(adapter_path, safe_serialization=True)
    save_bridge_delta(model, delta_path, metadata)
    return {
        "adapter": str(adapter_path.resolve()),
        "adapter_sha256": sha256(adapter_path / "adapter_model.safetensors"),
        "delta": str(delta_path.resolve()),
        "delta_sha256": sha256(delta_path),
    }


def main() -> int:
    args = arguments()
    outputs = (
        args.stage1_adapter,
        args.stage1_delta,
        args.output_adapter,
        args.output_delta,
        args.summary,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Refusing to overwrite a full SpotSound RelTwin run: {outputs}")
    if args.rbee_steps < 0 or args.setpo_steps < 0 or not args.rbee_steps + args.setpo_steps:
        raise ValueError("At least one training stage must contain steps")
    positive = (
        args.lora_learning_rate,
        args.bridge_learning_rate,
        args.prediction_temperature,
        args.target_temperature,
        args.reference_temperature,
    )
    nonnegative = (
        args.method_weight,
        args.exchange_weight,
        args.rbee_replay_weight,
        args.setpo_replay_weight,
        args.relation_reference_kl_weight,
        args.replay_reference_kl_weight,
        args.warmup_steps,
    )
    if min(positive) <= 0 or min(nonnegative) < 0 or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("Invalid optimizer or objective settings")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    twin_sources = parse_sources(args.twin_source, "twin")
    replay_sources = parse_sources(args.replay_source, "replay")
    twin_sources, replay_sources = prepare_source_pools(twin_sources, replay_sources)
    data_report = {
        "twin_sources": source_metadata(twin_sources),
        "replay_sources": source_metadata(replay_sources),
        "leakage_audit": audit_no_evaluation_overlap(
            twin_sources, replay_sources, args.eval_manifest
        ),
        "audio_audit": validate_audio_files([*twin_sources, *replay_sources]),
    }
    if args.audit_only:
        print(json.dumps(data_report, ensure_ascii=False, indent=2))
        return 0

    processor = AudioFlamingo3TemporalProcessor.from_pretrained(args.base)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model, args.adapter, torch_device="cpu", is_trainable=True
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainable, parameter_report = configure_spotsound_parameters(model, scope="full")
    named_trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    lora_parameters = [
        parameter for name, parameter in named_trainable if parameter_component(name) == "language_lora"
    ]
    bridge_parameters = [
        parameter for name, parameter in named_trainable if parameter_component(name) != "language_lora"
    ]
    if not lora_parameters or not bridge_parameters:
        raise RuntimeError("Full SpotSound RelTwin requires LoRA and bridge parameters")
    model = model.to("cuda")
    set_spotsound_training_mode(model, scope="full")
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.lora_learning_rate},
            {"params": bridge_parameters, "lr": args.bridge_learning_rate},
        ],
        weight_decay=0.0,
    )
    total_steps = args.rbee_steps + args.setpo_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(step, total_steps, args.warmup_steps, args.min_lr_ratio),
    )

    rbee_twins = weighted_schedule(
        [len(source["items"]) for source in twin_sources],
        [source["weight"] for source in twin_sources],
        args.rbee_steps,
        rng,
    )
    setpo_twins = weighted_schedule(
        [len(source["items"]) for source in twin_sources],
        [source["weight"] for source in twin_sources],
        args.setpo_steps,
        rng,
    )
    rbee_replay = weighted_schedule(
        [len(source["items"]) for source in replay_sources],
        [source["weight"] for source in replay_sources],
        args.rbee_steps,
        rng,
    )
    setpo_replay = weighted_schedule(
        [len(source["items"]) for source in replay_sources],
        [source["weight"] for source in replay_sources],
        args.setpo_steps,
        rng,
    )

    wave_cache: dict[str, np.ndarray] = {}

    def wave(source, row):
        path = resolve_audio_path(source["audio_root"], row)
        key = str(path.resolve())
        if key not in wave_cache:
            if args.cache_audios > 0 and len(wave_cache) >= args.cache_audios:
                wave_cache.pop(next(iter(wave_cache)))
            audio, _ = librosa.load(path, sr=16000, mono=True)
            wave_cache[key] = np.asarray(audio, dtype=np.float32)
        return wave_cache[key]

    def rbee_bundle(source_index, item_index):
        source = twin_sources[source_index]
        group = source["items"][item_index]
        ab, ba = group["AB"], group["BA"]
        audio = wave(source, ab)
        answer_ab, answer_ba = answer_for(ab["annotations"]), answer_for(ba["annotations"])
        return [
            prepare_example(processor, audio, ab["caption"], answer_ab),
            prepare_example(processor, audio, ab["caption"], answer_ba),
            prepare_example(processor, audio, ba["caption"], answer_ab),
            prepare_example(processor, audio, ba["caption"], answer_ba),
        ]

    def relation_bundle(source_index, item_index):
        source = twin_sources[source_index]
        group = source["items"][item_index]
        ab, ba = group["AB"], group["BA"]
        audio = wave(source, ab)
        duration = len(audio) / 16000.0
        candidates = relation_candidates(
            ab["annotations"], ba["annotations"], duration, args.jitter_ratio
        )
        answers = [candidate_answer(candidate) for candidate in candidates]
        examples = [
            prepare_example(processor, audio, ab["caption"], answer) for answer in answers
        ]
        examples.extend(
            prepare_example(processor, audio, ba["caption"], answer) for answer in answers
        )
        return {
            "examples": examples,
            "qualities_ab": candidate_quality_axes(ab["annotations"], candidates, duration),
            "qualities_ba": candidate_quality_axes(ba["annotations"], candidates, duration),
        }

    def replay_bundle(source_index, item_index, candidates):
        source = replay_sources[source_index]
        row = source["items"][item_index]
        audio = wave(source, row)
        exact = prepare_example(processor, audio, row["caption"], answer_for(row["annotations"]))
        if not candidates:
            return {"examples": [exact]}
        duration = len(audio) / 16000.0
        intervals = scale_cardinality_candidates(
            row["annotations"], duration, args.jitter_ratio
        )
        return {
            "examples": [
                prepare_example(processor, audio, row["caption"], candidate_answer(candidate))
                for candidate in intervals
            ],
            "qualities": candidate_quality_axes(row["annotations"], intervals, duration),
        }

    official_twin_references = {}
    for completed, key in enumerate(sorted(set(setpo_twins)), start=1):
        bundle = relation_bundle(*key)
        official_twin_references[key] = score_relation_reference(
            model, bundle["examples"], args.reference_temperature
        )
        if completed == 1 or completed % 16 == 0:
            print(
                json.dumps(
                    {
                        "official_twin_reference_precompute": completed,
                        "official_twin_reference_total": len(set(setpo_twins)),
                    }
                ),
                flush=True,
            )

    official_replay_references = {}
    for completed, key in enumerate(sorted(set(setpo_replay)), start=1):
        bundle = replay_bundle(*key, candidates=True)
        official_replay_references[key] = score_examples(
            model, bundle["examples"], args.reference_temperature
        )
        if completed == 1 or completed % 16 == 0:
            print(
                json.dumps(
                    {
                        "official_reference_precompute": completed,
                        "official_reference_total": len(set(setpo_replay)),
                    }
                ),
                flush=True,
            )
    set_spotsound_training_mode(model, scope="full")

    history = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def finish_step(stage, step, diagnostics, loss_value, source_names):
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).detach())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at {stage} step {step + 1}")
        optimizer.step()
        scheduler.step()
        event = {
            "global_step": len(history) + 1,
            "stage": stage,
            "stage_step": step + 1,
            "loss": loss_value,
            "grad_norm": grad_norm,
            "lora_lr": optimizer.param_groups[0]["lr"],
            "bridge_lr": optimizer.param_groups[1]["lr"],
            **source_names,
            **diagnostics,
        }
        history.append(event)
        if step == 0 or (step + 1) % 4 == 0:
            print(json.dumps(event), flush=True)

    for step, (twin_key, replay_key) in enumerate(zip(rbee_twins, rbee_replay)):
        optimizer.zero_grad(set_to_none=True)
        relation_value, diagnostics = backward_outer(
            model,
            rbee_bundle(*twin_key),
            lambda scores: outer_objective(
                "rbee",
                scores,
                args.prediction_temperature,
                args.method_weight,
                0.2,
                args.exchange_weight,
            ),
        )
        replay = replay_bundle(*replay_key, candidates=False)
        set_spotsound_training_mode(model, scope="full")
        replay_loss = -sequence_score(model, replay["examples"][0])
        (args.rbee_replay_weight * replay_loss).backward()
        diagnostics["replay_sft"] = float(replay_loss.detach())
        total_value = relation_value + args.rbee_replay_weight * float(replay_loss.detach())
        finish_step(
            "rbee",
            step,
            diagnostics,
            total_value,
            {
                "twin_source": twin_sources[twin_key[0]]["name"],
                "replay_source": replay_sources[replay_key[0]]["name"],
            },
        )

    stage1_metadata = {
        "method": "SpotSound-A Full RelTwin",
        "stage": "rbee",
        "steps": args.rbee_steps,
        "seed": args.seed,
        "official_adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "parameter_report": parameter_report,
        "data": data_report,
    }
    stage1_report = save_stage(
        model, args.stage1_adapter, args.stage1_delta, stage1_metadata
    )

    for step, (twin_key, replay_key) in enumerate(zip(setpo_twins, setpo_replay)):
        optimizer.zero_grad(set_to_none=True)
        relation = relation_bundle(*twin_key)
        qualities_ab = torch.tensor(relation["qualities_ab"], device="cuda")
        qualities_ba = torch.tensor(relation["qualities_ba"], device="cuda")
        reference_ab, reference_ba = official_twin_references[twin_key]
        relation_value, diagnostics = backward_outer(
            model,
            relation["examples"],
            lambda scores: relation_objective(
                scores,
                qualities_ab,
                qualities_ba,
                args.prediction_temperature,
                args.target_temperature,
                args.method_weight,
                args.exchange_weight,
                reference_ab,
                reference_ba,
                args.relation_reference_kl_weight,
                quality_mode="pareto",
                pareto_weight=args.pareto_weight,
            ),
        )
        replay = replay_bundle(*replay_key, candidates=True)
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
                official_replay_references[replay_key],
                args.replay_reference_kl_weight,
                quality_mode="pareto",
                pareto_weight=args.pareto_weight,
            ),
            scale=args.setpo_replay_weight,
        )
        diagnostics.update(replay_diagnostics)
        finish_step(
            "setpo",
            step,
            diagnostics,
            relation_value + replay_value,
            {
                "twin_source": twin_sources[twin_key[0]]["name"],
                "replay_source": replay_sources[replay_key[0]]["name"],
            },
        )

    metadata = {
        "method": "SpotSound-A Full RelTwin",
        "stages": ["rbee", "setpo"],
        "rbee_steps": args.rbee_steps,
        "setpo_steps": args.setpo_steps,
        "lora_learning_rate": args.lora_learning_rate,
        "bridge_learning_rate": args.bridge_learning_rate,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "prediction_temperature": args.prediction_temperature,
        "target_temperature": args.target_temperature,
        "method_weight": args.method_weight,
        "exchange_weight": args.exchange_weight,
        "rbee_replay_weight": args.rbee_replay_weight,
        "setpo_replay_weight": args.setpo_replay_weight,
        "relation_reference_kl_weight": args.relation_reference_kl_weight,
        "replay_reference_kl_weight": args.replay_reference_kl_weight,
        "reference_temperature": args.reference_temperature,
        "jitter_ratio": args.jitter_ratio,
        "pareto_weight": args.pareto_weight,
        "seed": args.seed,
        "official_adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "parameter_report": parameter_report,
        "data": data_report,
    }
    final_report = save_stage(model, args.output_adapter, args.output_delta, metadata)
    summary = {
        **metadata,
        "official_twin_reference_examples": len(official_twin_references),
        "official_replay_reference_examples": len(official_replay_references),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "stage1": stage1_report,
        "final": final_report,
        "final_event": history[-1],
        "history": history,
    }
    write_json(args.summary, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
