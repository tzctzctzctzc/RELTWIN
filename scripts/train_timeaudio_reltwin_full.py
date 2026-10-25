#!/usr/bin/env python3
"""Full two-stage RelTwin adaptation for the public TimeAudio checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from setpo_candidates import (
    candidate_quality_axes,
    relation_candidates,
    scale_cardinality_candidates,
)
from setpo_objective import relation_objective, rehearsal_objective
from timeaudio_reltwin import (
    build_timeaudio_sample,
    configure_reltwin_parameters,
    group_twin_rows,
    load_timeaudio_model,
    move_sample,
    prepare_audio_features,
    rbee_objective,
    resolve_audio_path,
    save_trainable_delta,
    set_reltwin_training_mode,
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
    parser.add_argument("--initial-delta", type=Path)
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
    parser.add_argument("--output-delta", type=Path, required=True)
    parser.add_argument("--stage1-delta", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--rbee-steps", type=int, default=384)
    parser.add_argument("--setpo-steps", type=int, default=128)
    parser.add_argument("--lora-learning-rate", type=float, default=2e-6)
    parser.add_argument("--temporal-learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--prediction-temperature", type=float, default=1.0)
    parser.add_argument("--target-temperature", type=float, default=0.15)
    parser.add_argument("--method-weight", type=float, default=1.0)
    parser.add_argument("--exchange-weight", type=float, default=1.0)
    parser.add_argument("--rbee-replay-weight", type=float, default=0.75)
    parser.add_argument("--setpo-replay-weight", type=float, default=0.75)
    parser.add_argument("--replay-reference-kl-weight", type=float, default=0.2)
    parser.add_argument("--reference-temperature", type=float, default=1.0)
    parser.add_argument("--jitter-ratio", type=float, default=0.15)
    parser.add_argument("--pareto-weight", type=float, default=0.5)
    parser.add_argument("--cache-audios", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    return rows


def parse_sources(raw_sources, kind: str) -> list[dict]:
    sources = []
    names = set()
    for name, manifest, audio_root, weight in raw_sources:
        if name in names:
            raise ValueError(f"Duplicate {kind} source name: {name}")
        names.add(name)
        manifest = Path(manifest).resolve()
        audio_root = Path(audio_root).resolve()
        numeric_weight = float(weight)
        if numeric_weight <= 0:
            raise ValueError(f"{kind} source weight must be positive: {name}")
        rows = _read_json(manifest)
        sources.append(
            {
                "name": name,
                "manifest": manifest,
                "audio_root": audio_root,
                "weight": numeric_weight,
                "rows": rows,
            }
        )
    return sources


def weighted_schedule(
    sizes: list[int], weights: list[float], steps: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    if steps < 0 or not sizes or len(sizes) != len(weights):
        raise ValueError("Invalid weighted schedule request")
    if any(size <= 0 for size in sizes) or any(weight <= 0 for weight in weights):
        raise ValueError("Schedule sizes and weights must be positive")
    if steps == 0:
        return []
    normalized = np.asarray(weights, dtype=float) / sum(weights)
    raw_counts = normalized * steps
    counts = np.floor(raw_counts).astype(int)
    for index in np.argsort(-(raw_counts - counts))[: steps - int(counts.sum())]:
        counts[index] += 1
    source_order = np.concatenate(
        [np.full(count, index, dtype=int) for index, count in enumerate(counts)]
    )
    rng.shuffle(source_order)
    permutations = [rng.permutation(size).tolist() for size in sizes]
    positions = [0] * len(sizes)
    schedule = []
    for source_index in source_order.tolist():
        if positions[source_index] == len(permutations[source_index]):
            permutations[source_index] = rng.permutation(sizes[source_index]).tolist()
            positions[source_index] = 0
        row_index = permutations[source_index][positions[source_index]]
        positions[source_index] += 1
        schedule.append((source_index, row_index))
    return schedule


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


def _identity(rows: list[dict]):
    audio = {Path(row["audio_path"]).stem for row in rows}
    queries = {
        (Path(row["audio_path"]).stem, " ".join(str(row["caption"]).lower().split()))
        for row in rows
    }
    return audio, queries


def audit_no_evaluation_overlap(
    twin_sources: list[dict], replay_sources: list[dict], eval_manifests: list[Path]
) -> dict:
    training_rows = []
    for source in twin_sources:
        for group in source["items"]:
            training_rows.extend((group["AB"], group["BA"]))
    for source in replay_sources:
        training_rows.extend(source["items"])
    train_audio, train_queries = _identity(training_rows)
    reports = []
    for manifest in eval_manifests:
        rows = _read_json(manifest)
        eval_audio, eval_queries = _identity(rows)
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


def validate_audio_files(sources: list[dict]) -> dict:
    missing = []
    checked = set()
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


def candidate_answer(intervals) -> str:
    return timeaudio_answer(intervals) if intervals else "No matching interval."


def sequence_score(model, sample):
    loss = model(move_sample(sample))["loss"]
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite TimeAudio sequence loss: {loss}")
    return -loss


def score_examples(model, examples, temperature: float) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        scores = torch.stack([sequence_score(model, example).float() for example in examples])
    return F.softmax(scores / temperature, dim=0).cpu()


def backward_outer(model, examples, objective_builder, scale: float = 1.0):
    set_reltwin_training_mode(model, scope="full")
    rng_states = []
    with torch.no_grad():
        values = []
        for example in examples:
            rng_states.append(torch.cuda.get_rng_state())
            values.append(float(sequence_score(model, example).detach()))
        final_rng_state = torch.cuda.get_rng_state()
    variables = torch.tensor(values, device="cuda", dtype=torch.float32, requires_grad=True)
    outer, diagnostics = objective_builder(variables)
    scaled_outer = outer * scale
    coefficients = torch.autograd.grad(scaled_outer, variables)[0].detach()
    for example, coefficient, rng_state in zip(examples, coefficients, rng_states):
        if float(coefficient) == 0.0:
            continue
        torch.cuda.set_rng_state(rng_state)
        sequence_score(model, example).backward(gradient=coefficient)
    torch.cuda.set_rng_state(final_rng_state)
    return float(scaled_outer.detach()), diagnostics


def cosine_multiplier(step: int, total_steps: int, warmup: int, minimum: float) -> float:
    if warmup and step < warmup:
        return max((step + 1) / warmup, 1 / warmup)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def source_metadata(sources: list[dict]) -> list[dict]:
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


def main() -> int:
    args = arguments()
    outputs = (args.output_delta, args.stage1_delta, args.summary)
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Refusing to overwrite a full RelTwin run: {outputs}")
    if args.rbee_steps < 0 or args.setpo_steps < 0 or not args.rbee_steps + args.setpo_steps:
        raise ValueError("At least one training stage must contain steps")
    positive = (
        args.lora_learning_rate,
        args.temporal_learning_rate,
        args.prediction_temperature,
        args.target_temperature,
        args.reference_temperature,
    )
    if min(positive) <= 0:
        raise ValueError("Learning rates and temperatures must be positive")
    nonnegative = (
        args.method_weight,
        args.exchange_weight,
        args.rbee_replay_weight,
        args.setpo_replay_weight,
        args.replay_reference_kl_weight,
        args.warmup_steps,
    )
    if min(nonnegative) < 0 or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("Invalid objective or schedule weights")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    twin_sources = parse_sources(args.twin_source, "twin")
    replay_sources = parse_sources(args.replay_source, "replay")
    twin_sources, replay_sources = prepare_source_pools(twin_sources, replay_sources)
    leakage_audit = audit_no_evaluation_overlap(
        twin_sources, replay_sources, args.eval_manifest
    )
    audio_audit = validate_audio_files([*twin_sources, *replay_sources])
    data_report = {
        "twin_sources": source_metadata(twin_sources),
        "replay_sources": source_metadata(replay_sources),
        "leakage_audit": leakage_audit,
        "audio_audit": audio_audit,
    }
    if args.audit_only:
        print(json.dumps(data_report, ensure_ascii=False, indent=2))
        return 0

    from transformers import WhisperFeatureExtractor

    model, _, load_report = load_timeaudio_model(
        args.timeaudio_repo,
        args.config,
        args.llama,
        args.whisper,
        args.beats,
        args.bert,
        args.checkpoint,
        args.initial_delta,
    )
    trainable, parameter_report = configure_reltwin_parameters(model, scope="full")
    named_trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    lora_parameters = [parameter for name, parameter in named_trainable if "lora_" in name]
    temporal_parameters = [parameter for name, parameter in named_trainable if "lora_" not in name]
    if not lora_parameters or not temporal_parameters:
        raise RuntimeError("Full RelTwin requires both LoRA and temporal parameters")
    model = model.to("cuda")
    set_reltwin_training_mode(model, scope="full")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.lora_learning_rate},
            {"params": temporal_parameters, "lr": args.temporal_learning_rate},
        ],
        weight_decay=0.0,
    )
    total_steps = args.rbee_steps + args.setpo_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(
            step, total_steps, args.warmup_steps, args.min_lr_ratio
        ),
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

    feature_cache: dict[str, dict] = {}

    def features(source, row):
        path = resolve_audio_path(source["audio_root"], row)
        key = str(path.resolve())
        if key not in feature_cache:
            if args.cache_audios > 0 and len(feature_cache) >= args.cache_audios:
                feature_cache.pop(next(iter(feature_cache)))
            feature_cache[key] = prepare_audio_features(path, feature_extractor)
        return feature_cache[key]

    def rbee_bundle(source_index, item_index):
        source = twin_sources[source_index]
        group = source["items"][item_index]
        ab, ba = group["AB"], group["BA"]
        audio_features = features(source, ab)
        answer_ab = timeaudio_answer(ab["annotations"])
        answer_ba = timeaudio_answer(ba["annotations"])
        return [
            build_timeaudio_sample(audio_features, ab["caption"], answer_ab),
            build_timeaudio_sample(audio_features, ab["caption"], answer_ba),
            build_timeaudio_sample(audio_features, ba["caption"], answer_ab),
            build_timeaudio_sample(audio_features, ba["caption"], answer_ba),
        ]

    def relation_bundle(source_index, item_index):
        source = twin_sources[source_index]
        group = source["items"][item_index]
        ab, ba = group["AB"], group["BA"]
        audio_features = features(source, ab)
        duration = float(audio_features["duration"][0])
        candidates = relation_candidates(
            ab["annotations"], ba["annotations"], duration, args.jitter_ratio
        )
        answers = [candidate_answer(candidate) for candidate in candidates]
        examples = [
            build_timeaudio_sample(audio_features, ab["caption"], answer)
            for answer in answers
        ]
        examples.extend(
            build_timeaudio_sample(audio_features, ba["caption"], answer)
            for answer in answers
        )
        return {
            "examples": examples,
            "qualities_ab": candidate_quality_axes(
                ab["annotations"], candidates, duration
            ),
            "qualities_ba": candidate_quality_axes(
                ba["annotations"], candidates, duration
            ),
        }

    def replay_bundle(source_index, item_index, candidates: bool):
        source = replay_sources[source_index]
        row = source["items"][item_index]
        audio_features = features(source, row)
        exact = build_timeaudio_sample(
            audio_features, row["caption"], timeaudio_answer(row["annotations"])
        )
        if not candidates:
            return {"examples": [exact]}
        duration = float(audio_features["duration"][0])
        intervals = scale_cardinality_candidates(
            row["annotations"], duration, args.jitter_ratio
        )
        return {
            "examples": [
                build_timeaudio_sample(
                    audio_features, row["caption"], candidate_answer(candidate)
                )
                for candidate in intervals
            ],
            "qualities": candidate_quality_axes(
                row["annotations"], intervals, duration
            ),
        }

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
    set_reltwin_training_mode(model, scope="full")

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
            "temporal_lr": optimizer.param_groups[1]["lr"],
            **source_names,
            **diagnostics,
        }
        history.append(event)
        if step == 0 or (step + 1) % 4 == 0:
            print(json.dumps(event), flush=True)

    for step, (twin_key, replay_key) in enumerate(zip(rbee_twins, rbee_replay)):
        optimizer.zero_grad(set_to_none=True)
        examples = rbee_bundle(*twin_key)
        relation_value, diagnostics = backward_outer(
            model,
            examples,
            lambda scores: rbee_objective(
                scores,
                args.prediction_temperature,
                args.method_weight,
                args.exchange_weight,
            ),
        )
        replay = replay_bundle(*replay_key, candidates=False)
        set_reltwin_training_mode(model, scope="full")
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
        "method": "TimeAudio Full RelTwin",
        "stage": "rbee",
        "steps": args.rbee_steps,
        "seed": args.seed,
        "official_checkpoint_sha256": sha256(args.checkpoint),
        "parameter_report": parameter_report,
        "data": data_report,
    }
    save_trainable_delta(model, args.stage1_delta, stage1_metadata)

    for step, (twin_key, replay_key) in enumerate(zip(setpo_twins, setpo_replay)):
        optimizer.zero_grad(set_to_none=True)
        relation = relation_bundle(*twin_key)
        qualities_ab = torch.tensor(relation["qualities_ab"], device="cuda")
        qualities_ba = torch.tensor(relation["qualities_ba"], device="cuda")
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
        "method": "TimeAudio Full RelTwin",
        "stages": ["rbee", "setpo"],
        "rbee_steps": args.rbee_steps,
        "setpo_steps": args.setpo_steps,
        "lora_learning_rate": args.lora_learning_rate,
        "temporal_learning_rate": args.temporal_learning_rate,
        "warmup_steps": args.warmup_steps,
        "min_lr_ratio": args.min_lr_ratio,
        "prediction_temperature": args.prediction_temperature,
        "target_temperature": args.target_temperature,
        "method_weight": args.method_weight,
        "exchange_weight": args.exchange_weight,
        "rbee_replay_weight": args.rbee_replay_weight,
        "setpo_replay_weight": args.setpo_replay_weight,
        "replay_reference_kl_weight": args.replay_reference_kl_weight,
        "reference_temperature": args.reference_temperature,
        "jitter_ratio": args.jitter_ratio,
        "pareto_weight": args.pareto_weight,
        "seed": args.seed,
        "official_checkpoint_sha256": sha256(args.checkpoint),
        "initial_delta_sha256": sha256(args.initial_delta) if args.initial_delta else None,
        "parameter_report": parameter_report,
        "data": data_report,
    }
    save_trainable_delta(model, args.output_delta, metadata)
    summary = {
        **metadata,
        "load_report": load_report,
        "official_reference_examples": len(official_replay_references),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "stage1_delta": str(args.stage1_delta.resolve()),
        "stage1_delta_sha256": sha256(args.stage1_delta),
        "output_delta": str(args.output_delta.resolve()),
        "delta_sha256": sha256(args.output_delta),
        "final_event": history[-1],
        "history": history,
    }
    write_json(args.summary, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
