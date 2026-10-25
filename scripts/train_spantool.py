#!/usr/bin/env python3
"""Train SpanTool head-only or jointly with the existing SpotSound LoRA."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel

from interval_metrics import temporal_set_iou
from spantool import SpanToolConfig, SpanToolHead, decode_spantool_output, spantool_loss
from spantool_runtime import (
    artifact_sha256,
    file_sha256,
    infer_hidden_dim,
    load_manifest,
    load_spantool_checkpoint,
    load_wave,
    move_model_inputs,
    prepare_spantool_input,
    save_spantool_checkpoint,
    select_audio_states,
    validate_rows,
)
from spotsound import (
    AudioFlamingo3ForTemporalConditionalGeneration,
    AudioFlamingo3SpanToolProcessor,
)


@dataclass
class TrainSource:
    name: str
    manifest: Path
    audio_root: Path
    weight: float
    rows: list[dict]
    order: list[int]
    position: int = 0

    def next_row(self, rng: random.Random) -> dict:
        if self.position >= len(self.order):
            rng.shuffle(self.order)
            self.position = 0
        row = self.rows[self.order[self.position]]
        self.position += 1
        return row


@dataclass
class ValidationSource:
    name: str
    manifest: Path
    audio_root: Path
    baseline_miou: float
    rows: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-head", type=Path)
    parser.add_argument(
        "--train-source", action="append", nargs=4,
        metavar=("NAME", "MANIFEST", "AUDIO_ROOT", "WEIGHT"), required=True,
    )
    parser.add_argument(
        "--validation-source", action="append", nargs=4,
        metavar=("NAME", "MANIFEST", "AUDIO_ROOT", "BASELINE_MIOU"),
    )
    parser.add_argument("--head-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio-layer", type=int, default=-1)
    parser.add_argument(
        "--audio-layers", type=int, nargs="+",
        help="Hidden layers to fuse; overrides --audio-layer when provided",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--transform-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--primary-pool", type=int, default=5)
    parser.add_argument("--coarse-pool", type=int, default=5)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--max-segment-seconds", type=float, default=0.0)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--updates", type=int, default=0)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--head-learning-rate", type=float, default=2e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--structural-weight", type=float, default=1.0)
    parser.add_argument("--occupancy-loss-weight", type=float, default=0.5)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.25)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.1)
    parser.add_argument("--risk-weight", type=float, default=0.0)
    parser.add_argument("--risk-samples", type=int, default=0)
    parser.add_argument("--risk-temperature", type=float, default=1.0)
    parser.add_argument("--risk-noise", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_sources(args: argparse.Namespace, rng: random.Random):
    train_sources = []
    for name, manifest, audio_root, weight in args.train_source:
        manifest_path = Path(manifest)
        rows = load_manifest(manifest_path)
        validate_rows(rows, name)
        numeric_weight = float(weight)
        if numeric_weight <= 0:
            raise ValueError(f"Source weight must be positive: {name}")
        order = list(range(len(rows)))
        rng.shuffle(order)
        train_sources.append(
            TrainSource(
                name, manifest_path, Path(audio_root), numeric_weight, rows, order
            )
        )
    validation_sources = []
    for values in args.validation_source or []:
        name, manifest, audio_root, baseline = values
        manifest_path = Path(manifest)
        rows = load_manifest(manifest_path)
        validate_rows(rows, name)
        validation_sources.append(
            ValidationSource(name, manifest_path, Path(audio_root), float(baseline), rows)
        )
    return train_sources, validation_sources


def git_state(project_root: Path) -> dict:
    def run(*command: str) -> str:
        result = subprocess.run(
            command, cwd=project_root, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()

    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "status": run("git", "status", "--short"),
        "diff_sha256": __import__("hashlib").sha256(
            run("git", "diff", "--binary").encode()
        ).hexdigest(),
    }


def run_backbone(
    model,
    processor,
    wave: np.ndarray,
    query: str,
    audio_layers: list[int],
    device: torch.device,
    head_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = move_model_inputs(
        prepare_spantool_input(processor, wave, query), device, model.dtype
    )
    context = torch.no_grad() if head_only else torch.enable_grad()
    with context:
        outputs = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        states, mask = select_audio_states(
            outputs,
            audio_layers,
            inputs["input_ids"],
            model.config.audio_token_id,
        )
    return states.float(), mask


def validation_subset(source: ValidationSource, limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(source.rows):
        return source.rows
    keyed = []
    for index, row in enumerate(source.rows):
        key = f"{seed}|{source.name}|{row.get('qid', row.get('benchmark_id', index))}"
        keyed.append((__import__("hashlib").sha256(key.encode()).hexdigest(), row))
    keyed.sort(key=lambda pair: pair[0])
    return [row for _, row in keyed[:limit]]


@torch.no_grad()
def evaluate_sources(
    model,
    processor,
    head: SpanToolHead,
    sources: list[ValidationSource],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()
    head.eval()
    results = {}
    improvements = []
    for source in sources:
        rows = validation_subset(source, args.validation_limit, args.seed)
        ious = []
        for row in rows:
            wave = load_wave(source.audio_root, row)
            duration = len(wave) / 16000.0
            states, mask = run_backbone(
                model, processor, wave, row["caption"], args.selected_audio_layers,
                device, True
            )
            output = head(states, mask, torch.tensor([duration], device=device))
            prediction = decode_spantool_output(head, output, 0, duration)
            ious.append(temporal_set_iou(row["annotations"], prediction))
        values = np.asarray(ious, dtype=np.float64)
        miou = float(values.mean() * 100) if len(values) else float("-inf")
        results[source.name] = {
            "rows": len(rows),
            "mIoU": miou,
            "R1@0.3": float((values >= 0.3).mean() * 100),
            "R1@0.5": float((values >= 0.5).mean() * 100),
            "R1@0.7": float((values >= 0.7).mean() * 100),
            "baseline_mIoU": source.baseline_miou,
            "improvement": miou - source.baseline_miou,
        }
        improvements.append(miou - source.baseline_miou)
    score = min(improvements) if improvements else float("-inf")
    return score, results


def make_scheduler(optimizer, total_updates: int, warmup_ratio: float):
    warmup = round(total_updates * warmup_ratio)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-3, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total_updates - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor), warmup


def main() -> int:
    args = parse_args()
    args.selected_audio_layers = args.audio_layers or [args.audio_layer]
    if args.gradient_accumulation < 1:
        raise ValueError("gradient accumulation must be positive")
    rng = random.Random(args.seed)
    risk_rng = random.Random(args.seed ^ 0x5A17)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_sources, validation_sources = parse_sources(args, rng)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    processor = AudioFlamingo3SpanToolProcessor.from_pretrained(args.base)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model, args.adapter, torch_device="cpu", is_trainable=not args.head_only
    )
    if args.head_only:
        model.requires_grad_(False)
        model.eval()
    else:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()
    model = model.to(device)

    head = SpanToolHead(
        SpanToolConfig(
            input_dim=infer_hidden_dim(model),
            layer_count=len(args.selected_audio_layers),
            hidden_dim=args.hidden_dim,
            transformer_layers=args.transform_layers,
            attention_heads=args.attention_heads,
            dropout=args.dropout,
            primary_pool=args.primary_pool,
            coarse_pool=args.coarse_pool,
            max_events=args.max_events,
            max_segment_seconds=args.max_segment_seconds,
        )
    ).to(device)
    if args.initial_head:
        initial_head, initial_payload = load_spantool_checkpoint(args.initial_head, device)
        initial_layers = initial_payload.get(
            "audio_layers", [initial_payload["audio_layer"]]
        )
        if initial_head.config != head.config:
            raise ValueError("Initial SpanTool head configuration does not match this run")
        if list(initial_layers) != args.selected_audio_layers:
            raise ValueError("Initial SpanTool audio layers do not match this run")
        head.load_state_dict(initial_head.state_dict())
        del initial_head

    parameter_groups = [
        {
            "params": list(head.parameters()),
            "lr": args.head_learning_rate,
            "weight_decay": args.weight_decay,
        }
    ]
    adapter_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if adapter_parameters:
        parameter_groups.append(
            {
                "params": adapter_parameters,
                "lr": args.adapter_learning_rate,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    examples_per_epoch = sum(len(source.rows) for source in train_sources)
    total_updates = args.updates or math.ceil(
        args.epochs * examples_per_epoch / args.gradient_accumulation
    )
    scheduler, warmup_updates = make_scheduler(
        optimizer, total_updates, args.warmup_ratio
    )
    source_names = [source.name for source in train_sources]
    source_weights = [source.weight for source in train_sources]
    source_counts: Counter[str] = Counter()
    loss_sums: Counter[str] = Counter()
    validation_history = []
    best_score = float("-inf")
    best_update = None
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for update in range(1, total_updates + 1):
        if not args.head_only:
            model.train()
        head.train()
        update_diagnostics: Counter[str] = Counter()
        for _ in range(args.gradient_accumulation):
            source = rng.choices(train_sources, weights=source_weights, k=1)[0]
            row = source.next_row(rng)
            source_counts[source.name] += 1
            wave = load_wave(source.audio_root, row)
            duration = len(wave) / 16000.0
            states, mask = run_backbone(
                model, processor, wave, row["caption"], args.selected_audio_layers, device,
                args.head_only,
            )
            output = head(states, mask, torch.tensor([duration], device=device))
            loss, diagnostics = spantool_loss(
                head,
                output,
                [row["annotations"]],
                torch.tensor([duration], device=device),
                structural_weight=args.structural_weight,
                occupancy_weight=args.occupancy_loss_weight,
                boundary_weight=args.boundary_loss_weight,
                count_weight=args.count_loss_weight,
                consistency_weight=args.consistency_loss_weight,
                risk_weight=args.risk_weight,
                risk_samples=args.risk_samples,
                risk_temperature=args.risk_temperature,
                risk_noise=args.risk_noise,
                rng=risk_rng,
            )
            (loss / args.gradient_accumulation).backward()
            for name, value in diagnostics.items():
                update_diagnostics[name] += value / args.gradient_accumulation
                loss_sums[name] += value

        trainable = list(head.parameters()) + adapter_parameters
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        record = {
            "update": update,
            "total_updates": total_updates,
            "loss": dict(update_diagnostics),
            "grad_norm": float(grad_norm),
            "head_lr": optimizer.param_groups[0]["lr"],
            "source_counts": dict(source_counts),
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)

        should_evaluate = bool(validation_sources) and (
            update % args.eval_every == 0 or update == total_updates
        )
        if should_evaluate:
            score, validation = evaluate_sources(
                model, processor, head, validation_sources, args, device
            )
            evaluation = {"update": update, "selection_score": score, "sources": validation}
            validation_history.append(evaluation)
            print(json.dumps({"validation": evaluation}, ensure_ascii=False), flush=True)
            if score > best_score:
                best_score = score
                best_update = update
                best_dir = args.output_dir / "best"
                save_spantool_checkpoint(
                    best_dir,
                    head,
                    audio_layers=args.selected_audio_layers,
                    metadata={"update": update, "selection_score": score},
                )
                if not args.head_only:
                    model.save_pretrained(best_dir / "adapter")
            if not args.head_only:
                model.train()
            head.train()

    final_dir = args.output_dir / "final"
    save_spantool_checkpoint(
        final_dir,
        head,
        audio_layers=args.selected_audio_layers,
        metadata={"update": total_updates},
    )
    if not args.head_only:
        model.save_pretrained(final_dir / "adapter")
    if not validation_sources:
        best_score = None
        best_update = total_updates

    elapsed = time.perf_counter() - started
    project_root = Path(__file__).resolve().parents[1]
    summary = {
        "method": "SpanTool",
        "head_only": args.head_only,
        "base": str(args.base.resolve()),
        "start_adapter": str(args.adapter.resolve()),
        "start_adapter_sha256": artifact_sha256(args.adapter),
        "initial_head": str(args.initial_head.resolve()) if args.initial_head else None,
        "initial_head_sha256": (
            file_sha256(args.initial_head / "spantool_head.pt")
            if args.initial_head else None
        ),
        "audio_layer": args.selected_audio_layers[0],
        "audio_layers": args.selected_audio_layers,
        "head_config": head.config.to_dict(),
        "seed": args.seed,
        "examples_per_epoch": examples_per_epoch,
        "gradient_accumulation": args.gradient_accumulation,
        "total_updates": total_updates,
        "warmup_updates": warmup_updates,
        "optimization": {
            "epochs": args.epochs,
            "head_learning_rate": args.head_learning_rate,
            "adapter_learning_rate": args.adapter_learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
        },
        "loss_weights": {
            "structural": args.structural_weight,
            "occupancy": args.occupancy_loss_weight,
            "boundary": args.boundary_loss_weight,
            "count": args.count_loss_weight,
            "consistency": args.consistency_loss_weight,
            "risk": args.risk_weight,
            "risk_samples": args.risk_samples,
            "risk_temperature": args.risk_temperature,
            "risk_noise": args.risk_noise,
        },
        "validation": {
            "eval_every": args.eval_every,
            "limit_per_source": args.validation_limit,
        },
        "source_counts": dict(source_counts),
        "train_sources": [
            {
                "name": source.name,
                "manifest": str(source.manifest.resolve()),
                "manifest_sha256": file_sha256(source.manifest),
                "audio_root": str(source.audio_root.resolve()),
                "rows": len(source.rows),
                "weight": source.weight,
            }
            for source in train_sources
        ],
        "validation_sources": [
            {
                "name": source.name,
                "manifest": str(source.manifest.resolve()),
                "manifest_sha256": file_sha256(source.manifest),
                "audio_root": str(source.audio_root.resolve()),
                "rows": len(source.rows),
                "baseline_mIoU": source.baseline_miou,
            }
            for source in validation_sources
        ],
        "loss_mean": {
            name: value / max(1, total_updates * args.gradient_accumulation)
            for name, value in loss_sums.items()
        },
        "best_selection_score": best_score,
        "best_update": best_update,
        "validation_history": validation_history,
        "trainable_head_parameters": sum(p.numel() for p in head.parameters()),
        "trainable_adapter_parameters": sum(p.numel() for p in adapter_parameters),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "layer_weights": (
            torch.softmax(head.layer_logits.detach(), dim=0).cpu().tolist()
            if head.layer_logits is not None else [1.0]
        ),
        "git": git_state(project_root),
    }
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
