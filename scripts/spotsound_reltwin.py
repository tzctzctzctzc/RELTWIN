"""Shared utilities for full SpotSound-A RelTwin adaptation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


BRIDGE_COMPONENTS = {"audio_projection", "audio_layer_norm"}


def group_twin_rows(rows: list[dict]) -> tuple[list[dict[str, dict]], str]:
    relation_schema = [
        all(key in row for key in ("pair_id", "template", "relation")) for row in rows
    ]
    if any(relation_schema) and not all(relation_schema):
        raise ValueError("Twin manifest mixes relation and ordinary schemas")
    grouped = defaultdict(dict)
    if all(relation_schema):
        for row in rows:
            relation = row["relation"]
            if relation not in {"AB", "BA"}:
                raise ValueError(f"Unsupported relation label: {relation}")
            key = (row["pair_id"], row["template"], row.get("variant", 0))
            if relation in grouped[key]:
                raise ValueError(f"Duplicate {relation} row in twin group {key}")
            grouped[key][relation] = row
        schema = "relation_query"
    else:
        by_audio = defaultdict(list)
        for row in rows:
            if not all(key in row for key in ("audio_path", "caption", "annotations")):
                raise ValueError("Ordinary twin rows require audio_path, caption, annotations")
            by_audio[str(Path(row["audio_path"]))].append(row)
        for audio_path, pair in by_audio.items():
            if len(pair) != 2:
                raise ValueError(f"Ordinary twin audio {audio_path} has {len(pair)} queries")
            pair = sorted(
                pair,
                key=lambda row: (str(row["caption"]), json.dumps(row["annotations"])),
            )
            if pair[0]["caption"] == pair[1]["caption"]:
                raise ValueError(f"Ordinary twin queries must differ: {audio_path}")
            grouped[(audio_path,)] = {"AB": pair[0], "BA": pair[1]}
        schema = "ordinary_event_query"
    groups = [value for _, value in sorted(grouped.items())]
    if not groups or any(set(group) != {"AB", "BA"} for group in groups):
        raise ValueError("Every twin group must contain exactly AB and BA")
    return groups, schema


def weighted_schedule(sizes, weights, steps, rng):
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_component(name: str) -> str | None:
    if "lora_" in name and ".language_model." in name:
        return "language_lora"
    if ".multi_modal_projector." in name:
        return "audio_projection"
    if ".audio_tower.layer_norm." in name:
        return "audio_layer_norm"
    return None


def configure_spotsound_parameters(
    model, scope: str = "full"
) -> tuple[list[torch.nn.Parameter], dict]:
    if scope not in {"lora", "full"}:
        raise ValueError(f"Unsupported SpotSound RelTwin scope: {scope}")
    trainable = []
    counts: dict[str, int] = {}
    names = []
    for name, parameter in model.named_parameters():
        component = parameter_component(name)
        enabled = component == "language_lora" or (
            scope == "full" and component in BRIDGE_COMPONENTS
        )
        parameter.requires_grad = enabled
        if enabled:
            trainable.append(parameter)
            names.append(name)
            counts[component] = counts.get(component, 0) + parameter.numel()
    if not counts.get("language_lora"):
        raise RuntimeError("No SpotSound language-model LoRA parameters found")
    if scope == "full":
        missing = sorted(BRIDGE_COMPONENTS - counts.keys())
        if missing:
            raise RuntimeError(f"Missing SpotSound bridge components: {missing}")
    report = {
        "scope": scope,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "component_parameters": counts,
        "trainable_tensors": len(trainable),
        "trainable_names": names,
    }
    return trainable, report


def set_spotsound_training_mode(model, scope: str = "full") -> None:
    if scope not in {"lora", "full"}:
        raise ValueError(f"Unsupported SpotSound RelTwin scope: {scope}")
    # Transformers activates gradient checkpointing only while the owning
    # modules are in training mode. Exact RNG replay in the outer objectives
    # makes the two dropout-bearing forward passes numerically consistent.
    model.train()


def capture_rng_state() -> dict:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_bridge_delta(model, output: Path, metadata: dict) -> None:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter_component(name) in BRIDGE_COMPONENTS
    }
    if not state:
        raise RuntimeError("No SpotSound bridge parameters selected for saving")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "metadata": metadata}, output)


def load_bridge_delta(model, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    available = dict(model.named_parameters())
    unknown = sorted(set(state) - set(available))
    mismatched = sorted(
        name
        for name, value in state.items()
        if name in available and tuple(value.shape) != tuple(available[name].shape)
    )
    invalid = sorted(
        name for name in state if parameter_component(name) not in BRIDGE_COMPONENTS
    )
    if unknown or mismatched or invalid:
        raise RuntimeError(
            f"Invalid SpotSound bridge delta: unknown={unknown[:5]}, "
            f"mismatched={mismatched[:5]}, invalid={invalid[:5]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected SpotSound bridge keys: {incompatible.unexpected_keys[:5]}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "loaded_keys": len(state),
        "loaded_parameters": sum(value.numel() for value in state.values()),
        "metadata": payload.get("metadata", {}) if isinstance(payload, dict) else {},
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
