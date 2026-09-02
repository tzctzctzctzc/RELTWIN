"""Shared model and data helpers for SpanTool training and evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from spantool import SPANTOOL_PROMPT, SpanToolConfig, SpanToolHead


def load_manifest(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must contain a list: {path}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    """Hash a checkpoint file or directory without depending on its location."""
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Checkpoint directory contains no files: {path}")
    for candidate in files:
        digest.update(candidate.relative_to(path).as_posix().encode())
        digest.update(file_sha256(candidate).encode())
    return digest.hexdigest()


def resolve_audio_path(audio_root: Path, row: dict) -> Path:
    relative = Path(row["audio_path"])
    candidates = [relative] if relative.is_absolute() else [audio_root / relative]
    candidates.append(audio_root / relative.name)
    for base in list(candidates):
        candidates.extend(base.with_suffix(suffix) for suffix in (".wav", ".flac", ".mp3"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Audio not found for {row['audio_path']} under {audio_root}")


def load_wave(audio_root: Path, row: dict) -> np.ndarray:
    import librosa

    path = resolve_audio_path(audio_root, row)
    wave, _ = librosa.load(path, sr=16000, mono=True)
    return np.asarray(wave, dtype=np.float32)


def prepare_spantool_input(processor, wave: np.ndarray, query: str) -> dict:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SPANTOOL_PROMPT + query + "\nAudio: "},
                {"type": "audio", "audio": wave},
            ],
        }
    ]
    return processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )


def move_model_inputs(inputs: dict, device: torch.device, dtype: torch.dtype) -> dict:
    moved = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype, non_blocking=True)
        else:
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


def extract_audio_states(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    audio_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract and right-pad audio-token states from an LM batch."""
    sequences = [
        hidden_states[index][input_ids[index] == audio_token_id]
        for index in range(hidden_states.shape[0])
    ]
    if any(sequence.shape[0] == 0 for sequence in sequences):
        raise ValueError("The processed prompt contains no audio tokens")
    padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], device=padded.device)
    mask = torch.arange(padded.shape[1], device=padded.device).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask


def infer_hidden_dim(model) -> int:
    for config_name in ("text_config", "language_config"):
        nested = getattr(model.config, config_name, None)
        if nested is not None and getattr(nested, "hidden_size", None):
            return int(nested.hidden_size)
    language_model = getattr(model, "language_model", None)
    if language_model is not None and getattr(language_model.config, "hidden_size", None):
        return int(language_model.config.hidden_size)
    if getattr(model.config, "hidden_size", None):
        return int(model.config.hidden_size)
    raise ValueError("Could not infer language-model hidden size")


def select_hidden_state(outputs, layer: int) -> torch.Tensor:
    if outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden states")
    if not -len(outputs.hidden_states) <= layer < len(outputs.hidden_states):
        raise ValueError(f"Audio layer {layer} outside {len(outputs.hidden_states)} hidden states")
    return outputs.hidden_states[layer]


def select_audio_states(
    outputs,
    layers: Sequence[int],
    input_ids: torch.Tensor,
    audio_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one layer or stack aligned audio states for learned fusion."""
    if not layers:
        raise ValueError("At least one audio layer is required")
    extracted = [
        extract_audio_states(
            select_hidden_state(outputs, layer), input_ids, audio_token_id
        )
        for layer in layers
    ]
    reference_mask = extracted[0][1]
    if any(not torch.equal(mask, reference_mask) for _, mask in extracted[1:]):
        raise RuntimeError("Audio-token masks differ across hidden layers")
    if len(extracted) == 1:
        return extracted[0]
    return torch.stack([states for states, _ in extracted], dim=1), reference_mask


def save_spantool_checkpoint(
    output_dir: Path,
    head: SpanToolHead,
    *,
    audio_layers: Sequence[int] | None = None,
    audio_layer: int | None = None,
    metadata: dict | None = None,
) -> None:
    selected_layers = list(audio_layers) if audio_layers is not None else [audio_layer]
    if not selected_layers or selected_layers[0] is None:
        raise ValueError("At least one audio layer is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), output_dir / "spantool_head.pt")
    payload = {
        "format_version": 1,
        "audio_layer": int(selected_layers[0]),
        "audio_layers": [int(layer) for layer in selected_layers],
        "head_config": head.config.to_dict(),
        "metadata": metadata or {},
    }
    (output_dir / "spantool_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_spantool_checkpoint(
    checkpoint_dir: Path, device: torch.device
) -> tuple[SpanToolHead, dict]:
    payload = json.loads(
        (checkpoint_dir / "spantool_config.json").read_text(encoding="utf-8")
    )
    head = SpanToolHead(SpanToolConfig(**payload["head_config"]))
    state = torch.load(
        checkpoint_dir / "spantool_head.pt", map_location=device, weights_only=True
    )
    head.load_state_dict(state)
    return head.to(device), payload


def validate_rows(rows: Sequence[dict], source_name: str) -> None:
    required = {"audio_path", "caption", "annotations"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{source_name} row {index} missing {sorted(missing)}")
