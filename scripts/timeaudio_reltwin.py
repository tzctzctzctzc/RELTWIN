"""Shared TimeAudio adapters for matched SFT/RelTwin experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interval_metrics import normalize_intervals, parse_canonical_intervals


TIMEAUDIO_TAG_PROMPT = (
    "<Speech><SpeechHere></Speech> What are the start and end times of audio "
    "matching '{query}'?"
)
_CHECKPOINT_ARCHITECTURE_KEYS = (
    "use_speech_Qformer",
    "window_level_Qformer",
    "num_speech_query_token",
    "second_per_window",
    "second_stride",
    "lora",
    "lora_rank",
    "lora_alpha",
    "added_time_token",
    "use_token_merge",
)
_RAW_PAIR = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:s|seconds)?\s*(?:-|–|—|to)\s*"
    r"(-?\d+(?:\.\d+)?)\s*(?:s|seconds)?",
    re.I,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timeaudio_question(query: str) -> str:
    return TIMEAUDIO_TAG_PROMPT.format(query=query)


def timeaudio_answer(intervals) -> str:
    pairs = []
    for start, end in intervals:
        start, end = round(float(start), 1), round(float(end), 1)
        if end > start:
            pairs.append(f"{start:.1f} - {end:.1f}")
    if not pairs:
        raise ValueError("TimeAudio training answer requires a non-empty interval")
    return ", ".join(pairs)


def format_timeaudio_v3(raw: str) -> str:
    """Match TimeAudio's v3 anchor/offset token formatting."""

    def encode(value: str) -> str:
        rounded = str(round(float(value), 1))
        integer, fraction = rounded.split(".")
        if integer.startswith("-"):
            raise ValueError("Negative timestamps are not valid training targets")
        return "".join(f"<a_{digit}>" for digit in integer) + "".join(
            f"<f_{digit}>" for digit in fraction
        )

    return _RAW_PAIR.sub(lambda match: f"{encode(match.group(1))} - {encode(match.group(2))}", raw)


def parse_timeaudio_intervals(text: str, duration: float | None = None):
    pairs = _RAW_PAIR.findall(text)
    if pairs:
        return normalize_intervals(pairs, duration)
    return parse_canonical_intervals(text, duration)


def checkpoint_architecture(payload: dict) -> dict:
    """Recover architecture-critical values saved with the public checkpoint."""
    model_config = payload.get("config", {}).get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("TimeAudio checkpoint does not contain a model config")
    missing = [key for key in _CHECKPOINT_ARCHITECTURE_KEYS if key not in model_config]
    if missing:
        raise ValueError(f"TimeAudio checkpoint is missing architecture keys: {missing}")
    return {key: model_config[key] for key in _CHECKPOINT_ARCHITECTURE_KEYS}


def resolve_audio_path(audio_dir: Path | None, row: dict) -> Path:
    raw = Path(row["audio_path"])
    candidates = [raw]
    if audio_dir is not None:
        candidates.extend((audio_dir / raw, audio_dir / raw.name))
    for candidate in list(candidates):
        candidates.extend(candidate.with_suffix(ext) for ext in (".wav", ".flac", ".mp3"))
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def prepare_audio_features(audio_path: Path, feature_extractor, max_len: int = 180):
    import librosa

    wave, _ = librosa.load(audio_path, sr=16000, mono=True)
    wave = np.asarray(wave, dtype=np.float32)
    if wave.size < 16000:
        wave = np.pad(wave, (0, 16000 - wave.size))
    wave = wave[: 16000 * max_len]
    duration = wave.size / 16000.0

    if wave.size <= 16000 * 30:
        spectrogram = feature_extractor(
            wave, sampling_rate=16000, return_tensors="pt"
        )["input_features"]
    else:
        chunks = []
        for start in range(0, wave.size, 16000 * 30):
            chunk = wave[start : start + 16000 * 30]
            if chunk.size < 16000 * 5:
                continue
            chunk = np.pad(chunk, (0, 16000 * 30 - chunk.size))
            chunks.append(
                feature_extractor(chunk, sampling_rate=16000, return_tensors="pt")[
                    "input_features"
                ].squeeze(0)
            )
        if not chunks:
            raise ValueError(f"No valid TimeAudio chunks for {audio_path}")
        spectrogram = torch.stack(chunks).unsqueeze(0)

    raw_wav = torch.from_numpy(wave).unsqueeze(0)
    return {
        "spectrogram": spectrogram,
        "raw_wav": raw_wav,
        "padding_mask": torch.zeros_like(raw_wav, dtype=torch.bool),
        "duration": torch.tensor([duration], dtype=torch.float32),
        "audio_path": str(audio_path),
    }


def build_timeaudio_sample(features: dict, query: str, answer: str = "") -> dict:
    return {
        "spectrogram": features["spectrogram"],
        "raw_wav": features["raw_wav"],
        "padding_mask": features["padding_mask"],
        "duration": features["duration"],
        "text": [format_timeaudio_v3(answer) if answer else ""],
        "task": ["tag"],
        "Q": [timeaudio_question(query)],
        "id": [features["audio_path"]],
        "id_audio": [Path(features["audio_path"]).name],
    }


def move_sample(sample: dict, device: str = "cuda") -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in sample.items()
    }


def rbee_objective(
    scores: torch.Tensor,
    temperature: float = 1.0,
    method_weight: float = 1.0,
    exchange_weight: float = 1.0,
):
    if scores.shape != (4,):
        raise ValueError("RelTwin score order must contain exactly four candidates")
    sft = -(scores[0] + scores[3]) / 2
    if method_weight == 0:
        return sft, {"sft": float(sft.detach())}
    logits_ab, logits_ba = scores[:2] / temperature, scores[2:] / temperature
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
    total = sft + method_weight * (candidate + exchange_weight * js)
    return total, {
        "sft": float(sft.detach()),
        "candidate_ce": float(candidate.detach()),
        "exchange_js": float(js.detach()),
    }


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_timeaudio_model(
    timeaudio_repo: Path,
    config_path: Path,
    llama_path: Path,
    whisper_path: Path,
    beats_path: Path,
    bert_path: Path,
    checkpoint: Path,
    delta: Path | None = None,
):
    """Load the public checkpoint and reject silent checkpoint-key loss."""
    from omegaconf import OmegaConf

    timeaudio_repo = timeaudio_repo.resolve()
    if str(timeaudio_repo) not in sys.path:
        sys.path.insert(0, str(timeaudio_repo))
    from models import load_model

    bert_target = timeaudio_repo / "pretrained_model" / "bert-base-uncased"
    if bert_target.resolve() != bert_path.resolve():
        if bert_target.exists() or bert_target.is_symlink():
            if bert_target.resolve() != bert_path.resolve():
                raise ValueError(f"BERT path conflict: {bert_target}")
        else:
            bert_target.parent.mkdir(parents=True, exist_ok=True)
            bert_target.symlink_to(bert_path.resolve(), target_is_directory=True)

    official_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    architecture = checkpoint_architecture(official_payload)
    cfg = OmegaConf.load(config_path.resolve())
    for key, value in architecture.items():
        cfg.model[key] = value
    cfg.model.llama_path = str(llama_path.resolve())
    cfg.model.whisper_path = str(whisper_path.resolve())
    cfg.model.beats_path = str(beats_path.resolve())
    cfg.model.prompt_path = str((timeaudio_repo / "prompts" / "train_prompt.json").resolve())
    cfg.model.test_prompt_path = str((timeaudio_repo / "prompts" / "test_prompt.json").resolve())
    cfg.model.ckpt = ""
    with working_directory(timeaudio_repo):
        model = load_model(cfg.model)

    reports = []
    for label, path in (("official", checkpoint), ("delta", delta)):
        if path is None:
            continue
        payload = official_payload if label == "official" else torch.load(
            path, map_location="cpu", weights_only=True
        )
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"{label} checkpoint has unexpected keys: {incompatible.unexpected_keys[:10]}")
        reports.append(
            {
                "label": label,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "loaded_keys": len(state),
                "missing_keys": len(incompatible.missing_keys),
                "unexpected_keys": 0,
                "checkpoint_architecture": architecture if label == "official" else None,
            }
        )
    return model, cfg, reports


def freeze_to_existing_lora(model) -> list[torch.nn.Parameter]:
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name and "llama_model" in name
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("No existing TimeAudio language-model LoRA parameters found")
    return trainable


def set_lora_training_mode(model) -> None:
    """Keep frozen audio modules deterministic while enabling LoRA dropout."""
    model.eval()
    for name, module in model.named_modules():
        if "lora_dropout" in name:
            module.train()


def save_trainable_delta(model, output: Path, metadata: dict) -> None:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "metadata": metadata}, output)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
