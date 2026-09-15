"""TEMPO model helpers used by the RelTwin post-training experiments."""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from transformers import AudioFlamingo3ForConditionalGeneration
from transformers.models.audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3MultiModalProjector,
)


TIMESTAMP_DIM = 64
FRAME_SECONDS = 0.04
MIN_PERIOD_SECONDS = 0.08
MAX_PERIOD_SECONDS = 60.0
GROUNDING_TAG = "[audio:ground]"
_ATOMIC_INTERVAL = re.compile(
    r"<\|\s*(\d+(?:\.\d+)?)\s*\|>\s*to\s*<\|\s*(\d+(?:\.\d+)?)\s*\|>",
    re.I,
)


class TempoAudioFlamingo3MultiModalProjector(AudioFlamingo3MultiModalProjector):
    """AF3 projector with TEMPO's learned wall-clock injection."""

    def __init__(self, config):
        super().__init__(config)
        self.time_proj = torch.nn.Linear(
            TIMESTAMP_DIM, config.audio_config.hidden_size, bias=False
        )
        frequencies = torch.logspace(
            math.log10(1.0 / MAX_PERIOD_SECONDS),
            math.log10(1.0 / MIN_PERIOD_SECONDS),
            TIMESTAMP_DIM // 2,
            dtype=torch.float32,
        )
        self.register_buffer("freqs", frequencies, persistent=True)

    def forward(self, audio_features):
        times = (
            torch.arange(
                audio_features.shape[1],
                device=audio_features.device,
                dtype=torch.float32,
            )
            * FRAME_SECONDS
        )
        phases = 2.0 * math.pi * times[:, None] * self.freqs.float()[None, :]
        encoding = torch.cat((phases.sin(), phases.cos()), dim=-1).to(
            audio_features.dtype
        )
        hidden_states = audio_features + self.time_proj(encoding).unsqueeze(0)
        return super().forward(hidden_states)


class TempoAudioFlamingo3ForConditionalGeneration(
    AudioFlamingo3ForConditionalGeneration
):
    """Audio Flamingo 3 with TEMPO's time-aware multi-modal projector."""

    def __init__(self, config):
        super().__init__(config)
        self.multi_modal_projector = TempoAudioFlamingo3MultiModalProjector(config)


def load_time_projector(model, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    normalized = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("time_proj.")
        normalized[key] = value
    state = normalized
    model.multi_modal_projector.time_proj.load_state_dict(state, strict=True)


def grounding_question(query: str) -> str:
    return (
        f"{GROUNDING_TAG} What is the time interval (start and end) for the query "
        f"'{query}' in the audio?"
    )


def grounding_conversation(wave, question: str, answer: str | None = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wave},
                {"type": "text", "text": question},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages


def atomic_answer(intervals) -> str:
    parts = []
    for start, end in intervals:
        start = min(60.0, max(0.0, round(float(start), 1)))
        end = min(60.0, max(0.0, round(float(end), 1)))
        if end > start:
            parts.append(f"{GROUNDING_TAG} <|{start:.1f}|> to <|{end:.1f}|>")
    return " ".join(parts)


def parse_atomic_intervals(text: str, duration: float | None = None):
    result = []
    for raw_start, raw_end in _ATOMIC_INTERVAL.findall(text):
        start, end = float(raw_start), float(raw_end)
        if duration is not None:
            start = min(duration, max(0.0, start))
            end = min(duration, max(0.0, end))
        if end > start:
            result.append((start, end))
    return result
