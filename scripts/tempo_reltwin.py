"""TEMPO model helpers used by the RelTwin post-training experiments."""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from transformers import AudioFlamingo3ForConditionalGeneration


TIMESTAMP_DIM = 64
FRAME_SECONDS = 0.04
MIN_PERIOD_SECONDS = 0.08
MAX_PERIOD_SECONDS = 60.0
GROUNDING_TAG = "[audio:ground]"
_ATOMIC_INTERVAL = re.compile(
    r"<\|\s*(\d+(?:\.\d+)?)\s*\|>\s*to\s*<\|\s*(\d+(?:\.\d+)?)\s*\|>",
    re.I,
)


class TempoAudioFlamingo3ForConditionalGeneration(
    AudioFlamingo3ForConditionalGeneration
):
    """Audio Flamingo 3 with TEMPO's sinusoidal wall-clock projector input."""

    def __init__(self, config):
        super().__init__(config)
        self.time_proj = torch.nn.Linear(
            TIMESTAMP_DIM, config.audio_config.hidden_size, bias=True
        )

    def _time_encoding(self, length: int, device, dtype):
        half = TIMESTAMP_DIM // 2
        periods = torch.logspace(
            math.log10(MIN_PERIOD_SECONDS),
            math.log10(MAX_PERIOD_SECONDS),
            half,
            device=device,
            dtype=torch.float32,
        )
        times = torch.arange(length, device=device, dtype=torch.float32) * FRAME_SECONDS
        phases = 2.0 * math.pi * times[:, None] / periods[None, :]
        encoding = torch.cat((phases.sin(), phases.cos()), dim=-1)
        return encoding.to(dtype)

    def get_audio_features(self, input_features, input_features_mask, **kwargs):
        kwargs.pop("return_dict", None)
        audio_output = self.audio_tower(
            input_features,
            input_features_mask=input_features_mask,
            return_dict=True,
            **kwargs,
        )
        features = audio_output.last_hidden_state
        time_features = self.time_proj(
            self._time_encoding(features.shape[1], features.device, features.dtype)
        )
        audio_embeds = self.multi_modal_projector(features + time_features.unsqueeze(0))

        input_lengths = input_features_mask.sum(-1).to(torch.long)
        _, post_lengths = self.audio_tower._get_feat_extract_output_lengths(input_lengths)
        valid_mask = (
            torch.arange(audio_embeds.shape[1], device=post_lengths.device)[None, :]
            < post_lengths[:, None]
        )
        audio_output.pooler_output = audio_embeds[valid_mask.to(audio_embeds.device)]
        return audio_output


def load_time_projector(model, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    normalized = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("time_proj.")
        normalized[key] = value
    state = normalized
    model.time_proj.load_state_dict(state, strict=True)


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
