"""Counterfactual audio interventions and verifier features for NOVA."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from interval_metrics import normalize_intervals


def interval_gain(
    sample_count: int,
    intervals: Sequence[Sequence[float]],
    sample_rate: int = 16000,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """Return a soft union mask for intervals, with short ramps to avoid clicks."""
    gain = np.zeros(sample_count, dtype=np.float32)
    fade = max(0, int(round(sample_rate * fade_ms / 1000.0)))
    duration = sample_count / float(sample_rate)
    for start, end in normalize_intervals(intervals, duration):
        left = max(0, min(sample_count, int(round(start * sample_rate))))
        right = max(left, min(sample_count, int(round(end * sample_rate))))
        if right <= left:
            continue
        local = np.ones(right - left, dtype=np.float32)
        ramp = min(fade, len(local) // 2)
        if ramp:
            phase = np.linspace(0.0, np.pi / 2.0, ramp, endpoint=False, dtype=np.float32)
            local[:ramp] = np.sin(phase) ** 2
            local[-ramp:] = local[:ramp][::-1]
        gain[left:right] = np.maximum(gain[left:right], local)
    return gain


def intervene_audio(
    wave: np.ndarray,
    intervals: Sequence[Sequence[float]],
    mode: str,
    sample_rate: int = 16000,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """Keep or drop an interval union while preserving the original timeline."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1:
        raise ValueError("NOVA expects a mono waveform")
    gain = interval_gain(len(wave), intervals, sample_rate=sample_rate, fade_ms=fade_ms)
    if mode == "keep":
        return wave * gain
    if mode == "drop":
        return wave * (1.0 - gain)
    raise ValueError(f"Unknown intervention mode: {mode}")


def prepare_detection_example(processor, wave: np.ndarray, query: str, answer: str):
    """Build a supervised Yes./No. example for sequence-likelihood scoring."""
    from spotsound import DETECTION_PROMPT

    user_content = [
        {"type": "audio", "audio": np.asarray(wave, dtype=np.float32)},
        {"type": "text", "text": DETECTION_PROMPT + query + " Answer: "},
    ]
    prompt = [{"role": "user", "content": user_content}]
    full = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    prompt_inputs = processor.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    inputs = processor.apply_chat_template(
        full, tokenize=True, add_generation_prompt=False, return_dict=True
    )
    labels = inputs["input_ids"].clone()
    labels[:, : int(prompt_inputs["input_ids"].shape[1])] = -100
    inputs["labels"] = labels
    return {
        key: value.cpu() if hasattr(value, "cpu") else value
        for key, value in inputs.items()
    }


def prepare_detection_prompt(processor, wave: np.ndarray, query: str):
    """Build the prompt-only form used for one-pass Yes/No logit scoring."""
    from spotsound import DETECTION_PROMPT

    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": np.asarray(wave, dtype=np.float32)},
                {"type": "text", "text": DETECTION_PROMPT + query + " Answer: "},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=True
    )
    return {
        key: value.cpu() if hasattr(value, "cpu") else value
        for key, value in inputs.items()
    }


def binary_log_odds(model, processor, wave: np.ndarray, query: str, sequence_score) -> float:
    """Return normalized log p(Yes.) - log p(No.) under the verifier model."""
    yes = prepare_detection_example(processor, wave, query, "Yes.")
    no = prepare_detection_example(processor, wave, query, "No.")
    with __import__("torch").no_grad():
        yes_score = float(sequence_score(model, yes).detach())
        no_score = float(sequence_score(model, no).detach())
    return yes_score - no_score


def binary_first_token_log_odds(model, processor, wave: np.ndarray, query: str) -> float:
    """Return the Yes-vs-No next-token logit difference in one forward pass."""
    import torch

    inputs = prepare_detection_prompt(processor, wave, query)
    batch = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            batch[key] = value
            continue
        value = value.to("cuda", non_blocking=True)
        if value.is_floating_point():
            value = value.to(model.dtype)
        batch[key] = value
    yes_id = processor.tokenizer.encode("Yes.", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No.", add_special_tokens=False)[0]
    with torch.no_grad():
        output = model(**batch, use_cache=False)
    position = int(batch["attention_mask"][0].sum()) - 1
    logits = output.logits[0, position]
    return float((logits[yes_id] - logits[no_id]).float().detach())


def counterfactual_features(
    model,
    processor,
    wave: np.ndarray,
    query: str,
    candidate: Sequence[Sequence[float]],
    sequence_score,
    sample_rate: int = 16000,
    fade_ms: float = 10.0,
) -> dict[str, float]:
    """Score component sufficiency and complement necessity for an interval set."""
    duration = len(wave) / float(sample_rate)
    intervals = normalize_intervals(candidate, duration)
    if not intervals:
        return {
            "component_mean": float("-inf"),
            "component_min": float("-inf"),
            "complement_no": -binary_log_odds(
                model, processor, wave, query, sequence_score
            ),
            "mean_plus_complement": float("-inf"),
            "min_plus_complement": float("-inf"),
            "duration_fraction": 0.0,
            "interval_count": 0.0,
        }

    component_scores = []
    for interval in intervals:
        kept = intervene_audio(
            wave, [interval], "keep", sample_rate=sample_rate, fade_ms=fade_ms
        )
        component_scores.append(
            binary_log_odds(model, processor, kept, query, sequence_score)
        )
    dropped = intervene_audio(
        wave, intervals, "drop", sample_rate=sample_rate, fade_ms=fade_ms
    )
    complement_no = -binary_log_odds(model, processor, dropped, query, sequence_score)
    component_mean = float(np.mean(component_scores))
    component_min = float(np.min(component_scores))
    covered = sum(end - start for start, end in intervals)
    return {
        "component_mean": component_mean,
        "component_min": component_min,
        "complement_no": complement_no,
        "mean_plus_complement": component_mean + complement_no,
        "min_plus_complement": component_min + complement_no,
        "duration_fraction": covered / duration if duration else 0.0,
        "interval_count": float(len(intervals)),
    }


def counterfactual_features_fast(
    model,
    processor,
    wave: np.ndarray,
    query: str,
    candidate: Sequence[Sequence[float]],
    sample_rate: int = 16000,
    fade_ms: float = 10.0,
) -> dict[str, float]:
    """One-pass-per-intervention variant used for full candidate extraction."""
    duration = len(wave) / float(sample_rate)
    intervals = normalize_intervals(candidate, duration)
    if not intervals:
        full_score = binary_first_token_log_odds(model, processor, wave, query)
        return {
            "component_mean": -20.0,
            "component_min": -20.0,
            "complement_no": -full_score,
            "mean_plus_complement": -20.0 - full_score,
            "min_plus_complement": -20.0 - full_score,
            "duration_fraction": 0.0,
            "interval_count": 0.0,
        }
    component_scores = [
        binary_first_token_log_odds(
            model,
            processor,
            intervene_audio(
                wave, [interval], "keep", sample_rate=sample_rate, fade_ms=fade_ms
            ),
            query,
        )
        for interval in intervals
    ]
    dropped = intervene_audio(
        wave, intervals, "drop", sample_rate=sample_rate, fade_ms=fade_ms
    )
    complement_no = -binary_first_token_log_odds(model, processor, dropped, query)
    component_mean = float(np.mean(component_scores))
    component_min = float(np.min(component_scores))
    covered = sum(end - start for start, end in intervals)
    return {
        "component_mean": component_mean,
        "component_min": component_min,
        "complement_no": complement_no,
        "mean_plus_complement": component_mean + complement_no,
        "min_plus_complement": component_min + complement_no,
        "duration_fraction": covered / duration if duration else 0.0,
        "interval_count": float(len(intervals)),
    }
