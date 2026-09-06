"""Pure helpers for deterministic long-audio inference."""

from __future__ import annotations

from interval_metrics import normalize_intervals


def chunk_bounds(
    num_samples: int,
    sample_rate: int,
    *,
    chunk_seconds: float,
    stride_seconds: float,
) -> list[tuple[int, int]]:
    if num_samples <= 0 or sample_rate <= 0:
        raise ValueError("audio and sample rate must be positive")
    chunk_samples = int(round(chunk_seconds * sample_rate))
    stride_samples = int(round(stride_seconds * sample_rate))
    if chunk_samples <= 0 or stride_samples <= 0 or stride_samples > chunk_samples:
        raise ValueError("require 0 < stride_seconds <= chunk_seconds")
    if num_samples <= chunk_samples:
        return [(0, num_samples)]
    starts = list(range(0, num_samples - chunk_samples + 1, stride_samples))
    final_start = num_samples - chunk_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, min(num_samples, start + chunk_samples)) for start in starts]


def offset_intervals(intervals, offset_seconds: float, duration: float):
    return normalize_intervals(
        [(start + offset_seconds, end + offset_seconds) for start, end in intervals],
        duration,
    )
