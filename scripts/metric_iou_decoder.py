"""Metric-aligned, incumbent-centred interval decoding.

The decoder treats calibrated occupancy probabilities as a piecewise-constant
posterior over time and maximises posterior expected temporal set IoU inside a
cardinality-preserving trust region.  It is deliberately independent of model
loading so the optimisation can be tested exhaustively on small grids.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


Interval = tuple[float, float]


@dataclass(frozen=True)
class DecodeResult:
    incumbent: list[Interval]
    candidate: list[Interval]
    selected: list[Interval]
    posterior_iou_incumbent: float
    posterior_iou_candidate: float
    gain_lower_quantile: float
    iterations: int
    converged: bool
    switch: bool
    abstain_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _Option:
    start: float
    end: float
    intersection: float
    length: float


def calibrate_occupancy(
    logits: Sequence[float] | np.ndarray, temperature: float, bias: float
) -> np.ndarray:
    """Apply scalar temperature/bias calibration without numerical overflow."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = (np.asarray(logits, dtype=np.float64) + float(bias)) / temperature
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def resample_probabilities(
    probabilities: Sequence[float] | np.ndarray, output_steps: int
) -> np.ndarray:
    """Linearly interpolate bin-centre probabilities to a finer uniform grid."""
    source = np.asarray(probabilities, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if output_steps < 1:
        raise ValueError("output_steps must be positive")
    if source.size == output_steps:
        return source.copy()
    old_x = (np.arange(source.size, dtype=np.float64) + 0.5) / source.size
    new_x = (np.arange(output_steps, dtype=np.float64) + 0.5) / output_steps
    return np.interp(new_x, old_x, source, left=source[0], right=source[-1])


def validate_intervals(
    intervals: Sequence[Sequence[float]], duration: float
) -> list[Interval]:
    """Validate rather than repair an incumbent, preserving its cardinality."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    result: list[Interval] = []
    for raw in intervals:
        if len(raw) < 2:
            raise ValueError("interval must contain start and end")
        start, end = float(raw[0]), float(raw[1])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError("interval boundaries must be finite")
        if start < 0 or end > duration or end <= start:
            raise ValueError("interval is outside the audio or has non-positive length")
        if result and start <= result[-1][1]:
            raise ValueError("incumbent intervals must be strictly ordered and disjoint")
        result.append((start, end))
    return result


def _posterior_cdf(probabilities: np.ndarray, duration: float, value: float) -> float:
    """Integral from zero to ``value`` for a uniform piecewise-constant grid."""
    bins = probabilities.size
    step = duration / bins
    clipped = min(duration, max(0.0, value))
    complete = min(bins, int(math.floor(clipped / step + 1e-12)))
    prefix = float(probabilities[:complete].sum() * step)
    if complete < bins:
        prefix += float((clipped - complete * step) * probabilities[complete])
    return prefix


def _integral(
    probabilities: np.ndarray, duration: float, start: float, end: float
) -> float:
    """Integrate a uniform piecewise-constant posterior on [start, end]."""
    if end <= start:
        return 0.0
    return _posterior_cdf(probabilities, duration, end) - _posterior_cdf(
        probabilities, duration, start
    )


def _integral_lookup(probabilities: np.ndarray, duration: float):
    """Build an O(1) interval-integral closure for candidate enumeration."""
    bins = probabilities.size
    step = duration / bins
    prefix = np.concatenate(([0.0], np.cumsum(probabilities) * step))

    def cdf(value: float) -> float:
        clipped = min(duration, max(0.0, value))
        complete = min(bins, int(math.floor(clipped / step + 1e-12)))
        result = float(prefix[complete])
        if complete < bins:
            result += float((clipped - complete * step) * probabilities[complete])
        return result

    return lambda start, end: cdf(end) - cdf(start)


def posterior_set_iou(
    probabilities: Sequence[float] | np.ndarray,
    intervals: Sequence[Sequence[float]],
    duration: float,
) -> float:
    """Posterior expected set-IoU surrogate from calibrated occupancy."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0 or not np.isfinite(probs).all():
        raise ValueError("probabilities must be a finite, non-empty vector")
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    clean = validate_intervals(intervals, duration) if intervals else []
    coverage = _coverage_vector(clean, duration, probs.size)
    intersection = float(probs @ coverage)
    predicted_length = float(coverage.sum())
    expected_gt_length = float(probs.mean() * duration)
    denominator = expected_gt_length + predicted_length - intersection
    if denominator <= 0:
        return 1.0 if expected_gt_length == 0 and predicted_length == 0 else 0.0
    return float(intersection / denominator)


def _coverage_vector(intervals: Sequence[Interval], duration: float, bins: int) -> np.ndarray:
    step = duration / bins
    left = np.arange(bins, dtype=np.float64) * step
    right = left + step
    coverage = np.zeros(bins, dtype=np.float64)
    for start, end in intervals:
        coverage += np.maximum(0.0, np.minimum(end, right) - np.maximum(start, left))
    return np.minimum(coverage, step)


def _posterior_scores_batch(
    probabilities: np.ndarray,
    intervals: Sequence[Interval],
    duration: float,
) -> np.ndarray:
    """Vectorised posterior score for a [samples, bins] probability matrix."""
    coverage = _coverage_vector(intervals, duration, probabilities.shape[1])
    intersection = probabilities @ coverage
    predicted_length = float(coverage.sum())
    expected_gt = probabilities.mean(axis=1) * duration
    denominator = expected_gt + predicted_length - intersection
    return np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)


def _endpoint_choices(value: float, duration: float, bins: int, radius: float) -> list[float]:
    step = duration / bins
    low, high = max(0.0, value - radius), min(duration, value + radius)
    first = max(0, int(math.ceil(low / step - 1e-12)))
    last = min(bins, int(math.floor(high / step + 1e-12)))
    values = [index * step for index in range(first, last + 1)]
    values.append(value)  # Exact continuous incumbent boundary is always available.
    return sorted(set(round(item, 12) for item in values))


def _build_options(
    probabilities: np.ndarray,
    incumbent: Sequence[Interval],
    duration: float,
    radius_seconds: float,
) -> list[list[_Option]]:
    bins = probabilities.size
    minimum = duration / bins
    integrate = _integral_lookup(probabilities, duration)
    layers: list[list[_Option]] = []
    for original_start, original_end in incumbent:
        starts = _endpoint_choices(original_start, duration, bins, radius_seconds)
        ends = _endpoint_choices(original_end, duration, bins, radius_seconds)
        options = []
        for start in starts:
            for end in ends:
                if end - start + 1e-12 < minimum:
                    continue
                options.append(
                    _Option(
                        start=start,
                        end=end,
                        intersection=integrate(start, end),
                        length=end - start,
                    )
                )
        if not options:
            raise RuntimeError("trust region contains no valid interval")
        layers.append(sorted(options, key=lambda item: (item.end, item.start)))
    return layers


def _best_additive_set(
    layers: Sequence[Sequence[_Option]], ratio: float
) -> tuple[list[Interval], float, float]:
    """Maximise the Dinkelbach additive objective with a proposal-indexed DP."""
    scores = np.asarray(
        [(1.0 + ratio) * item.intersection - ratio * item.length for item in layers[0]],
        dtype=np.float64,
    )
    back_layers: list[np.ndarray] = []
    for layer_index in range(1, len(layers)):
        previous = layers[layer_index - 1]
        previous_ends = [item.end for item in previous]
        prefix_best = np.empty(len(previous), dtype=np.int64)
        best = 0
        for index in range(len(previous)):
            if scores[index] > scores[best] + 1e-15:
                best = index
            prefix_best[index] = best
        current_scores = np.full(len(layers[layer_index]), -np.inf, dtype=np.float64)
        current_back = np.full(len(layers[layer_index]), -1, dtype=np.int64)
        for index, item in enumerate(layers[layer_index]):
            stop = bisect.bisect_left(previous_ends, item.start - 1e-12) - 1
            if stop < 0:
                continue
            parent = int(prefix_best[stop])
            local = (1.0 + ratio) * item.intersection - ratio * item.length
            current_scores[index] = scores[parent] + local
            current_back[index] = parent
        if not np.isfinite(current_scores).any():
            raise RuntimeError("trust-region intervals cannot remain strictly disjoint")
        scores = current_scores
        back_layers.append(current_back)
    chosen = int(np.nanargmax(scores))
    option_indices = [chosen]
    for back in reversed(back_layers):
        chosen = int(back[chosen])
        option_indices.append(chosen)
    option_indices.reverse()
    selected_options = [layers[i][option_indices[i]] for i in range(len(layers))]
    intervals = [(item.start, item.end) for item in selected_options]
    return (
        intervals,
        float(sum(item.intersection for item in selected_options)),
        float(sum(item.length for item in selected_options)),
    )


def dinkelbach_trust_region_decode(
    probabilities: Sequence[float] | np.ndarray,
    incumbent_intervals: Sequence[Sequence[float]],
    duration: float,
    radius_seconds: float,
    *,
    decision_probabilities: Sequence[float] | np.ndarray | None = None,
    bootstrap_probabilities: Sequence[Sequence[float] | np.ndarray] = (),
    lower_quantile: float = 0.05,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> DecodeResult:
    """Decode and conservatively decide whether to replace the incumbent."""
    probs = np.asarray(probabilities, dtype=np.float64)
    decision_probs = (
        probs
        if decision_probabilities is None
        else np.asarray(decision_probabilities, dtype=np.float64)
    )
    if decision_probs.shape != probs.shape:
        raise ValueError("decision_probabilities must match probabilities")
    if radius_seconds < 0 or not math.isfinite(radius_seconds):
        raise ValueError("radius_seconds must be finite and non-negative")
    if not incumbent_intervals:
        return DecodeResult([], [], [], 0.0, 0.0, 0.0, 0, True, False, "empty_incumbent")
    try:
        incumbent = validate_intervals(incumbent_intervals, duration)
        optimisation_incumbent_score = posterior_set_iou(probs, incumbent, duration)
        incumbent_score = posterior_set_iou(decision_probs, incumbent, duration)
        layers = _build_options(probs, incumbent, duration, radius_seconds)
    except (ValueError, RuntimeError) as error:
        raw = [(float(item[0]), float(item[1])) for item in incumbent_intervals if len(item) >= 2]
        return DecodeResult(raw, raw, raw, 0.0, 0.0, 0.0, 0, False, False, f"invalid_incumbent:{error}")

    ratio = optimisation_incumbent_score
    candidate = incumbent
    candidate_score = incumbent_score
    converged = False
    iterations = 0
    expected_gt_length = float(probs.mean() * duration)
    for iterations in range(1, max_iterations + 1):
        candidate, intersection, predicted_length = _best_additive_set(layers, ratio)
        denominator = expected_gt_length + predicted_length - intersection
        optimisation_candidate_score = intersection / denominator if denominator > 0 else 0.0
        residual = intersection - ratio * denominator
        if abs(residual) <= tolerance or abs(optimisation_candidate_score - ratio) <= tolerance:
            converged = True
            break
        ratio = optimisation_candidate_score

    if not converged:
        return DecodeResult(
            incumbent, candidate, incumbent, incumbent_score, candidate_score,
            0.0, iterations, False, False, "not_converged",
        )
    candidate_score = posterior_set_iou(decision_probs, candidate, duration)
    raw_gain = candidate_score - incumbent_score
    if raw_gain <= tolerance:
        return DecodeResult(
            incumbent, candidate, incumbent, incumbent_score, candidate_score,
            raw_gain, iterations, True, False, "non_positive_posterior_gain",
        )
    if bootstrap_probabilities:
        samples = np.asarray(bootstrap_probabilities, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1:] != probs.shape:
            raise ValueError("bootstrap probabilities must match calibrated probabilities")
        gains = _posterior_scores_batch(samples, candidate, duration) - _posterior_scores_batch(
            samples, incumbent, duration
        )
        gain_lower = float(np.quantile(gains, lower_quantile))
    else:
        gain_lower = raw_gain
    if gain_lower <= 0:
        return DecodeResult(
            incumbent, candidate, incumbent, incumbent_score, candidate_score,
            gain_lower, iterations, True, False, "gain_lower_bound_not_positive",
        )
    return DecodeResult(
        incumbent, candidate, candidate, incumbent_score, candidate_score,
        gain_lower, iterations, True, True, None,
    )
