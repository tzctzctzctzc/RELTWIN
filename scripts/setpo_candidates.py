#!/usr/bin/env python3
"""Deterministic candidate construction for RelTwin-SetPO."""

from __future__ import annotations

from collections.abc import Sequence

from interval_metrics import (
    Interval,
    merge_intervals,
    normalize_intervals,
    setpo_quality,
    setpo_quality_axes,
)


RELATION_EXCHANGE_PERMUTATION = (1, 0, 3, 2, 4, 5)


def jitter_intervals(
    intervals: Sequence[Interval], duration: float, ratio: float
) -> list[Interval]:
    expanded = []
    for start, end in intervals:
        delta = max(0.05, (end - start) * ratio)
        expanded.append((max(0.0, start - delta), min(duration, end + delta)))
    return normalize_intervals(expanded, duration)


def merge_to_span(intervals: Sequence[Interval]) -> list[Interval]:
    if not intervals:
        return []
    return [(min(start for start, _ in intervals), max(end for _, end in intervals))]


def add_false_interval(intervals: Sequence[Interval], duration: float) -> list[Interval]:
    merged = merge_intervals(intervals)
    gaps = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor > 0.1:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor > 0.1:
        gaps.append((cursor, duration))
    if not gaps:
        return list(intervals)
    gap_start, gap_end = max(gaps, key=lambda item: item[1] - item[0])
    average_width = (
        sum(end - start for start, end in intervals) / len(intervals) if intervals else 1.0
    )
    false_width = min(average_width, 0.8 * (gap_end - gap_start))
    false_width = max(0.05, false_width)
    center = (gap_start + gap_end) / 2
    false_interval = (center - false_width / 2, center + false_width / 2)
    return sorted(normalize_intervals([*intervals, false_interval], duration))


def relation_candidates(
    window_ab: Sequence[Sequence[float]],
    window_ba: Sequence[Sequence[float]],
    duration: float,
    jitter_ratio: float,
) -> list[list[Interval]]:
    ab = normalize_intervals(window_ab, duration)
    ba = normalize_intervals(window_ba, duration)
    both = sorted(normalize_intervals([*ab, *ba], duration))
    candidates = [
        ab,
        ba,
        jitter_intervals(ab, duration, jitter_ratio),
        jitter_intervals(ba, duration, jitter_ratio),
        merge_to_span(both),
        both,
    ]
    if len(candidates) != len(RELATION_EXCHANGE_PERMUTATION):
        raise AssertionError("Relation candidate/permutation size mismatch")
    return candidates


def ordinary_candidates(
    ground_truth: Sequence[Sequence[float]],
    duration: float,
    jitter_ratio: float,
) -> list[list[Interval]]:
    exact = sorted(normalize_intervals(ground_truth, duration))
    dropped = exact[:-1] if len(exact) > 1 else []
    return [
        exact,
        jitter_intervals(exact, duration, jitter_ratio),
        dropped,
        merge_to_span(exact),
        add_false_interval(exact, duration),
        [(0.0, duration)] if duration > 1e-3 else [],
    ]


def adjust_boundaries(
    intervals: Sequence[Interval],
    duration: float,
    start_ratio: float,
    end_ratio: float,
) -> list[Interval]:
    adjusted = []
    for start, end in intervals:
        width = end - start
        adjusted.append((start + start_ratio * width, end + end_ratio * width))
    return normalize_intervals(adjusted, duration)


def split_longest_interval(
    intervals: Sequence[Interval], duration: float, gap_ratio: float = 0.1
) -> list[Interval]:
    if not intervals:
        return []
    index = max(range(len(intervals)), key=lambda item: intervals[item][1] - intervals[item][0])
    start, end = intervals[index]
    midpoint = (start + end) / 2
    half_gap = max(0.025, (end - start) * gap_ratio / 2)
    split = [*intervals[:index], (start, midpoint - half_gap), (midpoint + half_gap, end)]
    split.extend(intervals[index + 1 :])
    return sorted(normalize_intervals(split, duration))


def _unique_candidates(candidates: Sequence[Sequence[Interval]]) -> list[list[Interval]]:
    unique = []
    seen = set()
    for candidate in candidates:
        key = tuple((round(start, 6), round(end, 6)) for start, end in candidate)
        if key not in seen:
            seen.add(key)
            unique.append(list(candidate))
    return unique


def scale_cardinality_candidates(
    ground_truth: Sequence[Sequence[float]],
    duration: float,
    jitter_ratio: float,
) -> list[list[Interval]]:
    """Candidates spanning boundary direction, temporal scale, and event count errors."""
    exact = sorted(normalize_intervals(ground_truth, duration))
    dropped = exact[:-1] if len(exact) > 1 else []
    candidates = [
        exact,
        adjust_boundaries(exact, duration, -jitter_ratio, jitter_ratio),
        adjust_boundaries(exact, duration, jitter_ratio, -jitter_ratio),
        adjust_boundaries(exact, duration, -jitter_ratio, 0.0),
        adjust_boundaries(exact, duration, jitter_ratio, 0.0),
        adjust_boundaries(exact, duration, 0.0, -jitter_ratio),
        adjust_boundaries(exact, duration, 0.0, jitter_ratio),
        dropped,
        merge_to_span(exact),
        add_false_interval(exact, duration),
        split_longest_interval(exact, duration),
        [(0.0, duration)] if duration > 1e-3 else [],
    ]
    return _unique_candidates(candidates)


def candidate_qualities(
    ground_truth: Sequence[Sequence[float]], candidates: Sequence[Sequence[Interval]]
) -> list[float]:
    normalized_gt = normalize_intervals(ground_truth)
    return [setpo_quality(normalized_gt, list(candidate)) for candidate in candidates]


def candidate_quality_axes(
    ground_truth: Sequence[Sequence[float]],
    candidates: Sequence[Sequence[Interval]],
    duration: float,
) -> list[tuple[float, float, float, float]]:
    normalized_gt = normalize_intervals(ground_truth, duration)
    return [setpo_quality_axes(normalized_gt, list(candidate), duration) for candidate in candidates]
