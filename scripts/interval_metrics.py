#!/usr/bin/env python3
"""Shared interval parsers and metrics for SpotSound and Auto-AEG protocols."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence


Interval = tuple[float, float]
THRESHOLDS = (0.3, 0.5, 0.7)

_SPOTSOUND_FROM_RE = re.compile(
    r"from\s*(-?\d+(?:\.\d+)?)\s*s?\s*(?:econds)?\s*to\s*"
    r"(-?\d+(?:\.\d+)?)\s*s?",
    re.I,
)
_SPOTSOUND_PAIR_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*s\s*(?:-|to|–)\s*"
    r"(-?\d+(?:\.\d+)?)\s*s",
    re.I,
)


def normalize_intervals(
    intervals: Iterable[Sequence[float]], duration: float | None = None
) -> list[Interval]:
    result: list[Interval] = []
    for item in intervals:
        if len(item) < 2:
            continue
        start, end = float(item[0]), float(item[1])
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        if end < start:
            start, end = end, start
        if duration is not None:
            start = max(0.0, min(start, duration))
            end = max(0.0, min(end, duration))
        if end - start > 1e-3:
            result.append((start, end))
    return result


def parse_spotsound_intervals(text: str, duration: float | None = None) -> list[Interval]:
    matches = _SPOTSOUND_FROM_RE.findall(text) or _SPOTSOUND_PAIR_RE.findall(text)
    return normalize_intervals(matches, duration)


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.I)
    return match.group(1).strip() if match else text.strip()


def parse_auto_aeg_intervals(text: str) -> list[Interval]:
    """Match the parser in AEGBench eval_benchmark_v3.py at commit 49a1d919."""
    answer = extract_answer(text)
    answer = re.sub(r"(\d+(?:\.\d+)?)s\b", r"\1", answer)
    for dash in ("–", "—", "‒"):
        answer = answer.replace(dash, ",")
    try:
        array = json.loads(answer)
        if isinstance(array, list):
            if len(array) == 2 and all(isinstance(x, (int, float)) for x in array):
                array = [array]
            result = []
            for item in array:
                if isinstance(item, dict) and "start" in item and "end" in item:
                    item = [item["start"], item["end"]]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    start, end = float(item[0]), float(item[1])
                    if end > start:
                        result.append((start, end))
            if result or not array:
                return result
    except Exception:
        pass
    pairs = re.findall(
        r"\[?\s*(\d+(?:\.\d+)?)\s*[,，]\s*"
        r"(\d+(?:\.\d+)?)\s*\]?",
        answer,
    )
    return [(float(start), float(end)) for start, end in pairs if float(end) > float(start)]


def parse_canonical_intervals(text: str, duration: float | None = None) -> list[Interval]:
    auto = parse_auto_aeg_intervals(text)
    if auto:
        return normalize_intervals(auto, duration)
    return parse_spotsound_intervals(text, duration)


def merge_intervals(intervals: Iterable[Sequence[float]]) -> list[Interval]:
    cleaned = sorted(normalize_intervals(intervals))
    merged: list[list[float]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def temporal_set_iou(
    ground_truth: Iterable[Sequence[float]], prediction: Iterable[Sequence[float]]
) -> float:
    ground_truth = merge_intervals(ground_truth)
    prediction = merge_intervals(prediction)
    if not ground_truth or not prediction:
        return 0.0
    intersection = 0.0
    gt_index = pred_index = 0
    while gt_index < len(ground_truth) and pred_index < len(prediction):
        intersection += max(
            0.0,
            min(ground_truth[gt_index][1], prediction[pred_index][1])
            - max(ground_truth[gt_index][0], prediction[pred_index][0]),
        )
        if ground_truth[gt_index][1] <= prediction[pred_index][1]:
            gt_index += 1
        else:
            pred_index += 1
    gt_length = sum(end - start for start, end in ground_truth)
    pred_length = sum(end - start for start, end in prediction)
    union = gt_length + pred_length - intersection
    return intersection / union if union > 0 else 0.0


def soft_precision_recall(
    ground_truth: Sequence[Interval], prediction: Sequence[Interval]
) -> tuple[float, float, float]:
    recall = (
        sum(max((interval_iou(gt, pred) for pred in prediction), default=0.0) for gt in ground_truth)
        / len(ground_truth)
        if ground_truth
        else 0.0
    )
    precision = (
        sum(max((interval_iou(pred, gt) for gt in ground_truth), default=0.0) for pred in prediction)
        / len(prediction)
        if prediction
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def threshold_metrics(
    ground_truth: Sequence[Interval],
    prediction: Sequence[Interval],
    threshold: float,
) -> dict[str, float]:
    gt_hits = [
        max((interval_iou(gt, pred) for pred in prediction), default=0.0) >= threshold
        for gt in ground_truth
    ]
    pred_hits = [
        max((interval_iou(pred, gt) for gt in ground_truth), default=0.0) >= threshold
        for pred in prediction
    ]
    recall = sum(gt_hits) / len(gt_hits) if gt_hits else 0.0
    precision = sum(pred_hits) / len(pred_hits) if pred_hits else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "all_recall": float(bool(gt_hits) and all(gt_hits)),
    }


def _maximum_matching_count(
    ground_truth: Sequence[Interval], prediction: Sequence[Interval], threshold: float
) -> int:
    adjacency = [
        [index for index, pred in enumerate(prediction) if interval_iou(gt, pred) >= threshold]
        for gt in ground_truth
    ]
    owner = [-1] * len(prediction)

    def augment(gt_index: int, visited: set[int]) -> bool:
        for pred_index in adjacency[gt_index]:
            if pred_index in visited:
                continue
            visited.add(pred_index)
            if owner[pred_index] < 0 or augment(owner[pred_index], visited):
                owner[pred_index] = gt_index
                return True
        return False

    return sum(augment(index, set()) for index in range(len(ground_truth)))


def event_f1_iou(
    ground_truth: Sequence[Interval], prediction: Sequence[Interval], threshold: float = 0.5
) -> dict[str, float]:
    true_positive = _maximum_matching_count(ground_truth, prediction, threshold)
    precision = true_positive / len(prediction) if prediction else 0.0
    recall = true_positive / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def onset_f1_auto_code(
    ground_truth: Sequence[Interval], prediction: Sequence[Interval], tolerance: float = 0.5
) -> dict[str, float]:
    """Reproduce the function currently labelled ev_F1 in the public AEGBench code."""
    if not ground_truth and not prediction:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not ground_truth:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    if not prediction:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for pred_index, (pred_start, _) in enumerate(prediction):
        for gt_index, (gt_start, _) in enumerate(ground_truth):
            if gt_index not in matched_gt and abs(pred_start - gt_start) <= tolerance:
                matched_gt.add(gt_index)
                matched_pred.add(pred_index)
                break
    precision = len(matched_pred) / len(prediction)
    recall = len(matched_pred) / len(ground_truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def segment_f1(
    ground_truth: Sequence[Interval],
    prediction: Sequence[Interval],
    duration: float,
    frame_seconds: float = 1.0,
) -> dict[str, float]:
    frame_count = max(1, int(math.ceil(duration / frame_seconds)))
    gt_mask = [False] * frame_count
    pred_mask = [False] * frame_count
    for intervals, mask in ((ground_truth, gt_mask), (prediction, pred_mask)):
        for start, end in intervals:
            first = max(0, int(start / frame_seconds))
            stop = min(frame_count, int(math.ceil(end / frame_seconds)))
            for index in range(first, stop):
                mask[index] = True
    true_positive = sum(gt and pred for gt, pred in zip(gt_mask, pred_mask))
    false_positive = sum(not gt and pred for gt, pred in zip(gt_mask, pred_mask))
    false_negative = sum(gt and not pred for gt, pred in zip(gt_mask, pred_mask))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def setpo_quality(ground_truth: Sequence[Interval], prediction: Sequence[Interval]) -> float:
    """Continuous quality used by SetPO: equal weight on set IoU and soft symmetric IoU."""
    _, _, soft_f1 = soft_precision_recall(ground_truth, prediction)
    return 0.5 * temporal_set_iou(ground_truth, prediction) + 0.5 * soft_f1


def _threshold_margin(
    ground_truth: Sequence[Interval], prediction: Sequence[Interval], threshold: float
) -> float:
    """Continuous precision/recall margin above an evaluation IoU threshold."""

    def margin(interval: Interval, choices: Sequence[Interval]) -> float:
        best = max((interval_iou(interval, other) for other in choices), default=0.0)
        return max(0.0, min(1.0, (best - threshold) / (1.0 - threshold)))

    recall = (
        sum(margin(gt, prediction) for gt in ground_truth) / len(ground_truth)
        if ground_truth
        else 0.0
    )
    precision = (
        sum(margin(pred, ground_truth) for pred in prediction) / len(prediction)
        if prediction
        else 0.0
    )
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def setpo_quality_axes(
    ground_truth: Sequence[Interval],
    prediction: Sequence[Interval],
    duration: float | None = None,
) -> tuple[float, float, float, float]:
    """Independent SetPO axes: coverage, threshold margin, count, and boundaries."""
    gt = normalize_intervals(ground_truth, duration)
    pred = normalize_intervals(prediction, duration)
    set_iou = temporal_set_iou(gt, pred)
    threshold_margin = sum(_threshold_margin(gt, pred, value) for value in (0.3, 0.5)) / 2

    largest_count = max(len(gt), len(pred), 1)
    cardinality = 1.0 - abs(len(gt) - len(pred)) / largest_count

    if not gt or not pred:
        boundary = float(not gt and not pred)
    else:
        errors = []
        for start, end in gt:
            matched = max(pred, key=lambda item: interval_iou((start, end), item))
            scale = max(end - start, 0.1)
            errors.append((abs(start - matched[0]) + abs(end - matched[1])) / (2 * scale))
        boundary = math.exp(-sum(errors) / len(errors))
    return set_iou, threshold_margin, cardinality, boundary
