"""Shared, label-free decision logic for the conservative NOVA-Safe v2 router."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from interval_metrics import event_f1_iou, interval_iou, normalize_intervals, temporal_set_iou


SCHEMA_VERSION = "nova-safe-v2"
EQUIVALENCE_IOU = 0.995
CATASTROPHIC_DROP = 0.5
BOOTSTRAP_QUANTILE = 0.05

GEOMETRY_FEATURES = (
    "agreement",
    "mean_boundary_shift",
    "max_boundary_shift",
    "signed_coverage_delta",
    "absolute_coverage_delta",
    "count_delta",
    "absolute_count_delta",
    "candidate_edge_fraction",
    "incumbent_edge_fraction",
    "duration_log1p",
    "query_words",
)
KEEP_FEATURES = GEOMETRY_FEATURES + (
    "component_mean_delta",
    "component_min_delta",
)
FULL_FEATURES = KEEP_FEATURES + ("complement_no_delta",)
FEATURE_SETS = {
    "geometry": GEOMETRY_FEATURES,
    "keep": KEEP_FEATURES,
    "full": FULL_FEATURES,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_key(row: dict, benchmark: str | None = None, fallback_index: int | None = None):
    resolved_benchmark = benchmark or row.get("benchmark")
    index = row.get("source_index", row.get("index", fallback_index))
    if resolved_benchmark is None or index is None:
        raise ValueError("Every prediction needs a benchmark and source_index/index")
    return str(resolved_benchmark), int(index)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def index_rows(rows: Sequence[dict], benchmark: str) -> dict[tuple[str, int], dict]:
    indexed = {}
    for fallback_index, row in enumerate(rows):
        key = canonical_key(row, benchmark, fallback_index)
        if key in indexed:
            raise ValueError(f"Duplicate prediction key: {key}")
        indexed[key] = row
    return indexed


def prediction_error(prediction: object, duration: float) -> str | None:
    if not isinstance(prediction, list):
        return "not_a_list"
    for item in prediction:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return "invalid_pair"
        try:
            start, end = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return "non_numeric"
        if not math.isfinite(start) or not math.isfinite(end):
            return "non_finite"
        if start < 0.0 or end > duration + 1e-6:
            return "out_of_bounds"
        if end - start <= 1e-3:
            return "non_positive_interval"
    return None


def checked_prediction(candidate: dict, duration: float) -> list[tuple[float, float]]:
    prediction = candidate.get("prediction")
    error = prediction_error(prediction, duration)
    if error:
        raise ValueError(f"Invalid prediction ({error}): {prediction}")
    return normalize_intervals(prediction, duration)


def hard_guard(incumbent: dict, challenger: dict, duration: float) -> str | None:
    incumbent_error = prediction_error(incumbent.get("prediction"), duration)
    if incumbent_error:
        raise ValueError(f"Invalid incumbent prediction: {incumbent_error}")
    challenger_error = prediction_error(challenger.get("prediction"), duration)
    if challenger_error:
        return f"invalid_candidate:{challenger_error}"
    incumbent_prediction = normalize_intervals(incumbent["prediction"], duration)
    challenger_prediction = normalize_intervals(challenger["prediction"], duration)
    if incumbent_prediction and not challenger_prediction:
        return "candidate_empty"
    if abs(len(challenger_prediction) - len(incumbent_prediction)) > 1:
        return "cardinality_jump"
    if temporal_set_iou(incumbent_prediction, challenger_prediction) >= EQUIVALENCE_IOU:
        return "equivalent_prediction"
    return None


def _coverage(intervals: Iterable[Sequence[float]], duration: float) -> float:
    normalized = normalize_intervals(intervals, duration)
    return sum(end - start for start, end in normalized) / duration if duration else 0.0


def _edge_fraction(intervals: Sequence[Sequence[float]], duration: float) -> float:
    if not intervals:
        return 0.0
    tolerance = max(0.04, duration * 1e-4)
    touches = sum(start <= tolerance or end >= duration - tolerance for start, end in intervals)
    return touches / len(intervals)


def _boundary_shifts(
    incumbent: Sequence[Sequence[float]], challenger: Sequence[Sequence[float]], duration: float
) -> list[float]:
    if not incumbent or not challenger:
        return [1.0]
    shifts = []
    for interval in challenger:
        matched = max(incumbent, key=lambda other: interval_iou(interval, other))
        shifts.append((abs(interval[0] - matched[0]) + abs(interval[1] - matched[1])) / (2 * duration))
    return shifts


def pair_feature_map(row: dict, challenger_name: str) -> dict[str, float]:
    incumbent_name = row["incumbent"]
    incumbent = row["candidates"][incumbent_name]
    challenger = row["candidates"][challenger_name]
    duration = float(row["duration_seconds"])
    incumbent_prediction = checked_prediction(incumbent, duration)
    challenger_prediction = checked_prediction(challenger, duration)
    incumbent_coverage = _coverage(incumbent_prediction, duration)
    challenger_coverage = _coverage(challenger_prediction, duration)
    shifts = _boundary_shifts(incumbent_prediction, challenger_prediction, duration)
    features = {
        "agreement": temporal_set_iou(incumbent_prediction, challenger_prediction),
        "mean_boundary_shift": float(np.mean(shifts)),
        "max_boundary_shift": float(np.max(shifts)),
        "signed_coverage_delta": challenger_coverage - incumbent_coverage,
        "absolute_coverage_delta": abs(challenger_coverage - incumbent_coverage),
        "count_delta": float(len(challenger_prediction) - len(incumbent_prediction)),
        "absolute_count_delta": float(abs(len(challenger_prediction) - len(incumbent_prediction))),
        "candidate_edge_fraction": _edge_fraction(challenger_prediction, duration),
        "incumbent_edge_fraction": _edge_fraction(incumbent_prediction, duration),
        "duration_log1p": math.log1p(duration),
        "query_words": float(len(str(row.get("query", "")).split())),
    }
    incumbent_verifier = incumbent.get("features", {})
    challenger_verifier = challenger.get("features", {})
    for name in ("component_mean", "component_min", "complement_no"):
        if name in incumbent_verifier and name in challenger_verifier:
            features[f"{name}_delta"] = float(challenger_verifier[name]) - float(
                incumbent_verifier[name]
            )
    return features


def feature_vector(row: dict, challenger_name: str, feature_names: Sequence[str]) -> np.ndarray:
    values = pair_feature_map(row, challenger_name)
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise ValueError(f"Missing features for {challenger_name}: {missing}")
    vector = np.asarray([values[name] for name in feature_names], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError(f"Non-finite feature vector for {challenger_name}")
    return vector


def training_pairs(rows: Sequence[dict], feature_names: Sequence[str]) -> list[dict]:
    pairs = []
    for row_number, row in enumerate(rows):
        incumbent_name = row["incumbent"]
        incumbent = row["candidates"][incumbent_name]
        if incumbent.get("iou") is None:
            raise ValueError("Development rows require candidate IoU labels")
        duration = float(row["duration_seconds"])
        for challenger_name, challenger in row["candidates"].items():
            if challenger_name == incumbent_name:
                continue
            guard = hard_guard(incumbent, challenger, duration)
            if guard:
                continue
            if challenger.get("iou") is None:
                raise ValueError("Development rows require candidate IoU labels")
            pairs.append(
                {
                    "row_number": row_number,
                    "source": str(row["source"]),
                    "group": f"{row['source']}|{row['audio_group']}",
                    "challenger": challenger_name,
                    "x": feature_vector(row, challenger_name, feature_names),
                    "y": float(challenger["iou"]) - float(incumbent["iou"]),
                }
            )
    if not pairs:
        raise ValueError("No guard-eligible development pairs")
    return pairs


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def fit_bootstrap_ensemble(
    pairs: Sequence[dict], alpha: float, seed: int, bootstrap_models: int
) -> dict:
    x = np.stack([pair["x"] for pair in pairs])
    y = np.asarray([pair["y"] for pair in pairs], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (x - mean) / scale
    grouped: dict[str, list[int]] = defaultdict(list)
    source_groups: dict[str, list[str]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        grouped[pair["group"]].append(index)
    for group in sorted(grouped):
        source = group.split("|", 1)[0]
        source_groups[source].append(group)
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(bootstrap_models):
        sampled_indices = []
        for source in sorted(source_groups):
            groups = source_groups[source]
            sampled = rng.choice(groups, size=len(groups), replace=True)
            for group in sampled:
                sampled_indices.extend(grouped[str(group)])
        sample = np.asarray(sampled_indices, dtype=np.int64)
        weights.append(_ridge(normalized[sample], y[sample], alpha))
    return {
        "mean": mean,
        "scale": scale,
        "weights": np.stack(weights),
    }


def ensemble_prediction(ensemble: dict, vector: np.ndarray) -> tuple[float, float]:
    normalized = (vector - ensemble["mean"]) / ensemble["scale"]
    predictions = ensemble["weights"] @ np.r_[1.0, normalized]
    return float(predictions.mean()), float(np.quantile(predictions, BOOTSTRAP_QUANTILE))


def choose_candidate(row: dict, model: dict) -> dict:
    incumbent_name = row["incumbent"]
    candidates = row["candidates"]
    if incumbent_name not in candidates:
        raise ValueError(f"Missing incumbent candidate: {incumbent_name}")
    duration = float(row["duration_seconds"])
    feature_names = tuple(model["feature_names"])
    ensemble = {
        "mean": np.asarray(model["mean"], dtype=np.float64),
        "scale": np.asarray(model["scale"], dtype=np.float64),
        "weights": np.asarray(model["weights"], dtype=np.float64),
    }
    threshold = float(model["threshold"])
    scored = {}
    guarded = {}
    for name, candidate in candidates.items():
        if name == incumbent_name:
            continue
        reason = hard_guard(candidates[incumbent_name], candidate, duration)
        if reason:
            guarded[name] = reason
            continue
        mean, lower = ensemble_prediction(ensemble, feature_vector(row, name, feature_names))
        scored[name] = {
            "predicted_gain": mean,
            "lower_confidence_bound": lower,
            "decision_margin": lower - threshold,
        }
    if scored:
        best = max(scored, key=lambda name: (scored[name]["decision_margin"], name))
        if scored[best]["decision_margin"] > 0.0:
            return {
                "selected": best,
                "scores": scored,
                "guarded": guarded,
                "decision_margin": scored[best]["decision_margin"],
                "abstain_reason": None,
            }
    if not scored:
        reason = "all_challengers_guarded"
    else:
        reason = "no_positive_lower_bound"
    return {
        "selected": incumbent_name,
        "scores": scored,
        "guarded": guarded,
        "decision_margin": max(
            (score["decision_margin"] for score in scored.values()), default=None
        ),
        "abstain_reason": reason,
    }


def evaluate_selection(rows: Sequence[dict], selections: Sequence[str]) -> dict:
    if len(rows) != len(selections):
        raise ValueError("Rows and selections must have equal length")
    incumbent_scores = []
    selected_scores = []
    event_scores = []
    for row, selected_name in zip(rows, selections):
        incumbent = row["candidates"][row["incumbent"]]
        selected = row["candidates"][selected_name]
        if incumbent.get("iou") is None or selected.get("iou") is None:
            raise ValueError("Evaluation requires candidate IoU labels")
        incumbent_scores.append(float(incumbent["iou"]))
        selected_scores.append(float(selected["iou"]))
        ground_truth = normalize_intervals(row.get("ground_truth", []), row["duration_seconds"])
        prediction = normalize_intervals(selected["prediction"], row["duration_seconds"])
        event_scores.append(event_f1_iou(ground_truth, prediction, 0.5)["f1"])
    incumbent_array = np.asarray(incumbent_scores)
    selected_array = np.asarray(selected_scores)
    delta = selected_array - incumbent_array
    return {
        "rows": len(rows),
        "incumbent_mIoU_percent": float(incumbent_array.mean() * 100),
        "router_mIoU_percent": float(selected_array.mean() * 100),
        "delta_mIoU_points": float(delta.mean() * 100),
        "R1@0.3_percent": float((selected_array >= 0.3).mean() * 100),
        "R1@0.5_percent": float((selected_array >= 0.5).mean() * 100),
        "R1@0.7_percent": float((selected_array >= 0.7).mean() * 100),
        "event_F1@0.5_percent": float(np.mean(event_scores) * 100),
        "improved_rows": int((delta > 1e-12).sum()),
        "tied_rows": int((np.abs(delta) <= 1e-12).sum()),
        "regressed_rows": int((delta < -1e-12).sum()),
        "new_catastrophic_regressions": int((delta <= -CATASTROPHIC_DROP).sum()),
        "selection_counts": dict(Counter(selections)),
    }

