#!/usr/bin/env python3
"""Learn conservative boundary-edit utility relative to a locked incumbent.

Unlike occupancy decoding, this module supervises the quantity used for the
decision: the change in temporal set-IoU caused by a local boundary edit.  The
identity edit has an exact feature vector and utility of zero, so keeping the
incumbent is both a learnable and an always-available action.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from interval_metrics import temporal_set_iou
from metric_iou_decoder import posterior_set_iou, resample_probabilities, validate_intervals
from run_metric_iou_pilot import file_sha256


ACTION_SCALE_SECONDS = 0.25
FEATURE_VERSION = "boundary-utility-v1"


@dataclass(frozen=True)
class RidgeUtilityModel:
    alpha: float
    scale: list[float]
    weights: list[float]

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        return (values / np.asarray(self.scale)) @ np.asarray(self.weights)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RidgeUtilityModel":
        return cls(float(payload["alpha"]), list(payload["scale"]), list(payload["weights"]))


@dataclass(frozen=True)
class UtilityDecodeResult:
    incumbent: list[tuple[float, float]]
    candidate: list[tuple[float, float]]
    selected: list[tuple[float, float]]
    predicted_gain: float
    gain_lower_quantile: float
    switch: bool
    abstain_reason: str | None


@dataclass(frozen=True)
class _Option:
    start: float
    end: float
    features: np.ndarray
    identity: bool


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _git_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
    ).strip()


def _sigmoid(values: Sequence[float]) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-array))


def evidence_arrays(record: dict) -> list[np.ndarray]:
    """Return bounded fine-grid evidence channels without label-fitted calibration."""
    fine_steps = int(record["fine_steps"])
    required = (
        "occupancy_logits", "onset_logits", "offset_logits",
        "fine_onset_logits", "fine_offset_logits",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"boundary evidence is missing {missing}")
    return [
        resample_probabilities(_sigmoid(record["occupancy_logits"]), fine_steps),
        resample_probabilities(_sigmoid(record["onset_logits"]), fine_steps),
        resample_probabilities(_sigmoid(record["offset_logits"]), fine_steps),
        _sigmoid(record["fine_onset_logits"]),
        _sigmoid(record["fine_offset_logits"]),
    ]


def _sample(values: np.ndarray, seconds: float, duration: float) -> float:
    positions = (np.arange(values.size, dtype=np.float64) + 0.5) * duration / values.size
    return float(np.interp(seconds, positions, values, left=values[0], right=values[-1]))


def _window_mean(
    values: np.ndarray, start: float, end: float, duration: float
) -> float:
    positions = (np.arange(values.size, dtype=np.float64) + 0.5) * duration / values.size
    mask = (positions >= max(0.0, start)) & (positions <= min(duration, end))
    return float(values[mask].mean()) if mask.any() else _sample(values, (start + end) / 2, duration)


def _boundary_signature(
    arrays: Sequence[np.ndarray], value: float, duration: float
) -> np.ndarray:
    window = min(0.12, duration / 20)
    signature = []
    for values in arrays:
        center = _sample(values, value, duration)
        left = _window_mean(values, value - window, value, duration)
        right = _window_mean(values, value, value + window, duration)
        signature.extend((center, right - left, max(left, center, right)))
    return np.asarray(signature, dtype=np.float64)


def boundary_edit_features(
    record: dict,
    interval_index: int,
    candidate_start: float,
    candidate_end: float,
    *,
    arrays: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Feature difference for one interval edit; the exact identity maps to zero."""
    duration = float(record["duration"])
    intervals = validate_intervals(record["incumbent_prediction"], duration)
    original_start, original_end = intervals[interval_index]
    arrays = list(arrays) if arrays is not None else evidence_arrays(record)
    delta_start = candidate_start - original_start
    delta_end = candidate_end - original_end
    start_scaled, end_scaled = delta_start / ACTION_SCALE_SECONDS, delta_end / ACTION_SCALE_SECONDS
    length_change = delta_end - delta_start
    center_change = (delta_start + delta_end) / 2
    coverage = sum(end - start for start, end in intervals) / duration
    previous_gap = original_start - intervals[interval_index - 1][1] if interval_index else original_start
    next_gap = intervals[interval_index + 1][0] - original_end if interval_index + 1 < len(intervals) else duration - original_end
    occupancy = arrays[0]
    context = np.asarray(
        [
            (original_end - original_start) / duration,
            original_start / duration,
            original_end / duration,
            coverage,
            len(intervals) / 8.0,
            interval_index / max(1, len(intervals) - 1),
            previous_gap / duration,
            next_gap / duration,
            posterior_set_iou(occupancy, intervals, duration),
            _window_mean(occupancy, original_start, original_end, duration),
        ],
        dtype=np.float64,
    )
    action = np.asarray(
        [
            start_scaled,
            end_scaled,
            abs(start_scaled),
            abs(end_scaled),
            start_scaled * start_scaled,
            end_scaled * end_scaled,
            start_scaled * end_scaled,
            length_change / ACTION_SCALE_SECONDS,
            center_change / ACTION_SCALE_SECONDS,
        ],
        dtype=np.float64,
    )
    original_signature = np.r_[
        _boundary_signature(arrays, original_start, duration),
        _boundary_signature(arrays, original_end, duration),
    ]
    candidate_signature = np.r_[
        _boundary_signature(arrays, candidate_start, duration),
        _boundary_signature(arrays, candidate_end, duration),
    ]
    evidence_change = candidate_signature - original_signature
    interactions = np.r_[start_scaled * context, end_scaled * context, (length_change / ACTION_SCALE_SECONDS) * context]
    features = np.r_[action, evidence_change, interactions]
    if delta_start == 0.0 and delta_end == 0.0 and not np.array_equal(features, np.zeros_like(features)):
        raise RuntimeError("identity edit must have an exact zero feature vector")
    return features


def _endpoint_choices(value: float, duration: float, steps: int, radius: float) -> list[float]:
    step = duration / steps
    low, high = max(0.0, value - radius), min(duration, value + radius)
    first = max(0, int(math.ceil(low / step - 1e-12)))
    last = min(steps, int(math.floor(high / step + 1e-12)))
    values = [index * step for index in range(first, last + 1)] + [value]
    return sorted(set(round(item, 12) for item in values))


def interval_options(record: dict, interval_index: int, radius: float) -> list[_Option]:
    duration, steps = float(record["duration"]), int(record["fine_steps"])
    intervals = validate_intervals(record["incumbent_prediction"], duration)
    original_start, original_end = intervals[interval_index]
    arrays = evidence_arrays(record)
    minimum = duration / steps
    options = []
    for start in _endpoint_choices(original_start, duration, steps, radius):
        for end in _endpoint_choices(original_end, duration, steps, radius):
            if end - start + 1e-12 < minimum:
                continue
            if interval_index and start <= intervals[interval_index - 1][1]:
                continue
            if interval_index + 1 < len(intervals) and end >= intervals[interval_index + 1][0]:
                continue
            options.append(
                _Option(
                    start,
                    end,
                    boundary_edit_features(
                        record, interval_index, start, end, arrays=arrays
                    ),
                    abs(start - original_start) <= 1e-12 and abs(end - original_end) <= 1e-12,
                )
            )
    if not options or not any(option.identity for option in options):
        raise RuntimeError("trust region lost the identity boundary edit")
    return sorted(options, key=lambda item: (item.end, item.start))


def make_training_examples(records: Sequence[dict], radius: float) -> list[dict]:
    examples = []
    for record in records:
        duration = float(record["duration"])
        incumbent = validate_intervals(record["incumbent_prediction"], duration)
        if not incumbent:
            continue
        incumbent_iou = temporal_set_iou(record["annotations"], incumbent)
        for interval_index in range(len(incumbent)):
            for option in interval_options(record, interval_index, radius):
                if option.identity:
                    continue
                candidate = list(incumbent)
                candidate[interval_index] = (option.start, option.end)
                examples.append(
                    {
                        "features": option.features,
                        "target": temporal_set_iou(record["annotations"], candidate) - incumbent_iou,
                        "source": str(record["source"]),
                        "audio_group": str(record["audio_group"]),
                        "row_key": (str(record["source"]), int(record["source_index"])),
                    }
                )
    if not examples:
        raise ValueError("no non-identity boundary edits were generated")
    return examples


def fit_ridge_utility(examples: Sequence[dict], alpha: float) -> RidgeUtilityModel:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    sources = sorted({example["source"] for example in examples})
    groups_by_source: dict[str, set[str]] = defaultdict(set)
    rows_by_group: dict[tuple[str, str], set[tuple]] = defaultdict(set)
    options_by_row = Counter(example["row_key"] for example in examples)
    for example in examples:
        groups_by_source[example["source"]].add(example["audio_group"])
        rows_by_group[(example["source"], example["audio_group"])].add(example["row_key"])
    sample_weight = np.asarray(
        [
            1.0
            / len(sources)
            / len(groups_by_source[example["source"]])
            / len(rows_by_group[(example["source"], example["audio_group"])])
            / options_by_row[example["row_key"]]
            for example in examples
        ],
        dtype=np.float64,
    )
    sample_weight /= sample_weight.sum()
    x = np.stack([example["features"] for example in examples])
    y = np.asarray([example["target"] for example in examples], dtype=np.float64)
    scale = np.sqrt(np.sum(sample_weight[:, None] * x * x, axis=0)).clip(min=1e-5)
    normalised = x / scale
    gram = normalised.T @ (sample_weight[:, None] * normalised)
    target = normalised.T @ (sample_weight * y)
    weights = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), target)
    return RidgeUtilityModel(float(alpha), scale.tolist(), weights.tolist())


def bootstrap_utility_models(
    examples: Sequence[dict], alpha: float, samples: int, seed: int
) -> list[RidgeUtilityModel]:
    """Fit group bootstraps from cached sufficient statistics."""
    reference = fit_ridge_utility(examples, alpha)
    scale = np.asarray(reference.scale)
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        grouped[example["source"]][example["audio_group"]].append(example)
    sufficient: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    feature_count = len(reference.weights)
    for source in sorted(grouped):
        for group in sorted(grouped[source]):
            members = grouped[source][group]
            options_by_row = Counter(example["row_key"] for example in members)
            row_count = len(options_by_row)
            weights = np.asarray(
                [1.0 / row_count / options_by_row[example["row_key"]] for example in members]
            )
            x = np.stack([example["features"] for example in members]) / scale
            y = np.asarray([example["target"] for example in members])
            sufficient[(source, group)] = (
                x.T @ (weights[:, None] * x),
                x.T @ (weights * y),
            )
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(samples):
        gram = np.zeros((feature_count, feature_count), dtype=np.float64)
        target = np.zeros(feature_count, dtype=np.float64)
        for source in sorted(grouped):
            names = sorted(grouped[source])
            chosen = rng.choice(names, size=len(names), replace=True)
            for name in chosen:
                local_gram, local_target = sufficient[(source, str(name))]
                gram += local_gram / len(grouped) / len(names)
                target += local_target / len(grouped) / len(names)
        weights = np.linalg.solve(gram + alpha * np.eye(feature_count), target)
        models.append(RidgeUtilityModel(float(alpha), scale.tolist(), weights.tolist()))
    return models


def _best_option_set(
    layers: Sequence[Sequence[_Option]], model: RidgeUtilityModel
) -> tuple[list[tuple[float, float]], list[_Option], float]:
    layer_scores = [np.asarray(model.predict(np.stack([option.features for option in layer]))) for layer in layers]
    scores = layer_scores[0].copy()
    backs = []
    for layer_index in range(1, len(layers)):
        previous = layers[layer_index - 1]
        previous_ends = [option.end for option in previous]
        prefix_best = np.empty(len(previous), dtype=np.int64)
        best = 0
        for index in range(len(previous)):
            if scores[index] > scores[best] + 1e-15:
                best = index
            prefix_best[index] = best
        current = np.full(len(layers[layer_index]), -np.inf)
        back = np.full(len(layers[layer_index]), -1, dtype=np.int64)
        for index, option in enumerate(layers[layer_index]):
            stop = bisect.bisect_left(previous_ends, option.start - 1e-12) - 1
            if stop >= 0:
                parent = int(prefix_best[stop])
                current[index] = scores[parent] + layer_scores[layer_index][index]
                back[index] = parent
        scores, back = current, back
        backs.append(back)
    chosen = int(np.argmax(scores))
    indices = [chosen]
    for back in reversed(backs):
        chosen = int(back[chosen])
        indices.append(chosen)
    indices.reverse()
    selected_options = [layers[index][option_index] for index, option_index in enumerate(indices)]
    return (
        [(option.start, option.end) for option in selected_options],
        selected_options,
        float(scores[indices[-1]]),
    )


def decode_boundary_utility(
    record: dict,
    model: RidgeUtilityModel,
    bootstrap_models: Sequence[RidgeUtilityModel],
    radius: float,
    margin: float,
) -> UtilityDecodeResult:
    duration = float(record["duration"])
    try:
        incumbent = validate_intervals(record["incumbent_prediction"], duration)
    except ValueError as error:
        raw = [(float(item[0]), float(item[1])) for item in record["incumbent_prediction"]]
        return UtilityDecodeResult(raw, raw, raw, 0.0, 0.0, False, f"invalid_incumbent:{error}")
    if not incumbent:
        return UtilityDecodeResult([], [], [], 0.0, 0.0, False, "empty_incumbent")
    layers = [interval_options(record, index, radius) for index in range(len(incumbent))]
    candidate, selected_options, predicted_gain = _best_option_set(layers, model)
    if candidate == incumbent or predicted_gain <= 0:
        return UtilityDecodeResult(
            incumbent, candidate, incumbent, predicted_gain, predicted_gain,
            False, "identity_or_non_positive_gain",
        )
    total_features = np.stack([option.features for option in selected_options]).sum(axis=0)
    gains = np.asarray([item.predict(total_features) for item in bootstrap_models], dtype=np.float64)
    lower = float(np.quantile(gains, 0.05)) if gains.size else predicted_gain
    if lower <= margin:
        return UtilityDecodeResult(
            incumbent, candidate, incumbent, predicted_gain, lower,
            False, "gain_lower_bound_below_margin",
        )
    return UtilityDecodeResult(incumbent, candidate, candidate, predicted_gain, lower, True, None)


def _metrics(records: Sequence[dict], results: Sequence[UtilityDecodeResult]) -> dict:
    incumbent = np.asarray(
        [temporal_set_iou(row["annotations"], result.incumbent) for row, result in zip(records, results)]
    )
    selected = np.asarray(
        [temporal_set_iou(row["annotations"], result.selected) for row, result in zip(records, results)]
    )
    delta = selected - incumbent
    return {
        "rows": len(records),
        "incumbent_mIoU_percent": float(incumbent.mean() * 100),
        "selected_mIoU_percent": float(selected.mean() * 100),
        "delta_mIoU_points": float(delta.mean() * 100),
        "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
        "catastrophic": int((delta <= -0.5).sum()),
        "switches": int(sum(result.switch for result in results)),
    }


def command_select(args) -> None:
    records_by_source = {
        path.stem.replace("_boundary_logits", ""): _jsonl(path) for path in args.logits
    }
    required = {"longneedle", "clotho", "sc"}
    if set(records_by_source) != required:
        raise ValueError(f"expected boundary logits for {sorted(required)}, got {sorted(records_by_source)}")
    alphas = [float(value) for value in args.alphas.split(",")]
    radii = [float(value) for value in args.radii.split(",")]
    margins = [float(value) for value in args.margins.split(",")]
    maximum_radius = max(radii)
    configs = {
        (alpha, radius, margin): {"sources": {}, "records": [], "results": []}
        for alpha in alphas for radius in radii for margin in margins
    }
    diagnostics = []
    for fold_index, heldout in enumerate(sorted(required)):
        training_records = [row for source, rows in records_by_source.items() if source != heldout for row in rows]
        examples = make_training_examples(training_records, maximum_radius)
        heldout_records = records_by_source[heldout]
        for alpha in alphas:
            model = fit_ridge_utility(examples, alpha)
            ensemble = bootstrap_utility_models(examples, alpha, args.bootstrap_models, args.seed + fold_index * 1000 + int(alpha * 1e6))
            for radius in radii:
                candidates = [
                    decode_boundary_utility(record, model, ensemble, radius, min(margins))
                    for record in heldout_records
                ]
                # Candidate identity and lower bound do not depend on the margin.
                for margin in margins:
                    results = []
                    for candidate in candidates:
                        switch = (
                            candidate.candidate != candidate.incumbent
                            and candidate.predicted_gain > 0
                            and candidate.gain_lower_quantile > margin
                        )
                        results.append(
                            UtilityDecodeResult(
                                candidate.incumbent,
                                candidate.candidate,
                                candidate.candidate if switch else candidate.incumbent,
                                candidate.predicted_gain,
                                candidate.gain_lower_quantile,
                                switch,
                                None if switch else "gain_lower_bound_below_margin",
                            )
                        )
                    entry = configs[(alpha, radius, margin)]
                    metrics = _metrics(heldout_records, results)
                    entry["sources"][heldout] = metrics
                    entry["records"].extend(heldout_records)
                    entry["results"].extend(results)
    passing = []
    report_configs = {}
    for key, entry in configs.items():
        alpha, radius, margin = key
        combined = _metrics(entry["records"], entry["results"])
        gate = (
            all(source["delta_mIoU_points"] >= 0 and source["catastrophic"] == 0 for source in entry["sources"].values())
            and combined["delta_mIoU_points"] > 0
        )
        name = f"a={alpha}|r={radius}|m={margin}"
        report_configs[name] = {"alpha": alpha, "radius_seconds": radius, "margin": margin, "sources": entry["sources"], "combined": combined, "gate_passed": gate}
        if gate:
            worst = min(source["delta_mIoU_points"] for source in entry["sources"].values())
            passing.append((worst, combined["delta_mIoU_points"], -radius, margin, key))
    selected_key = max(passing)[-1] if passing else None
    all_records = [row for source in sorted(required) for row in records_by_source[source]]
    final_examples = make_training_examples(all_records, maximum_radius)
    if selected_key is None:
        # Preserve a diagnostic model without authorising public evaluation.
        best_name = max(report_configs, key=lambda name: report_configs[name]["combined"]["delta_mIoU_points"])
        diagnostic = report_configs[best_name]
        alpha, radius, margin = diagnostic["alpha"], diagnostic["radius_seconds"], diagnostic["margin"]
    else:
        alpha, radius, margin = selected_key
    final_model = fit_ridge_utility(final_examples, alpha)
    final_ensemble = bootstrap_utility_models(final_examples, alpha, args.bootstrap_models, args.seed + 9000)
    artifact = {
        "format_version": 1,
        "feature_version": FEATURE_VERSION,
        "development_gate_passed": selected_key is not None,
        "selected_alpha": alpha if selected_key is not None else None,
        "selected_radius_seconds": radius if selected_key is not None else None,
        "selected_margin": margin if selected_key is not None else None,
        "diagnostic_alpha": alpha,
        "diagnostic_radius_seconds": radius,
        "diagnostic_margin": margin,
        "model": final_model.to_dict(),
        "bootstrap_models": [model.to_dict() for model in final_ensemble],
        "seed": args.seed,
        "code_hash": _git_hash(),
        "input_hashes": {path.stem: file_sha256(path) for path in args.logits},
    }
    report = {
        "development_gate_passed": selected_key is not None,
        "selected": None if selected_key is None else {"alpha": alpha, "radius_seconds": radius, "margin": margin},
        "diagnostic": {"alpha": alpha, "radius_seconds": radius, "margin": margin},
        "configs": report_configs,
    }
    _write_json(args.output, artifact)
    _write_json(args.report, report)
    print(json.dumps({"development_gate_passed": selected_key is not None, "selected": report["selected"], "diagnostic": report["diagnostic"]}, indent=2))


def command_decode(args) -> None:
    artifact = _json(args.model)
    diagnostic = not artifact["development_gate_passed"]
    if diagnostic and not args.diagnostic_override:
        raise RuntimeError("development gate failed; use explicit --diagnostic-override only for analysis")
    radius = float(artifact["diagnostic_radius_seconds"] if diagnostic else artifact["selected_radius_seconds"])
    margin = float(artifact["diagnostic_margin"] if diagnostic else artifact["selected_margin"])
    model = RidgeUtilityModel.from_dict(artifact["model"])
    ensemble = [RidgeUtilityModel.from_dict(payload) for payload in artifact["bootstrap_models"]]
    manifest, records = _json(args.manifest), _jsonl(args.logits)
    indexed = {int(row["source_index"]): row for row in records}
    if len(indexed) != len(records) or set(indexed) != {int(row["source_index"]) for row in manifest}:
        raise ValueError("manifest/logit alignment failed")
    output = []
    for manifest_row in manifest:
        row = indexed[int(manifest_row["source_index"])]
        if str(row["audio_group"]) != str(manifest_row["audio_group"]):
            raise ValueError(f"audio mismatch at {row['source_index']}")
        result = decode_boundary_utility(row, model, ensemble, radius, margin)
        incumbent_iou = temporal_set_iou(row["annotations"], result.incumbent)
        candidate_iou = temporal_set_iou(row["annotations"], result.candidate)
        selected_iou = temporal_set_iou(row["annotations"], result.selected)
        output.append(
            {
                "benchmark": row["benchmark"],
                "source_index": int(row["source_index"]),
                "audio_group": row["audio_group"],
                "audio": row["audio_path"],
                "query": row["caption"],
                "duration": float(row["duration"]),
                "ground_truth": row["annotations"],
                "incumbent_name": row["incumbent_name"],
                "incumbent_prediction": result.incumbent,
                "candidate_prediction": result.candidate,
                "selected_prediction": result.selected,
                "incumbent_iou": incumbent_iou,
                "candidate_iou": candidate_iou,
                "selected_iou": selected_iou,
                "delta_iou": selected_iou - incumbent_iou,
                "predicted_gain": result.predicted_gain,
                "gain_lower_quantile": result.gain_lower_quantile,
                "radius_seconds": radius,
                "margin": margin,
                "switch": result.switch,
                "selected_candidate": "boundary_utility" if result.switch else row["incumbent_name"],
                "abstain_reason": result.abstain_reason,
                "fixed_threshold_prediction": result.incumbent,
                "fixed_threshold_iou": incumbent_iou,
                "diagnostic_override": diagnostic,
                "development_gate_passed": bool(artifact["development_gate_passed"]),
                "feature_version": FEATURE_VERSION,
                "code_hash": artifact["code_hash"],
                "checkpoint_hash": row["export_id"],
                "input_hash": row["input_hash"],
                "config_hash": file_sha256(args.model),
            }
        )
    _write_jsonl(args.output, output)
    print(json.dumps({"rows": len(output), "switches": sum(row["switch"] for row in output), "sha256": file_sha256(args.output)}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--logits", type=Path, action="append", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--report", type=Path, required=True)
    select.add_argument("--alphas", default="0.001,0.01,0.1")
    select.add_argument("--radii", default="0.08,0.12,0.16,0.25")
    select.add_argument("--margins", default="0,0.0025,0.005,0.01,0.02")
    select.add_argument("--bootstrap-models", type=int, default=200)
    select.add_argument("--seed", type=int, default=20260903)
    select.set_defaults(function=command_select)
    decode = commands.add_parser("decode")
    decode.add_argument("--manifest", type=Path, required=True)
    decode.add_argument("--logits", type=Path, required=True)
    decode.add_argument("--model", type=Path, required=True)
    decode.add_argument("--output", type=Path, required=True)
    decode.add_argument("--diagnostic-override", action="store_true")
    decode.set_defaults(function=command_decode)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
