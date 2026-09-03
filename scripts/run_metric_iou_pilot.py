#!/usr/bin/env python3
"""Prepare, calibrate, decode, and audit the NOVA-MBR pilot.

Existing autoregressive/NOVA predictions are immutable inputs.  The only GPU
operation performed here is a frozen backbone + SpanTool forward pass that
exports occupancy logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from interval_metrics import event_f1_iou, normalize_intervals, temporal_set_iou
from metric_iou_decoder import (
    calibrate_occupancy,
    dinkelbach_trust_region_decode,
    resample_probabilities,
)


DEFAULT_CONFIG = Path("experiments/protocols/NOVA_MBR_PILOT_V1.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha256(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for candidate in files:
        digest.update(candidate.relative_to(path).as_posix().encode())
        digest.update(file_sha256(candidate).encode())
    if not files:
        raise ValueError(f"artifact contains no files: {path}")
    return digest.hexdigest()


def resolve_audio_path(audio_root: Path, row: dict) -> Path:
    relative = Path(str(row["audio_path"]))
    candidates = [relative] if relative.is_absolute() else [audio_root / relative]
    candidates.append(audio_root / relative.name)
    for base in list(candidates):
        candidates.extend(base.with_suffix(suffix) for suffix in (".wav", ".flac", ".mp3"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"audio not found for {relative} under {audio_root}")


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


def _stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _git_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
    ).strip()


def _duration(row: dict) -> float:
    for name in ("duration", "duration_seconds"):
        if row.get(name) is not None:
            return float(row[name])
    endpoints = [float(end) for item in row.get("annotations", row.get("ground_truth", [])) for end in item[1:2]]
    endpoints += [float(end) for item in row.get("incumbent_prediction", row.get("prediction", [])) for end in item[1:2]]
    if not endpoints:
        raise ValueError("duration is absent and cannot be inferred")
    return max(endpoints)


def _audio_group(row: dict) -> str:
    value = row.get("audio_group", row.get("audio_path", row.get("audio")))
    if value is None:
        raise ValueError("row has no audio identifier")
    return Path(str(value)).name


def _source_index(row: dict, fallback: int) -> int:
    return int(row.get("source_index", row.get("index", fallback)))


def _strict_index(rows: Sequence[dict], label: str) -> dict[int, dict]:
    result = {}
    for fallback, row in enumerate(rows):
        index = _source_index(row, fallback)
        if index in result:
            raise ValueError(f"duplicate {label} source_index {index}")
        result[index] = row
    return result


def _paired_rows(
    manifest_path: Path,
    incumbent_path: Path,
    source: str,
    incumbent_name: str,
    benchmark: str,
    audio_dir: Path | None = None,
) -> list[dict]:
    manifest_rows = _jsonl(manifest_path) if manifest_path.suffix == ".jsonl" else _json(manifest_path)
    incumbent_rows = _jsonl(incumbent_path)
    manifest = _strict_index(manifest_rows, f"{source} manifest")
    incumbent = _strict_index(incumbent_rows, f"{source} incumbent")
    if set(manifest) != set(incumbent):
        raise ValueError(
            f"{source} alignment mismatch: manifest-only={len(set(manifest)-set(incumbent))}, "
            f"incumbent-only={len(set(incumbent)-set(manifest))}"
        )
    output = []
    for index in sorted(manifest):
        row, prediction = manifest[index], incumbent[index]
        audio = row.get("audio_path", row.get("audio", prediction.get("audio")))
        query = row.get("caption", row.get("query", prediction.get("query")))
        truth = row.get("annotations", row.get("ground_truth", prediction.get("ground_truth", [])))
        duration = prediction.get("duration_seconds", row.get("duration", row.get("duration_seconds")))
        if duration is None and audio_dir is not None:
            import soundfile as sf

            info = sf.info(resolve_audio_path(audio_dir, {"audio_path": audio}))
            duration = info.frames / info.samplerate
        item = {
            "benchmark": benchmark,
            "source": source,
            "source_index": index,
            "audio_path": audio,
            "audio_group": Path(str(audio)).name,
            "caption": query,
            "annotations": truth,
            "incumbent_name": incumbent_name,
            "incumbent_prediction": prediction["prediction"],
            "stored_incumbent_iou": prediction.get("iou"),
        }
        if duration is not None:
            item["duration"] = float(duration)
        output.append(item)
    return output


def _longneedle_rows(config: dict) -> list[dict]:
    spec = config["development"]["longneedle"]
    manifest = _json(Path(spec["manifest"]))
    features = _jsonl(Path(spec["features"]))
    model = _json(Path(spec["router_model"]))
    feature_rows = _strict_index(features, "longneedle features")
    if len(manifest) != len(features):
        raise ValueError("longneedle manifest/features row count mismatch")
    names = model["feature_names"]
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    weights, threshold = np.asarray(model["weights"]), float(model["threshold"])
    official = model["official"]
    output = []
    for index, row in enumerate(manifest):
        feature = feature_rows[index]
        candidates = feature["candidates"]
        base = np.asarray([candidates[official]["features"][name] for name in names])
        deltas = {official: 0.0}
        for candidate_name, candidate in candidates.items():
            if candidate_name == official:
                continue
            vector = np.asarray([candidate["features"][name] for name in names])
            deltas[candidate_name] = float(np.r_[1.0, (vector - base - mean) / scale] @ weights)
        challenger = max((name for name in candidates if name != official), key=deltas.get)
        chosen = challenger if deltas[challenger] > threshold else official
        audio = row["audio_path"]
        output.append(
            {
                "benchmark": "LongNeedle-ESC50",
                "source": "longneedle",
                "source_index": index,
                "audio_path": audio,
                "audio_group": Path(audio).name,
                "caption": row["caption"],
                "annotations": row["annotations"],
                "duration": float(row.get("duration", 60.0)),
                "incumbent_name": f"e004_{chosen}",
                "incumbent_prediction": candidates[chosen]["prediction"],
                "stored_incumbent_iou": candidates[chosen].get("iou"),
            }
        )
    return output


def _bucket(value: float, low: float, high: float) -> str:
    return "low" if value <= low else ("medium" if value <= high else "high")


def _stratum(row: dict) -> str:
    duration = _duration(row)
    truth = normalize_intervals(row["annotations"], duration)
    incumbent = normalize_intervals(row["incumbent_prediction"], duration)
    midpoint = (
        sum((start + end) / 2 for start, end in truth) / len(truth) / duration if truth else 0.5
    )
    coverage = sum(end - start for start, end in incumbent) / duration
    count = "empty" if not incumbent else ("single" if len(incumbent) == 1 else "multi")
    query_length = len(str(row["caption"]).split())
    return "|".join(
        (count, _bucket(coverage, 0.1, 0.3), _bucket(midpoint, 1 / 3, 2 / 3), _bucket(query_length, 5, 10))
    )


def choose_audio_groups(rows: Sequence[dict], size: int, seed: int) -> list[dict]:
    """Deterministic round-robin stratification without splitting audio groups."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_audio_group(row)].append(dict(row, stratum=_stratum(row)))
    buckets: dict[str, list[str]] = defaultdict(list)
    for group, members in groups.items():
        buckets[Counter(item["stratum"] for item in members).most_common(1)[0][0]].append(group)
    for names in buckets.values():
        names.sort(key=lambda name: _stable_hash(seed, name))
    selected: list[str] = []
    total = 0
    active = sorted(buckets)
    while active and total < size:
        remaining = []
        for bucket in active:
            names = buckets[bucket]
            while names:
                name = names.pop(0)
                if total + len(groups[name]) <= size:
                    selected.append(name)
                    total += len(groups[name])
                    break
            if names:
                remaining.append(bucket)
            if total == size:
                break
        active = remaining
    if total != size:
        for name in sorted((name for name in groups if name not in selected), key=lambda x: _stable_hash(seed + 1, x)):
            if total + len(groups[name]) <= size:
                selected.append(name)
                total += len(groups[name])
            if total == size:
                break
    if total != size:
        raise RuntimeError(f"cannot select exactly {size} rows without splitting audio groups")
    output = [row for group in selected for row in groups[group]]
    return sorted(output, key=lambda row: row["source_index"])


def _excluded_audio(paths: Sequence[str]) -> set[str]:
    excluded = set()
    for raw_path in paths:
        path = Path(raw_path)
        rows = _jsonl(path) if path.suffix == ".jsonl" else _json(path)
        excluded.update(_audio_group(row) for row in rows)
    return excluded


def command_prepare(args) -> None:
    config = _json(args.config)
    run_dir = Path(config["run_dir"])
    prepared = run_dir / "data"
    dev_rows = {}
    for source in ("clotho", "sc"):
        spec = config["development"][source]
        dev_rows[source] = _paired_rows(
            Path(spec["manifest"]), Path(spec["incumbent"]), source,
            spec["incumbent_name"], spec["benchmark"], Path(spec["audio_dir"]),
        )
    dev_rows["longneedle"] = _longneedle_rows(config)
    for source, rows in dev_rows.items():
        _write_json(prepared / f"{source}_dev.json", rows)

    snapshot = {"git_commit": _git_hash(), "seed": config["seed"], "inputs": {}}
    for benchmark, spec in config["pilots"].items():
        rows = _paired_rows(
            Path(spec["manifest"]), Path(spec["incumbent"]), benchmark,
            spec["incumbent_name"], spec["benchmark"], Path(spec["audio_dir"]),
        )
        stored = [float(row["stored_incumbent_iou"]) for row in rows]
        full_miou = float(np.mean(stored) * 100)
        if abs(full_miou - float(spec["expected_full_miou"])) > 1e-9:
            raise ValueError(f"{benchmark} locked baseline mismatch: {full_miou}")
        excluded = _excluded_audio(spec.get("exclude", []))
        excluded.update(_audio_group(row) for source_rows in dev_rows.values() for row in source_rows)
        available = [row for row in rows if _audio_group(row) not in excluded]
        pilot = choose_audio_groups(available, int(spec["size"]), int(config["seed"]))
        smoke = choose_audio_groups(pilot, int(spec["smoke_size"]), int(config["seed"]) + 17)
        _write_json(prepared / f"{benchmark}_pilot.json", pilot)
        _write_json(prepared / f"{benchmark}_smoke.json", smoke)
        snapshot["inputs"][benchmark] = {
            "incumbent": str(Path(spec["incumbent"]).resolve()),
            "incumbent_sha256": file_sha256(Path(spec["incumbent"])),
            "full_rows": len(rows),
            "full_mIoU_percent": full_miou,
            "excluded_audio_groups": len(excluded),
            "pilot_rows": len(pilot),
            "pilot_audio_groups": len({_audio_group(row) for row in pilot}),
            "pilot_sha256": file_sha256(prepared / f"{benchmark}_pilot.json"),
            "smoke_sha256": file_sha256(prepared / f"{benchmark}_smoke.json"),
        }
    snapshot["config_sha256"] = file_sha256(args.config)
    _write_json(run_dir / "frozen_input_snapshot.json", snapshot)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


def _export_id(args, manifest: Path) -> str:
    digest = hashlib.sha256()
    base_metadata = [
        candidate for name in (
            "config.json", "preprocessor_config.json", "processor_config.json",
            "model.safetensors.index.json",
        ) if (candidate := args.base / name).is_file()
    ]
    if not base_metadata:
        raise ValueError(f"base model has no identity metadata: {args.base}")
    for value in (
        file_sha256(manifest),
        hashlib.sha256("".join(file_sha256(path) for path in base_metadata).encode()).hexdigest(),
        artifact_sha256(args.adapter),
        artifact_sha256(args.spantool_checkpoint), file_sha256(Path(__file__)),
    ):
        digest.update(value.encode())
    return digest.hexdigest()[:20]


def command_export(args) -> None:
    import torch
    from peft import PeftModel
    from spantool_runtime import (
        load_spantool_checkpoint, load_wave, move_model_inputs, prepare_spantool_input,
        select_audio_states,
    )
    from spotsound import AudioFlamingo3ForTemporalConditionalGeneration, AudioFlamingo3SpanToolProcessor

    rows = _json(args.manifest)
    expected = _export_id(args, args.manifest)
    completed = {}
    if args.output.is_file():
        for row in _jsonl(args.output):
            if row.get("export_id") != expected:
                raise ValueError("existing logit file has a different export_id")
            index = int(row["source_index"])
            if index in completed:
                raise ValueError(f"duplicate completed source_index {index}")
            completed[index] = row
    device = torch.device("cuda")
    processor = AudioFlamingo3SpanToolProcessor.from_pretrained(args.base)
    model = AudioFlamingo3ForTemporalConditionalGeneration.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter, torch_device="cpu", is_trainable=False).merge_and_unload()
    model = model.eval().to(device)
    head, payload = load_spantool_checkpoint(args.spantool_checkpoint, device)
    head = head.eval()
    layers = [int(value) for value in payload.get("audio_layers", [payload["audio_layer"]])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for ordinal, row in enumerate(rows):
            index = int(row["source_index"])
            if index in completed:
                continue
            wave = load_wave(args.audio_dir, row)
            duration = len(wave) / 16000.0
            if row.get("duration") is not None and abs(duration - float(row["duration"])) > 1e-3:
                raise ValueError(f"duration mismatch for source_index {index}: {duration} vs {row['duration']}")
            inputs = move_model_inputs(
                prepare_spantool_input(processor, wave, row["caption"]), device, model.dtype
            )
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
                states, frame_mask = select_audio_states(
                    outputs, layers, inputs["input_ids"], model.config.audio_token_id
                )
                head_output = head(states.float(), frame_mask, torch.tensor([duration], device=device))
            primary_steps = int(head_output["primary_mask"][0].sum())
            fine_steps = int(frame_mask[0].sum())
            audio_path = resolve_audio_path(args.audio_dir, row)
            record = dict(row)
            record.update(
                {
                    "export_id": expected,
                    "duration": duration,
                    "primary_steps": primary_steps,
                    "fine_steps": fine_steps,
                    "occupancy_logits": head_output["occupancy_logits"][0, :primary_steps].float().cpu().tolist(),
                    "inference_seconds": time.perf_counter() - started,
                    "audio_sha256": file_sha256(audio_path),
                    "input_hash": hashlib.sha256(
                        json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + file_sha256(audio_path).encode()
                    ).hexdigest(),
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed[index] = record
            print(json.dumps({"ordinal": ordinal, "source_index": index, "seconds": record["inference_seconds"]}), flush=True)


def _targets(record: dict) -> np.ndarray:
    bins, duration = len(record["occupancy_logits"]), float(record["duration"])
    step = duration / bins
    target = np.zeros(bins, dtype=np.float64)
    for start, end in normalize_intervals(record["annotations"], duration):
        for index in range(bins):
            overlap = max(0.0, min(end, (index + 1) * step) - max(start, index * step))
            target[index] = max(target[index], overlap / step)
    return target


def fit_calibrator(records: Sequence[dict]) -> dict[str, float]:
    from scipy.optimize import minimize

    if not records:
        raise ValueError("cannot calibrate an empty record set")
    by_source: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source"])].append(
            (np.asarray(record["occupancy_logits"], dtype=np.float64), _targets(record))
        )
    logits, targets, weights = [], [], []
    for source in sorted(by_source):
        source_bins = sum(len(item[0]) for item in by_source[source])
        for source_logits, source_targets in by_source[source]:
            logits.append(source_logits)
            targets.append(source_targets)
            weights.append(np.full(len(source_logits), 1.0 / len(by_source) / source_bins))
    x, y, w = np.concatenate(logits), np.concatenate(targets), np.concatenate(weights)

    def objective(parameters):
        log_temperature, bias = parameters
        temperature = math.exp(log_temperature)
        z = np.clip((x + bias) / temperature, -60, 60)
        probability = 1.0 / (1.0 + np.exp(-z))
        loss = -np.sum(w * (y * np.log(probability + 1e-12) + (1 - y) * np.log(1 - probability + 1e-12)))
        error = w * (probability - y)
        gradient = np.asarray([np.sum(error * -z), np.sum(error / temperature)])
        return float(loss), gradient

    result = minimize(objective, np.asarray([0.0, 0.0]), jac=True, method="L-BFGS-B", bounds=[(-4, 4), (-20, 20)])
    if not result.success:
        raise RuntimeError(f"calibration failed: {result.message}")
    return {"temperature": float(math.exp(result.x[0])), "bias": float(result.x[1]), "loss": float(result.fun)}


def bootstrap_calibrators(records: Sequence[dict], samples: int, seed: int) -> list[dict]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[str(record["source"])][_audio_group(record)].append(record)
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(samples):
        sampled = []
        for source in sorted(grouped):
            names = sorted(grouped[source])
            chosen = rng.choice(names, size=len(names), replace=True)
            sampled.extend(row for name in chosen for row in grouped[source][str(name)])
        output.append(fit_calibrator(sampled))
    return output


def _probabilities(record: dict, calibration: dict) -> np.ndarray:
    primary = calibrate_occupancy(
        record["occupancy_logits"], calibration["temperature"], calibration["bias"]
    )
    return resample_probabilities(primary, int(record["fine_steps"]))


def _decode(
    record: dict,
    calibration: dict,
    ensemble: Sequence[dict],
    radius: float,
    threshold: float | None = None,
    prepared_probabilities: tuple[np.ndarray, list[np.ndarray]] | None = None,
):
    if prepared_probabilities is None:
        probabilities = _probabilities(record, calibration)
        bootstrap = [_probabilities(record, item) for item in ensemble]
    else:
        probabilities, bootstrap = prepared_probabilities
    optimisation = probabilities if threshold is None else (probabilities >= threshold).astype(np.float64)
    return dinkelbach_trust_region_decode(
        optimisation,
        record["incumbent_prediction"],
        float(record["duration"]),
        radius,
        decision_probabilities=probabilities,
        bootstrap_probabilities=bootstrap,
    )


def _metrics(records: Sequence[dict], predictions: Sequence[Sequence[Sequence[float]]]) -> dict:
    incumbent_ious, selected_ious = [], []
    catastrophic = 0
    for record, prediction in zip(records, predictions):
        truth = record["annotations"]
        incumbent = temporal_set_iou(truth, record["incumbent_prediction"])
        selected = temporal_set_iou(truth, prediction)
        incumbent_ious.append(incumbent)
        selected_ious.append(selected)
        catastrophic += selected - incumbent <= -0.5
    delta = np.asarray(selected_ious) - np.asarray(incumbent_ious)
    return {
        "incumbent_mIoU_percent": float(np.mean(incumbent_ious) * 100),
        "selected_mIoU_percent": float(np.mean(selected_ious) * 100),
        "delta_mIoU_points": float(np.mean(delta) * 100),
        "catastrophic": int(catastrophic),
        "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
    }


def _group_bootstrap_difference(
    records: Sequence[dict], first: Sequence[float], second: Sequence[float], samples: int, seed: int
) -> list[float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record, left, right in zip(records, first, second):
        groups[_audio_group(record)].append(float(left - right))
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        chosen = rng.choice(names, size=len(names), replace=True)
        estimates.append(np.mean([value for name in chosen for value in groups[str(name)]]) * 100)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def command_select(args) -> None:
    config = _json(args.config)
    records_by_source = {path.stem.replace("_logits", ""): _jsonl(path) for path in args.logits}
    required = {"longneedle", "sc", "clotho"}
    if set(records_by_source) != required:
        raise ValueError(f"development logits must be exactly {sorted(required)}")
    radii = [float(value) for value in config["radii_seconds"]]
    thresholds = [float(value) for value in config["fixed_thresholds"]]
    candidates = {}
    diagnostics: dict[float, list[dict]] = {radius: [] for radius in radii}
    calibrator_samples = int(config["bootstrap_calibrators"])
    for radius in radii:
        candidates[str(radius)] = {"sources": {}, "pooled_dinkelbach": [], "pooled_threshold": [], "pooled_records": []}
    for fold_index, heldout in enumerate(sorted(required)):
        training = [row for source, rows in records_by_source.items() if source != heldout for row in rows]
        calibration = fit_calibrator(training)
        ensemble = bootstrap_calibrators(training, calibrator_samples, int(config["seed"]) + fold_index * 1000)
        heldout_rows = records_by_source[heldout]
        fold_results = {
            radius: {"dinkelbach": [], "thresholds": {threshold: [] for threshold in thresholds}}
            for radius in radii
        }
        for record in heldout_rows:
            prepared_probabilities = (
                _probabilities(record, calibration),
                [_probabilities(record, item) for item in ensemble],
            )
            for radius in radii:
                fold_results[radius]["dinkelbach"].append(
                    _decode(
                        record, calibration, ensemble, radius,
                        prepared_probabilities=prepared_probabilities,
                    )
                )
                for threshold in thresholds:
                    fold_results[radius]["thresholds"][threshold].append(
                        _decode(
                            record, calibration, ensemble, radius, threshold,
                            prepared_probabilities=prepared_probabilities,
                        )
                    )
        for radius in radii:
            d_results = fold_results[radius]["dinkelbach"]
            d_predictions = [result.selected for result in d_results]
            threshold_runs = {}
            for threshold in thresholds:
                results = fold_results[radius]["thresholds"][threshold]
                threshold_runs[threshold] = (results, _metrics(heldout_rows, [item.selected for item in results]))
            best_threshold = max(thresholds, key=lambda value: threshold_runs[value][1]["delta_mIoU_points"])
            t_results, t_metrics = threshold_runs[best_threshold]
            d_metrics = _metrics(heldout_rows, d_predictions)
            key = str(radius)
            candidates[key]["sources"][heldout] = {
                "dinkelbach": d_metrics,
                "best_fixed_threshold": best_threshold,
                "fixed_threshold": t_metrics,
                "dinkelbach_switches": sum(item.switch for item in d_results),
                "threshold_switches": sum(item.switch for item in t_results),
            }
            for record, d_result, t_result in zip(heldout_rows, d_results, t_results):
                incumbent_iou = temporal_set_iou(record["annotations"], record["incumbent_prediction"])
                d_iou = temporal_set_iou(record["annotations"], d_result.selected)
                t_iou = temporal_set_iou(record["annotations"], t_result.selected)
                diagnostics[radius].append(
                    {
                        "source": heldout,
                        "source_index": int(record["source_index"]),
                        "audio_group": _audio_group(record),
                        "query": record["caption"],
                        "ground_truth": record["annotations"],
                        "incumbent_prediction": record["incumbent_prediction"],
                        "dinkelbach_candidate": d_result.candidate,
                        "dinkelbach_prediction": d_result.selected,
                        "fixed_threshold_prediction": t_result.selected,
                        "incumbent_iou": incumbent_iou,
                        "dinkelbach_iou": d_iou,
                        "fixed_threshold_iou": t_iou,
                        "delta_iou": d_iou - incumbent_iou,
                        "dinkelbach_minus_threshold_iou": d_iou - t_iou,
                        "dinkelbach_switch": d_result.switch,
                        "fixed_threshold_switch": t_result.switch,
                        "gain_lower_quantile": d_result.gain_lower_quantile,
                        "abstain_reason": d_result.abstain_reason,
                        "radius_seconds": radius,
                    }
                )
            candidates[key]["pooled_records"].extend(heldout_rows)
            candidates[key]["pooled_dinkelbach"].extend(d_predictions)
            candidates[key]["pooled_threshold"].extend(item.selected for item in t_results)
    passing = []
    for radius in radii:
        entry = candidates[str(radius)]
        records = entry.pop("pooled_records")
        d_predictions = entry.pop("pooled_dinkelbach")
        t_predictions = entry.pop("pooled_threshold")
        d_ious = [temporal_set_iou(row["annotations"], pred) for row, pred in zip(records, d_predictions)]
        t_ious = [temporal_set_iou(row["annotations"], pred) for row, pred in zip(records, t_predictions)]
        entry["combined_dinkelbach"] = _metrics(records, d_predictions)
        entry["combined_fixed_threshold"] = _metrics(records, t_predictions)
        entry["mechanism_delta_points"] = float((np.mean(d_ious) - np.mean(t_ious)) * 100)
        entry["mechanism_bootstrap_95ci"] = _group_bootstrap_difference(
            records, d_ious, t_ious, int(config["evaluation_bootstrap_samples"]), int(config["seed"])
        )
        entry["gate_passed"] = (
            all(value["dinkelbach"]["delta_mIoU_points"] >= 0 and value["dinkelbach"]["catastrophic"] == 0 for value in entry["sources"].values())
            and entry["combined_dinkelbach"]["delta_mIoU_points"] > 0
            and entry["mechanism_bootstrap_95ci"][0] > 0
        )
        if entry["gate_passed"]:
            passing.append(radius)
    selected_radius = None
    if passing:
        best_gain = max(candidates[str(radius)]["combined_dinkelbach"]["delta_mIoU_points"] for radius in passing)
        selected_radius = min(
            radius for radius in passing
            if best_gain - candidates[str(radius)]["combined_dinkelbach"]["delta_mIoU_points"] <= 0.02
        )
    all_records = [row for source in sorted(required) for row in records_by_source[source]]
    final_calibration = fit_calibrator(all_records)
    final_ensemble = bootstrap_calibrators(all_records, calibrator_samples, int(config["seed"]) + 9000)
    best_threshold = None
    if selected_radius is not None:
        threshold_metrics = {}
        for threshold in thresholds:
            results = []
            for row in all_records:
                prepared_probabilities = (
                    _probabilities(row, final_calibration),
                    [_probabilities(row, item) for item in final_ensemble],
                )
                results.append(
                    _decode(
                        row, final_calibration, final_ensemble, selected_radius, threshold,
                        prepared_probabilities=prepared_probabilities,
                    )
                )
            threshold_metrics[threshold] = _metrics(all_records, [item.selected for item in results])
        best_threshold = max(thresholds, key=lambda value: threshold_metrics[value]["delta_mIoU_points"])
    artifact = {
        "format_version": 1,
        "development_gate_passed": selected_radius is not None,
        "selected_radius_seconds": selected_radius,
        "selected_fixed_threshold": best_threshold,
        "calibration": final_calibration,
        "bootstrap_calibrations": final_ensemble,
        "seed": config["seed"],
        "development_logit_hashes": {source: file_sha256(path) for source, path in zip(records_by_source, args.logits)},
        "config_sha256": file_sha256(args.config),
        "code_hash": _git_hash(),
    }
    report = {"development_gate_passed": selected_radius is not None, "selected_radius_seconds": selected_radius, "radii": candidates}
    _write_json(args.output, artifact)
    _write_json(args.report, report)
    diagnostic_radius = max(
        radii, key=lambda radius: candidates[str(radius)]["combined_dinkelbach"]["delta_mIoU_points"]
    )
    if args.diagnostics_dir is not None:
        for radius in radii:
            _write_jsonl(args.diagnostics_dir / f"radius_{radius}.jsonl", diagnostics[radius])
        failure_rows = []
        for row in diagnostics[diagnostic_radius]:
            category = None
            if row["delta_iou"] <= -0.5:
                category = "catastrophic_wrong_window"
            elif row["dinkelbach_switch"] and row["delta_iou"] < 0:
                category = "false_switch"
            elif not row["dinkelbach_switch"] and temporal_set_iou(
                row["ground_truth"], row["dinkelbach_candidate"]
            ) > row["incumbent_iou"] + 1e-12:
                category = "missed_rescue"
            if category:
                failure_rows.append(dict(row, failure_category=category))
        failure_rows.sort(key=lambda row: (row["delta_iou"], row["source"], row["source_index"]))
        _write_jsonl(args.diagnostics_dir / "development_failures.jsonl", failure_rows)
        report["diagnostic_radius_seconds"] = diagnostic_radius
        report["diagnostics_sha256"] = file_sha256(args.diagnostics_dir / f"radius_{diagnostic_radius}.jsonl")
        report["failures_sha256"] = file_sha256(args.diagnostics_dir / "development_failures.jsonl")
        _write_json(args.report, report)
    if args.promotion is not None:
        _write_json(
            args.promotion,
            {
                "decision": "READY_FOR_SMOKE" if selected_radius is not None else "DO_NOT_PROMOTE",
                "stage": "development",
                "public_smoke_started": False,
                "public_pilot_started": False,
                "selected_radius_seconds": selected_radius,
                "diagnostic_radius_seconds": diagnostic_radius,
                "reasons": [] if selected_radius is not None else ["development_gate_failed"],
                "report_sha256": file_sha256(args.report),
            },
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_logit_alignment(manifest: Sequence[dict], records: Sequence[dict]) -> None:
    left, right = _strict_index(manifest, "manifest"), _strict_index(records, "logits")
    if set(left) != set(right):
        raise ValueError("manifest/logit source_index sets differ")
    for index in left:
        if _audio_group(left[index]) != _audio_group(right[index]):
            raise ValueError(f"audio mismatch at source_index {index}")
        if abs(float(left[index].get("duration", right[index]["duration"])) - float(right[index]["duration"])) > 1e-3:
            raise ValueError(f"duration mismatch at source_index {index}")


def command_decode(args) -> None:
    artifact = _json(args.calibrator)
    if not artifact["development_gate_passed"]:
        raise RuntimeError("development gate failed; pilot decoding is forbidden")
    manifest, records = _json(args.manifest), _jsonl(args.logits)
    _validate_logit_alignment(manifest, records)
    by_index = _strict_index(records, "logits")
    ordered = [by_index[int(row["source_index"])] for row in manifest]
    radius = float(artifact["selected_radius_seconds"])
    threshold = float(artifact["selected_fixed_threshold"])
    config_hash = artifact["config_sha256"]
    output = []
    for record in ordered:
        prepared_probabilities = (
            _probabilities(record, artifact["calibration"]),
            [_probabilities(record, item) for item in artifact["bootstrap_calibrations"]],
        )
        result = _decode(
            record, artifact["calibration"], artifact["bootstrap_calibrations"], radius,
            prepared_probabilities=prepared_probabilities,
        )
        threshold_result = _decode(
            record, artifact["calibration"], artifact["bootstrap_calibrations"], radius, threshold,
            prepared_probabilities=prepared_probabilities,
        )
        incumbent_iou = temporal_set_iou(record["annotations"], result.incumbent)
        candidate_iou = temporal_set_iou(record["annotations"], result.candidate)
        selected_iou = temporal_set_iou(record["annotations"], result.selected)
        threshold_iou = temporal_set_iou(record["annotations"], threshold_result.selected)
        output.append(
            {
                "benchmark": record["benchmark"],
                "source_index": int(record["source_index"]),
                "audio_group": _audio_group(record),
                "audio": record["audio_path"],
                "query": record["caption"],
                "duration": float(record["duration"]),
                "ground_truth": record["annotations"],
                "incumbent_name": record["incumbent_name"],
                "incumbent_prediction": result.incumbent,
                "candidate_prediction": result.candidate,
                "selected_prediction": result.selected,
                "incumbent_iou": incumbent_iou,
                "candidate_iou": candidate_iou,
                "selected_iou": selected_iou,
                "delta_iou": selected_iou - incumbent_iou,
                "posterior_iou_incumbent": result.posterior_iou_incumbent,
                "posterior_iou_candidate": result.posterior_iou_candidate,
                "gain_lower_quantile": result.gain_lower_quantile,
                "radius_seconds": radius,
                "dinkelbach_iterations": result.iterations,
                "switch": result.switch,
                "selected_candidate": "nova_mbr" if result.switch else record["incumbent_name"],
                "abstain_reason": result.abstain_reason,
                "fixed_threshold": threshold,
                "fixed_threshold_prediction": threshold_result.selected,
                "fixed_threshold_iou": threshold_iou,
                "code_hash": artifact["code_hash"],
                "checkpoint_hash": record["export_id"],
                "input_hash": record["input_hash"],
                "config_hash": config_hash,
            }
        )
    _write_jsonl(args.output, output)
    print(json.dumps({"rows": len(output), "switches": sum(row["switch"] for row in output), "sha256": file_sha256(args.output)}, indent=2))


def command_evaluate(args) -> None:
    rows = _jsonl(args.predictions)
    if not rows:
        raise ValueError("no predictions")
    _strict_index(rows, "predictions")
    incumbent = np.asarray([row["incumbent_iou"] for row in rows])
    selected = np.asarray([row["selected_iou"] for row in rows])
    threshold = np.asarray([row["fixed_threshold_iou"] for row in rows])
    delta = selected - incumbent
    event_f1 = [event_f1_iou(row["ground_truth"], row["selected_prediction"], 0.5)["f1"] for row in rows]
    shifts = [
        abs(new - old)
        for row in rows
        for old_interval, new_interval in zip(row["incumbent_prediction"], row["selected_prediction"])
        for old, new in zip(old_interval, new_interval)
    ]
    report = {
        "benchmark": rows[0]["benchmark"],
        "rows": len(rows),
        "audio_groups": len({_audio_group(row) for row in rows}),
        "incumbent_mIoU_percent": float(incumbent.mean() * 100),
        "nova_mbr_mIoU_percent": float(selected.mean() * 100),
        "delta_mIoU_points": float(delta.mean() * 100),
        "delta_mIoU_audio_group_bootstrap_95ci": _group_bootstrap_difference(
            rows, selected, incumbent, args.bootstrap_samples, args.seed
        ),
        "fixed_threshold_mIoU_percent": float(threshold.mean() * 100),
        "dinkelbach_minus_threshold_points": float((selected.mean() - threshold.mean()) * 100),
        "R1@0.3_percent": float((selected >= 0.3).mean() * 100),
        "R1@0.5_percent": float((selected >= 0.5).mean() * 100),
        "R1@0.7_percent": float((selected >= 0.7).mean() * 100),
        "event_F1@0.5_percent": float(np.mean(event_f1) * 100),
        "wins": int((delta > 1e-12).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "losses": int((delta < -1e-12).sum()),
        "new_catastrophic_regressions": int((delta <= -0.5).sum()),
        "switches": int(sum(row["switch"] for row in rows)),
        "switch_rate": float(np.mean([row["switch"] for row in rows])),
        "abstain_reasons": dict(Counter(row["abstain_reason"] for row in rows if row["abstain_reason"])),
        "boundary_shift_seconds": {
            "mean": float(np.mean(shifts)) if shifts else 0.0,
            "p95": float(np.quantile(shifts, 0.95)) if shifts else 0.0,
            "max": float(max(shifts)) if shifts else 0.0,
        },
        "alignment_valid": True,
        "promotion_gate": {
            "positive_delta": float(delta.mean()) > 0,
            "zero_catastrophic": int((delta <= -0.5).sum()) == 0,
            "beats_fixed_threshold": float(selected.mean()) > float(threshold.mean()),
        },
        "predictions_sha256": file_sha256(args.predictions),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
    }
    report["promotion_gate"]["passed"] = all(report["promotion_gate"].values())
    failures = []
    for row in rows:
        category = None
        if row["delta_iou"] <= -0.5:
            category = "catastrophic_wrong_window"
        elif row["switch"] and row["delta_iou"] < 0:
            category = "false_switch"
        elif not row["switch"] and row["candidate_iou"] > row["incumbent_iou"] + 1e-12:
            category = "missed_rescue"
        if category:
            failures.append(dict(row, failure_category=category))
    failures.sort(key=lambda row: (row["delta_iou"], row["source_index"]))
    _write_json(args.report, report)
    _write_jsonl(args.failures, failures)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_promote(args) -> None:
    reports = [_json(path) for path in args.reports]
    by_benchmark = {row["benchmark"]: row for row in reports}
    required = {"SpotSound-Bench", "Clotho-Moment"}
    passed = set(by_benchmark) == required and all(row["promotion_gate"]["passed"] for row in reports)
    reasons = []
    if set(by_benchmark) != required:
        reasons.append("missing_benchmark_report")
    for name, report in by_benchmark.items():
        if not report["promotion_gate"]["passed"]:
            reasons.append(f"{name}:pilot_gate_failed")
    payload = {
        "decision": "PROMOTE" if passed else "DO_NOT_PROMOTE",
        "full_evaluation_started": False,
        "reasons": reasons,
        "reports": {name: {"delta_mIoU_points": row["delta_mIoU_points"], "gate": row["promotion_gate"]} for name, row in by_benchmark.items()},
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.set_defaults(function=command_prepare)

    export = subparsers.add_parser("export")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--audio-dir", type=Path, required=True)
    export.add_argument("--base", type=Path, required=True)
    export.add_argument("--adapter", type=Path, required=True)
    export.add_argument("--spantool-checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.set_defaults(function=command_export)

    select = subparsers.add_parser("select")
    select.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    select.add_argument("--logits", type=Path, action="append", required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--report", type=Path, required=True)
    select.add_argument("--diagnostics-dir", type=Path)
    select.add_argument("--promotion", type=Path)
    select.set_defaults(function=command_select)

    decode = subparsers.add_parser("decode")
    decode.add_argument("--manifest", type=Path, required=True)
    decode.add_argument("--logits", type=Path, required=True)
    decode.add_argument("--calibrator", type=Path, required=True)
    decode.add_argument("--output", type=Path, required=True)
    decode.set_defaults(function=command_decode)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--report", type=Path, required=True)
    evaluate.add_argument("--failures", type=Path, required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=10000)
    evaluate.add_argument("--seed", type=int, default=20260903)
    evaluate.set_defaults(function=command_evaluate)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--reports", type=Path, action="append", required=True)
    promote.add_argument("--output", type=Path, required=True)
    promote.set_defaults(function=command_promote)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
