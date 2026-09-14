"""CPU-only identity and summary helpers for frozen RelTwin transfer evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from interval_metrics import normalize_intervals, temporal_set_iou, soft_precision_recall, event_f1_iou


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    if content and not content.endswith("\n"):
        raise ValueError(f"Interrupted/incomplete JSONL tail; preserve and inspect: {path}")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    ids = [r["index"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate prediction indices: {path}")
    return sorted(rows, key=lambda r: r["index"])


def validate_rows(rows, manifest, *, complete=False):
    seen = set()
    for row in rows:
        i = row["index"]
        if not isinstance(i, int) or i in seen or not 0 <= i < len(manifest):
            raise ValueError(f"Invalid/duplicate index {i}")
        seen.add(i)
        item = manifest[i]
        if row.get("status") != "ok" or row["query"] != item["caption"]:
            raise ValueError(f"Query/status mismatch at {i}")
        if row["ground_truth"] != item["annotations"]:
            raise ValueError(f"Target mismatch at {i}")
        if Path(row["audio"]).name != Path(item["audio_path"]).name:
            raise ValueError(f"Audio mismatch at {i}")
        duration = float(row["duration_seconds"])
        if not math.isfinite(duration) or abs(duration - item["decoded_duration"]) > 1 / 16000 + 1e-6:
            raise ValueError(f"Duration mismatch at {i}")
        for start, end in row["prediction"]:
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + 1e-6):
                raise ValueError(f"Invalid predicted interval at {i}")
        expected = temporal_set_iou(item["annotations"], row["prediction"])
        if not math.isclose(expected, row["iou"], abs_tol=1e-9):
            raise ValueError(f"Recorded IoU mismatch at {i}")
    if complete and seen != set(range(len(manifest))):
        raise ValueError(f"Incomplete predictions: {len(seen)}/{len(manifest)}")
    return rows


def cluster_interval(values, groups, *, draws=20000, seed=20260914):
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=float))
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 100):
        selected = rng.integers(0, len(unique), size=(min(100, draws-start), len(unique)))
        samples.extend((sums[selected].sum(1) / counts[selected].sum(1)).tolist())
    return (100 * np.quantile(samples, [.025, .975])).tolist()


def row_metrics(row):
    duration = row["duration_seconds"]
    gt = normalize_intervals(row["ground_truth"], duration)
    pred = normalize_intervals(row["prediction"], duration)
    return {
        "set_iou": temporal_set_iou(gt, pred),
        "auto_aeg_mean_best_iou": soft_precision_recall(gt, pred)[1],
        "event_f1_iou_0.5": event_f1_iou(gt, pred, .5)["f1"],
    }


def summarize(rows, manifest):
    validate_rows(rows, manifest, complete=True)
    rows = sorted(rows, key=lambda row: row["index"])
    values = [row_metrics(row) for row in rows]
    ious = np.asarray([v["set_iou"] for v in values])
    groups = [manifest[r["index"]]["audio_group"] for r in rows]
    return {
        "count": len(rows), "audio_groups": len(set(groups)), "scale": "percent",
        "mIoU": float(100 * ious.mean()),
        "released_endpoint_mIoU": float(100 * np.mean([r["iou"] for r in rows])),
        **{f"R1@{t}": float(100 * (ious >= t).mean()) for t in (.3, .5, .7)},
        "event_F1_IoU@0.5": float(100 * np.mean([v["event_f1_iou_0.5"] for v in values])),
        "auto_aeg_mean_best_mIoU": float(100 * np.mean([v["auto_aeg_mean_best_iou"] for v in values])),
        "audio_group_mIoU_95CI": cluster_interval(ious, groups),
        "empty_predictions": sum(not r["prediction"] for r in rows),
        "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in rows])),
        "bootstrap_draws": 20000, "bootstrap_seed": 20260914,
        "endpoint_policy": "clip GT and prediction to decoded audio; also report released endpoints",
    }


def paired_summary(ours, control, manifest):
    validate_rows(ours, manifest, complete=True)
    validate_rows(control, manifest, complete=True)
    ours = sorted(ours, key=lambda r: r["index"])
    control = sorted(control, key=lambda r: r["index"])
    delta = np.array([row_metrics(a)["set_iou"] - row_metrics(b)["set_iou"] for a, b in zip(ours, control)])
    groups = [m["audio_group"] for m in manifest]
    return {"delta_mIoU": float(100 * delta.mean()), "audio_group_95CI": cluster_interval(delta, groups),
            "wins": int((delta > 1e-9).sum()), "ties": int((abs(delta) <= 1e-9).sum()),
            "losses": int((delta < -1e-9).sum()), "drops_at_least_0.5": int((delta <= -.5).sum())}
