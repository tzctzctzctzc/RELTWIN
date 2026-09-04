import json

import numpy as np
import pytest

from run_metric_iou_pilot import (
    _paired_rows,
    _validate_logit_alignment,
    choose_audio_groups,
    fit_calibrator,
)


def _row(index, audio, prediction=((1.0, 2.0),)):
    return {
        "benchmark": "test",
        "source": "test",
        "source_index": index,
        "audio_path": audio,
        "audio_group": audio,
        "caption": "short sound",
        "annotations": [[1.0, 2.0]],
        "duration": 4.0,
        "incumbent_name": "baseline",
        "incumbent_prediction": list(prediction),
    }


def test_grouped_selection_is_exact_deterministic_and_leak_free():
    rows = [_row(index, f"audio_{index // 2}.wav") for index in range(12)]
    first = choose_audio_groups(rows, 6, 20260903)
    second = choose_audio_groups(rows, 6, 20260903)
    assert first == second
    assert len(first) == 6
    selected_groups = {row["audio_group"] for row in first}
    assert all(sum(row["audio_group"] == group for row in first) == 2 for group in selected_groups)


def test_grouped_selection_fails_when_exact_size_is_impossible():
    rows = [_row(index, "same.wav") for index in range(3)]
    with pytest.raises(RuntimeError, match="exactly 2"):
        choose_audio_groups(rows, 2, 1)


def test_paired_rows_reject_missing_or_duplicate_indices(tmp_path):
    manifest = tmp_path / "manifest.json"
    incumbent = tmp_path / "predictions.jsonl"
    manifest.write_text(json.dumps([{"audio_path": "a.wav", "caption": "q", "annotations": []}]))
    incumbent.write_text(
        json.dumps({"index": 1, "audio": "a.wav", "prediction": [], "iou": 0.0}) + "\n"
    )
    with pytest.raises(ValueError, match="alignment mismatch"):
        _paired_rows(manifest, incumbent, "x", "base", "bench")


def test_alignment_rejects_audio_or_duration_mismatch():
    manifest = [_row(0, "a.wav")]
    logits = [dict(_row(0, "b.wav"), occupancy_logits=[0.0], fine_steps=1)]
    with pytest.raises(ValueError, match="audio mismatch"):
        _validate_logit_alignment(manifest, logits)
    logits = [dict(_row(0, "a.wav"), duration=5.0, occupancy_logits=[0.0], fine_steps=1)]
    with pytest.raises(ValueError, match="duration mismatch"):
        _validate_logit_alignment(manifest, logits)


def test_calibrator_learns_direction_on_separable_logits():
    records = []
    for source in ("one", "two"):
        for index in range(2):
            records.append(
                {
                    "source": source,
                    "audio_group": f"{source}_{index}",
                    "duration": 4.0,
                    "annotations": [[0.0, 2.0]],
                    "occupancy_logits": [2.0, 2.0, -2.0, -2.0],
                }
            )
    calibration = fit_calibrator(records)
    assert calibration["temperature"] > 0
    positive = 1 / (1 + np.exp(-(2 + calibration["bias"]) / calibration["temperature"]))
    negative = 1 / (1 + np.exp(-(-2 + calibration["bias"]) / calibration["temperature"]))
    assert positive > negative

