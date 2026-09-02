import numpy as np

from compare_spantool_predictions import compare, paired_bootstrap_ci


def record(index, prediction):
    return {
        "status": "ok",
        "index": index,
        "ground_truth": [[0.0, 1.0]],
        "prediction": prediction,
    }


def test_compare_reports_paired_gain_and_oracle():
    baseline = {0: record(0, [[0.0, 0.5]]), 1: record(1, [[0.0, 1.0]])}
    candidate = {0: record(0, [[0.0, 1.0]]), 1: record(1, [[0.0, 0.5]])}
    result = compare(baseline, candidate, bootstrap_samples=100, seed=3)
    assert result["delta_mIoU"] == 0.0
    assert result["wins"] == result["losses"] == 1
    assert result["oracle"]["mIoU"] == 100.0


def test_bootstrap_can_be_disabled():
    assert paired_bootstrap_ci(np.asarray([1.0]), 0, 1) is None


def test_compare_aligns_stratified_gate_by_source_index():
    baseline = {3: record(3, [[0.0, 0.5]])}
    candidate_row = record(0, [[0.0, 1.0]])
    candidate_row["source_index"] = 3
    result = compare(baseline, {0: candidate_row}, bootstrap_samples=0, seed=1)
    assert result["delta_mIoU"] == 50.0
