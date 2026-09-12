from copy import deepcopy
from pathlib import Path

import pytest

from summarize_reltwin_review_controls import aligned_rows, check_durations, endpoint_status, evaluator_iou


def fixture():
    truth = [{"audio_path": "a.wav", "caption": "sound", "annotations": [[0., 1.]]}]
    row = {"index": 0, "status": "ok", "audio": "a.wav", "query": "sound", "ground_truth": [[0., 1.]],
           "duration_seconds": 2., "prediction": [[0., 1.]], "iou": 1.}
    iou = evaluator_iou(Path(__file__).resolve().parents[1] / "scripts/evaluate_checkpoint.py")
    return truth, row, iou


def test_exact_metric_loads_without_gpu_dependencies():
    truth, row, iou = fixture()
    assert aligned_rows([row], truth, iou) == [row]
    assert iou([[0., 1.]], [[0., .5]]) == .5


@pytest.mark.parametrize("field,value", [("query", "other"), ("audio", "b.wav"), ("iou", .1), ("duration_seconds", float("nan")), ("prediction", [[0., 3.]])])
def test_alignment_rejects_bad_rows(field, value):
    truth, row, iou = fixture()
    row[field] = value
    with pytest.raises(ValueError):
        aligned_rows([row], truth, iou)


def test_duplicate_rows_rejected():
    truth, row, iou = fixture()
    with pytest.raises(ValueError):
        aligned_rows([row, deepcopy(row)], truth*2, iou)


def test_duration_disagreement_between_models_rejected():
    _, row, _ = fixture()
    known = {}
    check_durations([row], known)
    row["duration_seconds"] = 2.001
    with pytest.raises(ValueError):
        check_durations([row], known)


def test_incomplete_positive_seeds_never_support_claim():
    result = {"planned_seeds_complete": False, "metrics": {}}
    assert endpoint_status(result) == "PENDING_ALL_PLANNED_SEEDS"


@pytest.mark.parametrize("effect,ci,label", [(0., [-1., 1.], "NO_POSITIVE_MEAN_INCREMENT"),
    (1., [-1., 2.], "POSITIVE_MEAN_WITH_UNCERTAIN_CLUSTER_INTERVAL"),
    (1., [.1, 2.], "POSITIVE_DEVELOPMENT_CONTRAST_WITH_CLUSTER_SUPPORT")])
def test_completed_evidence_label(effect, ci, label):
    result = {"planned_seeds_complete": True, "metrics": {"JointPairAcc@0.5": {
        "audio_cluster": {"mean_delta_points": effect, "ci95_points": ci},
        "source_connected_cluster": {"mean_delta_points": effect, "ci95_points": ci}}}}
    assert endpoint_status(result) == label
