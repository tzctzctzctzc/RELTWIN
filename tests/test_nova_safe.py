import json

import numpy as np
import pytest

from extract_nova_safe_features import validate_alignment
from nova_safe import (
    SCHEMA_VERSION,
    choose_candidate,
    hard_guard,
    index_rows,
    pair_feature_map,
    prediction_error,
)
from prepare_nova_safe_pilot import choose_groups, stratum


def candidate(prediction, iou=0.5, **features):
    return {"prediction": prediction, "iou": iou, "features": features}


def row(incumbent_prediction=None, challenger_prediction=None):
    incumbent_prediction = incumbent_prediction or [[10.0, 20.0]]
    challenger_prediction = challenger_prediction or [[30.0, 40.0]]
    return {
        "feature_schema_version": SCHEMA_VERSION,
        "benchmark": "test",
        "source": "source-a",
        "source_index": 1,
        "audio": "a.wav",
        "audio_group": "a.wav",
        "query": "a target sound",
        "duration_seconds": 60.0,
        "ground_truth": [[10.0, 20.0]],
        "incumbent": "official",
        "candidates": {
            "official": candidate(incumbent_prediction, 1.0, component_mean=1.0, component_min=1.0, complement_no=1.0),
            "setpo": candidate(challenger_prediction, 0.0, component_mean=2.0, component_min=2.0, complement_no=2.0),
        },
    }


def test_prediction_validation_rejects_invalid_intervals():
    assert prediction_error([[0.0, 1.0]], 2.0) is None
    assert prediction_error([[1.0, 0.0]], 2.0) == "non_positive_interval"
    assert prediction_error([[-0.1, 1.0]], 2.0) == "out_of_bounds"
    assert prediction_error([[0.0, 3.0]], 2.0) == "out_of_bounds"


def test_hard_guards_cover_empty_cardinality_and_equivalence():
    incumbent = candidate([[1.0, 2.0]])
    assert hard_guard(incumbent, candidate([]), 10.0) == "candidate_empty"
    assert hard_guard(incumbent, candidate([[1, 2], [3, 4], [5, 6]]), 10.0) == "cardinality_jump"
    assert hard_guard(incumbent, candidate([[1.0, 2.0]]), 10.0) == "equivalent_prediction"


def test_pair_features_include_geometry_and_verifier_deltas():
    features = pair_feature_map(row(), "setpo")
    assert features["agreement"] == 0.0
    assert features["component_mean_delta"] == 1.0
    assert features["component_min_delta"] == 1.0
    assert features["complement_no_delta"] == 1.0
    assert features["query_words"] == 3.0


def test_router_keeps_incumbent_without_positive_lower_bound():
    example = row()
    model = {
        "feature_names": ["agreement"],
        "mean": [0.0],
        "scale": [1.0],
        "weights": [[-0.1, 0.0], [-0.2, 0.0]],
        "threshold": 0.0,
    }
    decision = choose_candidate(example, model)
    assert decision["selected"] == "official"
    assert decision["abstain_reason"] == "no_positive_lower_bound"


def test_router_switches_only_when_lower_bound_clears_threshold():
    example = row()
    model = {
        "feature_names": ["absolute_coverage_delta"],
        "mean": [0.0],
        "scale": [1.0],
        "weights": [[0.2, 0.0], [0.3, 0.0]],
        "threshold": 0.1,
    }
    decision = choose_candidate(example, model)
    assert decision["selected"] == "setpo"
    assert decision["decision_margin"] > 0.0


def test_router_abstains_before_scoring_equivalent_prediction():
    example = row(challenger_prediction=[[10.0, 20.0]])
    model = {
        "feature_names": ["agreement"],
        "mean": [0.0],
        "scale": [1.0],
        "weights": [[1.0, 0.0]],
        "threshold": 0.0,
    }
    decision = choose_candidate(example, model)
    assert decision["selected"] == "official"
    assert decision["guarded"]["setpo"] == "equivalent_prediction"


def test_strict_alignment_rejects_missing_candidate_rows():
    first = index_rows([{"index": 0}, {"index": 1}], "bench")
    second = index_rows([{"index": 0}], "bench")
    manifest = {("bench", 0): {}, ("bench", 1): {}}
    with pytest.raises(ValueError, match="Candidate key mismatch"):
        validate_alignment({"official": first, "setpo": second}, manifest)


def test_duplicate_source_index_is_rejected():
    with pytest.raises(ValueError, match="Duplicate prediction key"):
        index_rows([{"source_index": 2}, {"source_index": 2}], "bench")


def test_grouped_pilot_never_splits_an_audio_group():
    groups = {
        "a.wav": [{"audio_group": "a.wav", "stratum": "x"}] * 2,
        "b.wav": [{"audio_group": "b.wav", "stratum": "x"}],
        "c.wav": [{"audio_group": "c.wav", "stratum": "y"}],
    }
    selected = choose_groups(groups, 3, 20260902)
    counts = {name: sum(row["audio_group"] == name for row in selected) for name in groups}
    assert len(selected) == 3
    assert counts["a.wav"] in (0, 2)


def test_pilot_stratum_ignores_explicit_null_duration_and_uses_manifest_value():
    manifest = {
        "duration": 60.0,
        "annotations": [[10.0, 20.0]],
        "caption": "target sound",
    }
    prediction = {"duration_seconds": None, "prediction": [[10.0, 20.0]]}
    assert stratum(manifest, prediction).startswith("single|")
