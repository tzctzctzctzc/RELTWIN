import itertools

import numpy as np
import pytest

from interval_metrics import temporal_set_iou
from metric_iou_decoder import (
    calibrate_occupancy,
    dinkelbach_trust_region_decode,
    posterior_set_iou,
    resample_probabilities,
)


def test_binary_posterior_equals_temporal_set_iou():
    probabilities = np.asarray([0, 1, 1, 0, 1, 0], dtype=float)
    truth = [(1.0, 3.0), (4.0, 5.0)]
    prediction = [(0.0, 2.0), (4.0, 6.0)]
    assert posterior_set_iou(probabilities, prediction, 6.0) == pytest.approx(
        temporal_set_iou(truth, prediction)
    )


def test_calibration_and_resampling_are_stable():
    calibrated = calibrate_occupancy([-1000, 0, 1000], 2.0, 0.0)
    assert calibrated.tolist() == pytest.approx([0.0, 0.5, 1.0], abs=1e-12)
    assert resample_probabilities([0.0, 1.0], 4).tolist() == pytest.approx(
        [0.0, 0.25, 0.75, 1.0]
    )


def test_dinkelbach_matches_bruteforce_on_small_grid():
    probabilities = np.asarray([0.05, 0.9, 0.9, 0.05])
    incumbent = [(0.0, 1.0)]
    result = dinkelbach_trust_region_decode(
        probabilities, incumbent, 4.0, 3.0, tolerance=1e-10
    )
    choices = []
    for start, end in itertools.combinations(range(5), 2):
        choices.append((posterior_set_iou(probabilities, [(start, end)], 4.0), [(start, end)]))
    optimum = max(score for score, _ in choices)
    assert result.posterior_iou_candidate == pytest.approx(optimum)
    assert result.selected == [(1.0, 3.0)]


def test_identity_and_cardinality_are_preserved():
    result = dinkelbach_trust_region_decode(
        [0.9, 0.9, 0.0, 0.0, 0.9, 0.9],
        [(0.0, 2.0), (4.0, 6.0)],
        6.0,
        0.0,
    )
    assert result.selected == [(0.0, 2.0), (4.0, 6.0)]
    assert len(result.selected) == 2
    assert not result.switch


def test_empty_and_invalid_incumbents_abstain():
    empty = dinkelbach_trust_region_decode([0.5, 0.5], [], 2.0, 1.0)
    assert empty.abstain_reason == "empty_incumbent"
    invalid = dinkelbach_trust_region_decode(
        [0.5, 0.5], [(0.0, 1.5), (1.0, 2.0)], 2.0, 1.0
    )
    assert invalid.abstain_reason.startswith("invalid_incumbent:")
    assert not invalid.switch


def test_negative_bootstrap_lower_bound_abstains():
    result = dinkelbach_trust_region_decode(
        [0.05, 0.9, 0.9, 0.05],
        [(0.0, 1.0)],
        4.0,
        3.0,
        bootstrap_probabilities=[[0.9, 0.05, 0.05, 0.05]],
    )
    assert result.candidate != result.incumbent
    assert result.selected == result.incumbent
    assert result.abstain_reason == "gain_lower_bound_not_positive"


def test_strict_non_overlap_and_clipping():
    result = dinkelbach_trust_region_decode(
        [0.9, 0.8, 0.7, 0.7, 0.8, 0.9],
        [(0.1, 2.0), (4.0, 5.9)],
        6.0,
        3.0,
    )
    assert all(0 <= start < end <= 6 for start, end in result.candidate)
    assert result.candidate[0][1] < result.candidate[1][0]

