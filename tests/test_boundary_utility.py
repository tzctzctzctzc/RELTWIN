import numpy as np

import boundary_utility

from boundary_utility import (
    bootstrap_utility_models,
    boundary_edit_features,
    decode_boundary_utility,
    evidence_arrays,
    fit_ridge_utility,
    interval_radii,
    interval_options,
    make_training_examples,
)


def _record(index=0):
    fine_steps = 100
    times = (np.arange(fine_steps) + 0.5) * 4 / fine_steps
    occupancy = np.where((times >= 1.0) & (times <= 2.0), 5.0, -5.0)
    fine_onset = -5 * np.ones(fine_steps)
    fine_offset = -5 * np.ones(fine_steps)
    fine_onset[np.argmin(abs(times - 1.0))] = 5
    fine_offset[np.argmin(abs(times - 2.0))] = 5
    return {
        "source": "synthetic",
        "source_index": index,
        "audio_group": f"audio_{index}.wav",
        "duration": 4.0,
        "fine_steps": fine_steps,
        "occupancy_logits": occupancy.reshape(20, 5).mean(axis=1).tolist(),
        "onset_logits": fine_onset.reshape(20, 5).max(axis=1).tolist(),
        "offset_logits": fine_offset.reshape(20, 5).max(axis=1).tolist(),
        "fine_onset_logits": fine_onset.tolist(),
        "fine_offset_logits": fine_offset.tolist(),
        "incumbent_prediction": [[0.8, 1.8]],
        "annotations": [[1.0, 2.0]],
    }


def test_identity_boundary_edit_has_exact_zero_features():
    record = _record()
    features = boundary_edit_features(record, 0, 0.8, 1.8)
    assert np.array_equal(features, np.zeros_like(features))


def test_identity_survives_sub_picosecond_gap_between_intervals():
    record = _record()
    record["incumbent_prediction"] = [
        [0.8, 1.0000000000004],
        [1.00000000000049, 1.8],
    ]
    for interval_index in range(2):
        options = interval_options(record, interval_index, 0.25)
        assert any(option.identity for option in options)


def test_adaptive_radius_is_relative_and_capped():
    intervals = [(0.0, 0.2), (1.0, 3.0)]
    assert interval_radii(intervals, 0.25, 0.25) == [0.05, 0.25]
    assert interval_radii(intervals, 0.25) == [0.25, 0.25]


def test_adaptive_decode_never_moves_outside_event_specific_radius():
    records = [_record(index) for index in range(4)]
    examples = make_training_examples(records, 0.25)
    model = fit_ridge_utility(examples, 1e-4)
    record = records[0]
    ratio = 0.1
    result = decode_boundary_utility(
        record, model, [model], 0.25, -1e-9, radius_ratio=ratio
    )
    allowed = ratio * (1.8 - 0.8)
    for old, new in zip(result.incumbent, result.candidate):
        assert abs(old[0] - new[0]) <= allowed + 1e-12
        assert abs(old[1] - new[1]) <= allowed + 1e-12


def test_invalid_adaptive_radius_rejected():
    with np.testing.assert_raises(ValueError):
        interval_radii([(0.0, 1.0)], 0.25, -0.1)


def test_invalid_trust_region_abstains_instead_of_aborting(monkeypatch):
    record = _record()
    features = boundary_edit_features(record, 0, 0.8, 1.8)
    model = boundary_utility.RidgeUtilityModel(
        alpha=1.0,
        scale=np.ones_like(features).tolist(),
        weights=np.zeros_like(features).tolist(),
    )

    def fail_options(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(boundary_utility, "interval_options", fail_options)
    result = decode_boundary_utility(record, model, [model], 0.25, 0.02)
    assert not result.switch
    assert result.selected == result.incumbent
    assert result.abstain_reason == "invalid_trust_region:synthetic failure"


def test_decode_honors_forced_memory_abstention_without_evidence():
    record = {
        "duration": 404.0,
        "incumbent_prediction": [[2.0, 402.0]],
        "force_abstain_reason": "boundary_crop_exceeds_memory_budget",
    }
    model = boundary_utility.RidgeUtilityModel(1.0, [1.0], [0.0])
    result = decode_boundary_utility(record, model, [], 0.25, 0.02)
    assert result.selected == [(2.0, 402.0)]
    assert not result.switch
    assert result.abstain_reason == "boundary_crop_exceeds_memory_budget"


def test_evidence_arrays_are_aligned_and_bounded():
    arrays = evidence_arrays(_record())
    assert len(arrays) == 5
    assert all(array.shape == (100,) for array in arrays)
    assert all(((array >= 0) & (array <= 1)).all() for array in arrays)


def test_group_bootstrap_models_are_deterministic():
    records = [_record(index) for index in range(3)]
    examples = make_training_examples(records, 0.25)
    left = bootstrap_utility_models(examples, 1e-3, 3, 7)
    right = bootstrap_utility_models(examples, 1e-3, 3, 7)
    assert [model.weights for model in left] == [model.weights for model in right]


def test_learned_utility_keeps_cardinality_and_improves_training_pattern():
    records = [_record(index) for index in range(4)]
    examples = make_training_examples(records, 0.25)
    model = fit_ridge_utility(examples, 1e-4)
    result = decode_boundary_utility(records[0], model, [model], 0.25, -1e-9)
    assert len(result.selected) == 1
    assert result.switch
    assert result.selected != result.incumbent
