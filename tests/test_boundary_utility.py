import numpy as np

from boundary_utility import (
    bootstrap_utility_models,
    boundary_edit_features,
    decode_boundary_utility,
    evidence_arrays,
    fit_ridge_utility,
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

