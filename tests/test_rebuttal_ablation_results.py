"""Regression checks for the CPU-only supplementary-results verifier."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'rebuttal_results', ROOT / 'results/rebuttal_ablation_20261027/verify_results.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bootstrap_constant_gain():
    groups = {'audio_a': {'mIoU': [2., 2.]}, 'audio_b': {'mIoU': [2., 2.]}}
    assert MODULE.bootstrap(groups)['mIoU'] == dict(estimate=2., ci95=[2., 2.])


def test_unpaired_groups_rejected():
    with pytest.raises(ValueError, match='Unpaired audio groups'):
        MODULE.contrast({'a': {'x': {'mIoU': [1.]}}, 'b': {'y': {'mIoU': [1.]}}},
                        {'a': 1, 'b': -1})


def test_metric_mismatch_rejected():
    with pytest.raises(ValueError, match='Report value differs'):
        MODULE.same({'mIoU': 60.}, {'mIoU': 59.})


def test_wrong_sample_count_rejected():
    with pytest.raises(ValueError, match='Wrong sample count'):
        MODULE.groups_for([], [], relation=True)


def test_exact_half_iou_preserves_evaluator_threshold_arithmetic():
    import json
    folder = ROOT / 'results/rebuttal_ablation_20261027'
    manifest = MODULE.read(folder / 'manifests/relation600.json')
    rows = [json.loads(line) for line in
            (folder / 's0_continue128/relation600/predictions.jsonl').read_text().splitlines()]
    groups = MODULE.groups_for(manifest, rows, relation=True)
    result = MODULE.bootstrap(groups)
    assert result['R1@0.5']['estimate'] == pytest.approx(94.66666666666667)
    assert result['JointPairAcc@0.5']['estimate'] == pytest.approx(91.33333333333333)
