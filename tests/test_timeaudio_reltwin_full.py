import json

import numpy as np
import pytest

from train_timeaudio_reltwin_full import (
    audit_no_evaluation_overlap,
    prepare_source_pools,
    weighted_schedule,
)


def test_weighted_schedule_is_exact_and_deterministic():
    left = weighted_schedule([10, 10], [3, 1], 12, np.random.default_rng(7))
    right = weighted_schedule([10, 10], [3, 1], 12, np.random.default_rng(7))
    assert left == right
    assert sum(source == 0 for source, _ in left) == 9
    assert sum(source == 1 for source, _ in left) == 3


def _sources(tmp_path):
    twin_rows = [
        {"audio_path": "train.wav", "caption": "a", "annotations": [[0, 1]]},
        {"audio_path": "train.wav", "caption": "b", "annotations": [[1, 2]]},
    ]
    replay_rows = [
        {"audio_path": "replay.wav", "caption": "c", "annotations": [[0, 1]]}
    ]
    twin_manifest = tmp_path / "twin.json"
    replay_manifest = tmp_path / "replay.json"
    twin_manifest.write_text(json.dumps(twin_rows), encoding="utf-8")
    replay_manifest.write_text(json.dumps(replay_rows), encoding="utf-8")
    twins = [{"name": "t", "manifest": twin_manifest, "audio_root": tmp_path,
              "weight": 1.0, "rows": twin_rows}]
    replay = [{"name": "r", "manifest": replay_manifest, "audio_root": tmp_path,
               "weight": 1.0, "rows": replay_rows}]
    return prepare_source_pools(twins, replay)


def test_leakage_audit_accepts_disjoint_eval(tmp_path):
    twins, replay = _sources(tmp_path)
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(json.dumps([
        {"audio_path": "test.wav", "caption": "a", "annotations": [[0, 1]]}
    ]), encoding="utf-8")
    report = audit_no_evaluation_overlap(twins, replay, [evaluation])
    assert report["status"] == "pass"


def test_leakage_audit_rejects_shared_audio(tmp_path):
    twins, replay = _sources(tmp_path)
    evaluation = tmp_path / "eval.json"
    evaluation.write_text(json.dumps([
        {"audio_path": "train.wav", "caption": "different", "annotations": [[0, 1]]}
    ]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="overlap"):
        audit_no_evaluation_overlap(twins, replay, [evaluation])
