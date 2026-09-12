import numpy as np
import pytest

from audit_reltwin_review import cluster_bootstrap, pair_structure, score_predictions, source_components


def pair(audio="x.wav", pair_id=0, sources=None):
    shared = {"audio_path": audio, "pair_id": pair_id, "template": "followed_by", "variant": 0,
              "source_files": sources or ["1-10-A-0.wav", "2-11-A-1.wav"]}
    return [{**shared, "relation": "AB", "annotations": [[0., 1.]]},
            {**shared, "relation": "BA", "annotations": [[2., 3.]]}]


def test_joint_accuracy_rejects_union_counterexample():
    values = score_predictions(pair(), [[[0., 1.], [2., 3.]], [[0., 1.], [2., 3.]]])
    assert values["PairAcc@0.5"].tolist() == [1.]
    assert values["SwapError"].tolist() == [1.]
    assert values["JointPairAcc@0.5"].tolist() == [0.]


def test_joint_accuracy_accepts_relation_specific_windows():
    values = score_predictions(pair(), [[[0., 1.]], [[2., 3.]]])
    assert values["JointPairAcc@0.5"].tolist() == [1.]


def test_duplicate_relations_fail():
    rows = pair()
    with pytest.raises(ValueError):
        pair_structure(rows + [rows[0]])


def test_source_groups_connect_different_takes_and_transitive_reuse():
    rows = pair("a.wav", 1) + pair("b.wav", 2, ["3-11-B-1.wav", "1-12-A-2.wav"])
    rows += pair("c.wav", 3, ["4-12-C-2.wav", "1-13-A-3.wav"])
    _, sizes = source_components(rows)
    assert sizes == [3]


def test_bootstrap_constant_effect_and_determinism():
    values = np.full((3, 8), .1)
    a = cluster_bootstrap(values, ["a"]*4+["b"]*4, 100)
    b = cluster_bootstrap(values, ["a"]*4+["b"]*4, 100)
    assert a == b
    assert np.allclose(a["ci95_points"], [10., 10.])


def test_shared_seed_cluster_draws_preserve_cancellation():
    values = np.asarray([[.2, .2, -.1, -.1], [-.2, -.2, .1, .1]])
    result = cluster_bootstrap(values, ["a", "a", "b", "b"], 100)
    assert result["ci95_points"] == [0., 0.]


def test_one_source_component_does_not_invent_a_ci():
    assert cluster_bootstrap([[.1, .2]], ["a", "a"], 10)["ci95_points"] is None
