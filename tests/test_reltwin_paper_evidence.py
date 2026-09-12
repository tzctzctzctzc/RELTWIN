from pathlib import Path
import ast

import numpy as np
import pytest

from audit_reltwin_paper_evidence import behavior


def sample():
    common = {"audio_path": "x.wav", "pair_id": 0, "variant": 0, "template": "followed_by"}
    return [{**common, "relation": "AB", "annotations": [[0., 1.]]},
            {**common, "relation": "BA", "annotations": [[2., 3.]]}]


@pytest.mark.parametrize("predictions,expected", [
    ([[[0., 1.]], [[2., 3.]]], (0, 0, 1)),
    ([[[0., 1.]], [[0., 1.]]], (1, 0, 0)),
    ([[], []], (1, 1, 0)),
    ([[[0., 1.], [2., 3.]], [[0., 1.], [2., 3.]]], (1, 0, 0)),
    ([[[2., 3.]], [[0., 1.]]], (0, 0, 0)),
])
def test_response_change_is_not_automatically_correct(predictions, expected):
    result, mask, joint = behavior(sample(), [{"prediction": p} for p in predictions])
    assert (result["same_pairs"], result["both_empty"], result["joint_correct"]) == expected
    assert int(mask.sum()) == result["same_pairs"]
    assert int(joint.sum()) == result["joint_correct"]


def test_equivalence_is_distinct_from_exact_output():
    result, _, _ = behavior(sample(), [{"prediction": [[0., 1.]]}, {"prediction": [[0., .999]]}])
    assert result["same_pairs"] == 0
    assert result["equivalent_pairs"] == 1


def test_queue_has_bounded_timeout_and_kills_own_process_group():
    source = (Path(__file__).resolve().parents[1] / "scripts/run_reltwin_paper_bridge.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert '"budget_seconds": 7200' in source
    assert "process.wait(timeout=remaining)" in source
    assert "start_new_session=True" in source
    assert "os.killpg(process.pid, signal.SIGTERM)" in source
    assert "os.killpg(process.pid, signal.SIGKILL)" in source
    assert 'runtime[key] != old[key]' in source
