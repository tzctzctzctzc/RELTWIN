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


def test_timing_intervention_preserves_samples_and_removes_layout_clock():
    source = (Path(__file__).resolve().parents[1] / "scripts/reltwin_timing_stress.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "uniform_timing")
    scope = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "stress", "exec"), scope)
    sr = 16000
    a, b = np.full(5 * sr, .125, np.float32), np.full(5 * sr, -.25, np.float32)
    z = lambda t: np.zeros(round(t * sr), np.float32)
    ab, ba = np.concatenate([a, z(.25), b]), np.concatenate([b, z(.25), a])
    original = np.concatenate([z(1.5), ab, z(3), ba, z(2)])
    row = {"window_ab": [[1.5, 11.75]], "window_ba": [[14.75, 25.]], "layout": "AB_first"}
    result, windows = scope["uniform_timing"](original, sr, row)
    assert np.array_equal(result, original)
    assert windows == {"AB": [[1.5, 11.75]], "BA": [[14.75, 25.]]}
    reverse = np.concatenate([z(2), b, z(.65), a, z(5), a, z(.65), b, z(2)])
    row = {"window_ab": [[17.65, 28.3]], "window_ba": [[2., 12.65]], "layout": "BA_first"}
    result, windows = scope["uniform_timing"](reverse, sr, row)
    assert np.array_equal(result, np.concatenate([z(1.5), ba, z(3), ab, z(2)]))
    assert len(result) == 27 * sr
    assert windows == {"BA": [[1.5, 11.75]], "AB": [[14.75, 25.]]}
