import json
import pytest
from reltwin_crossbench_support import load_rows, validate_rows, summarize, paired_summary


def fixture():
    manifest = [dict(audio_path="/data/a.wav", audio_group="a", caption="dog",
                     annotations=[[0, 1]], decoded_duration=2.)]
    rows = [dict(index=0, status="ok", audio="a.wav", query="dog", ground_truth=[[0, 1]],
                 duration_seconds=2., prediction=[[0, 1]], iou=1., inference_seconds=.1)]
    return rows, manifest


def test_identical_complete_and_metrics():
    rows, manifest = fixture()
    assert summarize(rows, manifest)["mIoU"] == 100
    assert paired_summary(rows, rows, manifest)["audio_group_95CI"] == [0., 0.]


@pytest.mark.parametrize("field,value", [("query", "cat"), ("audio", "b.wav"),
                                         ("duration_seconds", 2.1), ("iou", .5),
                                         ("ground_truth", [[0, 2]]), ("prediction", [[0, 3]])])
def test_reject_identity_mismatch(field, value):
    rows, manifest = fixture()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_rows(rows, manifest)


def test_partial_and_duplicates():
    rows, manifest = fixture()
    validate_rows([], manifest)
    with pytest.raises(ValueError):
        validate_rows([], manifest, complete=True)
    with pytest.raises(ValueError):
        validate_rows(rows * 2, manifest)


def test_jsonl_damage_and_duplicate(tmp_path):
    rows, _ = fixture()
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError):
        load_rows(path)
    path.write_text((json.dumps(rows[0]) + "\n") * 2)
    with pytest.raises(ValueError):
        load_rows(path)


def test_endpoint_conventions_explicit():
    rows, manifest = fixture()
    manifest[0]["annotations"] = [[0, 3]]
    rows[0].update(ground_truth=[[0, 3]], prediction=[[0, 2]], iou=2/3)
    result = summarize(rows, manifest)
    assert result["mIoU"] == 100
    assert result["released_endpoint_mIoU"] == pytest.approx(100 * 2/3)
