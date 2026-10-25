import copy

import pytest

from paper_metrics import ROOT, main, score_public


def example():
    return [dict(index=0, status='ok', duration_seconds=2.,
                 ground_truth=[[0., 3.]], prediction=[[0., 2.]])]


def test_duration_profiles():
    rows = example()
    original = copy.deepcopy(rows)
    assert score_public(rows, 'spotsound', profile='archive')['mIoU'] == 100
    assert score_public(rows, 'timeaudio', profile='archive')['mIoU'] == pytest.approx(200 / 3)
    assert rows == original


def test_paper_profile_is_shared_and_preserves_targets():
    rows = example()
    original = copy.deepcopy(rows)
    for backbone in ['spotsound', 'timeaudio']:
        assert score_public(rows, backbone)['mIoU'] == pytest.approx(200 / 3)
    assert rows == original


def test_unknown_scoring_profile_fails():
    with pytest.raises(ValueError, match='profile'):
        score_public(example(), 'spotsound', profile='unknown')


def test_prediction_clipping_is_shared():
    rows = example()
    rows[0]['ground_truth'] = [[0., 2.]]
    rows[0]['prediction'] = [[0., 4.]]
    for name in ['spotsound', 'timeaudio']:
        assert score_public(rows, name)['mIoU'] == 100


def test_empty_prediction_remains_in_denominator():
    rows = example()
    rows[0]['prediction'] = []
    assert score_public(rows, 'spotsound')['mIoU'] == 0


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'index', 'status', 'nan', 'interval'])
def test_reject_corruption(fault):
    rows = example()
    if fault == 'missing':
        rows.clear()
    elif fault == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif fault == 'index':
        rows[0]['index'] = 3
    elif fault == 'status':
        rows[0]['status'] = 'error'
    elif fault == 'nan':
        rows[0]['duration_seconds'] = float('nan')
    else:
        rows[0]['prediction'] = [[2., 1.]]
    with pytest.raises(ValueError):
        score_public(rows, 'spotsound', expected_count=1)


def test_unav_cli(capsys):
    assert main(['--backbone', 'spotsound', '--benchmark', 'unav100', '--predictions',
                 str(ROOT / 'results/paper_spotsound/unav100/predictions.jsonl')]) == 0
    output = capsys.readouterr().out
    assert '73.957494' in output
    assert 'paper_prediction_clipping' in output


def test_archive_unav_cli(capsys):
    assert main(['--backbone', 'spotsound', '--benchmark', 'unav100', '--profile', 'archive',
                 '--predictions', str(ROOT / 'results/paper_spotsound/unav100/predictions.jsonl')]) == 0
    assert '74.034272' in capsys.readouterr().out


def test_wrong_benchmark_count_fails():
    with pytest.raises(ValueError, match='indices'):
        main(['--backbone', 'spotsound', '--benchmark', 'clotho', '--predictions',
              str(ROOT / 'results/paper_spotsound/unav100/predictions.jsonl')])
