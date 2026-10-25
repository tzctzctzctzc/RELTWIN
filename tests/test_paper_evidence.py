import pytest

from verify_paper_evidence import aligned_rows, public_metrics, relation_metrics, verify


def fixture_pair(predictions):
    targets = [[[0., 1.]], [[2., 3.]]]
    manifest = [dict(pair_id=0, template='followed_by', variant=0, relation=r, audio_path='/audio/x.wav',
                     caption=r, annotations=g) for r, g in zip(['AB', 'BA'], targets)]
    from interval_metrics import temporal_set_iou
    rows = [dict(status='ok', index=i, audio='x.wav', query=m['caption'], duration_seconds=3.,
                 ground_truth=m['annotations'], prediction=p, iou=temporal_set_iou(m['annotations'], p))
            for i, (m, p) in enumerate(zip(manifest, predictions))]
    return rows, manifest


def test_exact_matches():
    rows, manifest = fixture_pair([[[0., 1.]], [[2., 3.]]])
    result = relation_metrics(rows, manifest)
    assert result['JointPairAcc@0.5'] == 100
    assert result['SwapError'] == 0


def test_union_is_coverage_success_but_swap_tie():
    rows, manifest = fixture_pair([[[0., 1.], [2., 3.]]] * 2)
    result = relation_metrics(rows, manifest)
    assert result['PairAcc@0.5'] == 100
    assert result['JointPairAcc@0.5'] == 0
    assert result['SwapError'] == 100
    assert result['same_answer_pairs'] == 1


def test_empty_answers_are_kept():
    rows, manifest = fixture_pair([[], []])
    assert public_metrics(rows)['parse_failure_count'] == 2
    assert relation_metrics(rows, manifest)['mIoU'] == 0


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'query', 'target', 'audio', 'iou', 'duration'])
def test_corruption_fails(fault):
    rows, manifest = fixture_pair([[[0., 1.]], [[2., 3.]]])
    if fault == 'duplicate':
        rows[1]['index'] = 0
    elif fault == 'missing':
        rows.pop()
    elif fault == 'query':
        rows[0]['query'] = 'wrong'
    elif fault == 'target':
        rows[0]['ground_truth'] = [[1., 2.]]
    elif fault == 'audio':
        rows[0]['audio'] = 'wrong.wav'
    elif fault == 'iou':
        rows[0]['iou'] = 0
    else:
        rows[0]['duration_seconds'] = .5
    with pytest.raises(ValueError):
        aligned_rows(rows, 2, manifest)


def test_archived_evidence():
    report = verify()
    assert report['table1_timeaudio']['clotho1000']['queries'] == 6649
    assert report['table2_timeaudio']['reltwin']['pair_count'] == 300
    assert report['table1_spotsound']['clotho6649']['queries'] == 6649
    assert round(report['table1_spotsound']['unav100']['mIoU'], 2) == 74.03
    assert report['table2_spotsound']['candidate256']['same_answer_pairs'] == 2
    assert report['archived_prediction_tables_verified']
    assert not report['full_training_reproduction_certified']
