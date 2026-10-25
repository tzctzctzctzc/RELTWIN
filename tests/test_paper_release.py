import json

import pytest

from check_paper_release import ROOT, check_paper, main, validate


def inputs():
    return (json.loads((ROOT / 'experiments/paper/release.json').read_text(encoding='utf-8')),
            (ROOT / 'README.md').read_text(encoding='utf-8'))


def test_current_catalog():
    validate(*inputs())


@pytest.mark.parametrize('mutation', ['clotho_pilot', 'old_diagnostic', 'nova', 'ablation', 'metric',
                                    'baseline', 'relation_kl', 'scoring'])
def test_detect_identity_drift(mutation):
    spec, readme = inputs()
    if mutation == 'clotho_pilot':
        spec['benchmarks'][1]['queries'] = 1000
    elif mutation == 'old_diagnostic':
        spec['relation_table']['pairs'] = 160
    elif mutation == 'nova':
        spec['full_reltwin']['inference'] = 'nova_router'
    elif mutation == 'ablation':
        spec['table2_spotsound']['gamma'] = 1
    elif mutation == 'metric':
        spec['main_table'][3]['values'][0][2] = 59.386
    elif mutation == 'baseline':
        spec['main_table'][0]['provenance'] = 'same_harness_reproduction'
    elif mutation == 'relation_kl':
        spec['full_reltwin']['kappa_relation_reference'] = 0.2
    else:
        spec['evaluation']['target_clipping'] = 'audio_duration'
    with pytest.raises(ValueError):
        validate(spec, readme)


def test_readiness_is_not_implied_by_catalog_validation():
    assert main([]) == 0
    assert main(['--require-reproduction-ready']) == 2


def test_frozen_sources_are_checked(tmp_path):
    spec, _ = inputs()
    with pytest.raises(FileNotFoundError):
        check_paper(spec, tmp_path)


def test_wrong_submission_pdf_rejected(tmp_path):
    submitted = tmp_path / 'finalpaper.pdf'
    submitted.write_bytes(b'not the submitted PDF')
    with pytest.raises(ValueError, match='Submitted PDF hash mismatch'):
        main(['--submitted-pdf', str(submitted)])
