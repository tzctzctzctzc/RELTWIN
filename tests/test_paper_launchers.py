"""CPU-only checks of paper entry points; the trainer is replaced by an argv spy."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from check_paper_release import ROOT


GIT_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = str(GIT_BASH) if os.name == 'nt' and GIT_BASH.is_file() else shutil.which('bash')


def launch(tmp_path, backbone, existing_output=False):
    if BASH is None:
        pytest.skip('Bash is required to exercise the training launchers')
    spy = tmp_path / 'argv-spy.sh'
    spy.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@"\n', encoding='utf-8')
    spy.chmod(0o755)
    output = tmp_path / 'new training output'
    if existing_output:
        output.mkdir()
    env = dict(os.environ, PYTHON_BIN=spy.as_posix())
    result = subprocess.run(
        [BASH, 'reproduction/train_paper.sh', backbone, '/example/assets with spaces',
         output.as_posix(), '--audit-only'], cwd=ROOT, env=env, capture_output=True,
    )
    return result, [arg.decode('utf-8') for arg in result.stdout.split(b'\0') if arg]


@pytest.mark.parametrize('backbone', ['spotsound', 'timeaudio'])
def test_paper_launches_full_configuration(tmp_path, backbone):
    result, argv = launch(tmp_path, backbone)
    assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
    assert argv[0].endswith(f'/scripts/train_{backbone}_reltwin_full.py')
    spec = json.loads((ROOT / 'experiments/paper/release.json').read_text(encoding='utf-8'))
    full = spec['full_reltwin']
    for flag, key in [('--rbee-steps', 'stage1_updates'), ('--setpo-steps', 'stage2_updates'),
                      ('--lora-learning-rate', 'lora_learning_rate'), ('--method-weight', 'lambda'),
                      ('--exchange-weight', 'gamma'), ('--rbee-replay-weight', 'beta'),
                      ('--setpo-replay-weight', 'beta'), ('--prediction-temperature', 'tau'),
                      ('--target-temperature', 'tau_q'), ('--seed', 'seed'),
                      ('--replay-reference-kl-weight', 'kappa_replay_reference')]:
        assert float(argv[argv.index(flag) + 1]) == full[key]
    bridge_flag = '--bridge-learning-rate' if backbone == 'spotsound' else '--temporal-learning-rate'
    assert float(argv[argv.index(bridge_flag) + 1]) == full['bridge_learning_rate']
    assert '--audit-only' == argv[-1]
    assert '256' not in argv
    if backbone == 'spotsound':
        assert float(argv[argv.index('--relation-reference-kl-weight') + 1]) == full['kappa_relation_reference'] == 0
        assert argv[argv.index('--output-adapter') + 1].endswith('/final_adapter')
        assert argv[argv.index('--output-delta') + 1].endswith('/final_bridge.pt')
    else:
        assert argv[argv.index('--config') + 1].endswith('/configs/timeaudio_paper.yaml')
        assert argv[argv.index('--output-delta') + 1].endswith('/final_delta.pt')
    # Paths with spaces must remain single arguments.
    assert '/example/assets with spaces' in argv[argv.index('--twin-source') + 2]


@pytest.mark.parametrize('backbone', ['spotsound', 'timeaudio'])
def test_existing_output_is_preserved(tmp_path, backbone):
    result, argv = launch(tmp_path, backbone, existing_output=True)
    assert result.returncode == 2
    assert not argv
    assert b'new output directory' in result.stderr


def test_unknown_backbone_is_rejected(tmp_path):
    result, argv = launch(tmp_path, 'unknown')
    assert result.returncode == 2
    assert not argv


def test_paper_yaml_preserves_model_configuration():
    archived = (ROOT / 'reproduction/archive/timeaudio_infer_config.yaml').read_text(encoding='utf-8')
    paper = (ROOT / 'configs/timeaudio_paper.yaml').read_text(encoding='utf-8')
    expected = (archived.replace('max_new_tokens: 200', 'max_new_tokens: 128')
                .replace('num_beams: 4', 'num_beams: 1').replace('top_p: 0.9', 'top_p: 1.0')
                .replace('seed: 42', 'seed: 0'))
    assert paper.split('\n', 1)[1] == expected


def test_paper_confirmation_and_artifact_profiles_are_separate():
    spec = json.loads((ROOT / 'experiments/paper/release.json').read_text(encoding='utf-8'))
    profiles = json.loads((ROOT / 'configs/release_profiles.json').read_text(encoding='utf-8'))
    confirmation = next(c for c in spec['run_confirmations'] if c['method'] == 'SpotSound-A + RelTwin')
    assert confirmation['source'] == 'author_confirmation'
    assert confirmation['training_configuration'] == 'full_reltwin'
    assert not confirmation['original_run_archive_available']
    assert profiles['spotsound']['training']['steps'] == spec['table2_spotsound']['updates'] == 256
    assert profiles['spotsound']['checkpoint_sha256'] == (
        '2b1ae8c344175afcf5116f603747c446ce6f8e46f5fa991212bdc909ccf248b7')


def test_spotsound_trainer_defaults_to_replay_only_kl(monkeypatch):
    import argparse
    import ast
    import sys

    tree = ast.parse((ROOT / 'scripts/train_spotsound_reltwin_full.py').read_text(encoding='utf-8'))
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'arguments')
    namespace = {'argparse': argparse, 'Path': Path}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<trainer-arguments>', 'exec'), namespace)
    argv = ['trainer', '--base', 'base', '--adapter', 'adapter', '--twin-source', 't', 't.json', 'audio', '1',
            '--replay-source', 'r', 'r.json', 'audio', '1']
    for flag in ['--stage1-adapter', '--stage1-delta', '--output-adapter', '--output-delta', '--summary']:
        argv.extend([flag, 'output'])
    monkeypatch.setattr(sys, 'argv', argv)
    args = namespace['arguments']()
    assert args.relation_reference_kl_weight == 0
    assert args.replay_reference_kl_weight == 0.2
