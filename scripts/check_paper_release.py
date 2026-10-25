#!/usr/bin/env python3
"""Validate the paper-facing catalog, without training or loading model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(spec, readme):
    require(spec['schema_version'] == 1, 'Unknown specification version')
    require(spec['configuration_kind'] == 'submitted_paper_specification', 'Paper specification kind')
    require(spec['training_launcher'] == 'reproduction/train_paper.sh', 'Paper training entry mismatch')
    require(spec['timeaudio_paper_config'] == 'configs/timeaudio_paper.yaml', 'Paper inference config mismatch')
    for confirmation in spec['run_confirmations']:
        require(confirmation['source'] == 'author_confirmation', 'Run confirmation source mismatch')
        require(confirmation['table'] == 'main_table'
                and confirmation['method'] in {r['method'] for r in spec['main_table']},
                'Run confirmation table mismatch')
        require(confirmation['training_configuration'] == 'full_reltwin', 'Confirmed configuration mismatch')
    require([b['queries'] for b in spec['benchmarks']] == [400, 6649, 100],
            'Public benchmark protocol/count mismatch')
    require(spec['metric_order'] == ['R1@0.3', 'R1@0.5', 'mIoU'], 'Metric order mismatch')
    rows = spec['main_table']
    require(len(rows) == 4 and len({r['method'] for r in rows}) == 4, 'Main table row identities')
    for row in rows:
        require(len(row['values']) == 3, 'Missing benchmark')
        for values in row['values']:
            require(len(values) == 3 and all(0 <= v <= 100 for v in values), 'Invalid public metrics')
            require(values[0] >= values[1], 'Recall thresholds reversed')
        expected = ' | '.join(' / '.join(f'{v:.2f}' for v in vs) for vs in row['values'])
        main_table_text = readme.split('## Relation diagnostics', 1)[0]
        matching = [line.replace('**', '') for line in main_table_text.splitlines()
                    if line.startswith('| ' + row['method'] + ' |')]
        require(len(matching) == 1 and expected in matching[0], 'README mismatch: ' + row['method'])
        require(row['provenance'] == ('paper_reported' if '+ RelTwin' in row['method'] else 'published_baseline'),
                'Baseline/adaptation provenance mismatch')
    relation = spec['relation_table']
    require((relation['pairs'], relation['queries']) == (300, 600), 'Relation protocol mismatch')
    require(len(relation['rows']) == 6, 'Missing relation table rows')
    for row in relation['rows']:
        values = row['values']
        require(len(values) == 4 and all(0 <= x <= 100 for x in values[:3]), 'Invalid relation metrics')
        require(type(values[3]) is int and 0 <= values[3] <= 300, 'Invalid same-answer count')
        expected = f"| {row['backbone']} | {row['training']} | " + ' | '.join(
            [f'{v:.2f}' for v in values[:3]] + [f'{values[3]}/300']) + ' |'
        require(expected in readme, 'README relation row mismatch: ' + row['training'])
    full = spec['full_reltwin']
    expected_full = dict(stage1_updates=384, stage2_updates=128, lora_learning_rate=2e-6,
                         bridge_learning_rate=5e-6, seed=0, gamma=1, beta=0.75,
                         kappa_replay_reference=0.2, kappa_relation_reference=0, tau=1, tau_q=0.15,
                         inference='direct_generation', num_beams=1, max_new_tokens=128)
    expected_full['lambda'] = 1
    for key, value in expected_full.items():
        require(full[key] == value, 'Full configuration mismatch: ' + key)
    require(spec['evaluation'] == dict(profile='paper', prediction_clipping='audio_duration',
                                      target_clipping='none', empty_union_iou=0,
                                      recall_thresholds=[0.3, 0.5], percentage_scale=100),
            'Paper scoring configuration mismatch')
    ablation = spec['table2_spotsound']
    require((ablation['updates'], ablation['gamma'], ablation['beta']) == (256, 0, 0.5),
            'SpotSound ablation relabeled as full RelTwin')
    require((ablation['paired_sft_lambda'], ablation['candidate_lambda']) == (0, 1), 'Ablation switches')
    ta = spec['table2_timeaudio']
    require((ta['stage1_updates'], ta['stage2_updates'], ta['paired_sft_lambda'], ta['reltwin_lambda'])
            == (384, 128, 0, 1), 'TimeAudio control mismatch')


def check_paper(spec, paper_dir):
    for relative, expected in spec['paper']['text_sha256_lf'].items():
        data = (paper_dir / relative).read_bytes().replace(b'\r\n', b'\n')
        require(hashlib.sha256(data).hexdigest() == expected, 'Paper source changed: ' + relative)
    pdf = (paper_dir / 'paper.pdf').read_bytes()
    require(hashlib.sha256(pdf).hexdigest() == spec['paper']['pdf_sha256'], 'Paper PDF changed')
    main = (paper_dir / 'tables/main_results.tex').read_text(encoding='utf-8')
    for row in spec['main_table']:
        lines = [x for x in main.splitlines() if x.startswith(row['method'] + ' &')]
        require(len(lines) == 1, 'Missing paper row: ' + row['method'])
        values = [float(x) for x in re.findall(r'\d+\.\d+|\b\d+\b', lines[0].split('&', 1)[1])]
        require(values == [v for triple in row['values'] for v in triple], 'Paper main metrics mismatch')
    training = (paper_dir / 'tables/training_results.tex').read_text(encoding='utf-8')
    for row in spec['relation_table']['rows']:
        prefix = f"{row['backbone']} & {row['training']} &"
        lines = [x for x in training.splitlines() if x.startswith(prefix)]
        require(len(lines) == 1, 'Missing paper relation row')
        values = [float(x) for x in re.findall(r'\d+\.\d+|\b\d+\b', lines[0][len(prefix):])]
        require(values == row['values'] + [300], 'Paper relation metrics mismatch')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path)
    parser.add_argument('--submitted-pdf', type=Path, help='Check the author-confirmed submission PDF')
    parser.add_argument('--require-reproduction-ready', action='store_true')
    args = parser.parse_args(argv)
    spec = json.loads((ROOT / 'experiments/paper/release.json').read_text(encoding='utf-8'))
    validate(spec, (ROOT / 'README.md').read_text(encoding='utf-8'))
    if args.paper_dir:
        check_paper(spec, args.paper_dir)
    if args.submitted_pdf:
        require(hashlib.sha256(args.submitted_pdf.read_bytes()).hexdigest() == spec['paper']['pdf_sha256'],
                'Submitted PDF hash mismatch')
    print(json.dumps({'catalog_consistent': True, 'paper_files_checked': bool(args.paper_dir),
                      'training_launcher': spec['training_launcher'],
                      'reproduction_status': spec['reproduction_status'],
                      'training_or_benchmark_run': False}, indent=2))
    if args.require_reproduction_ready and spec['reproduction_status'] != 'verified':
        print('Available verification covers archived prediction tables; no fresh training run is certified.')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
