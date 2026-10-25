#!/usr/bin/env python3
"""Verify released prediction files and rescore both paper tables on CPU."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from interval_metrics import temporal_set_iou
from analyze_reltwin_probe import set_iou as diagnostic_set_iou
from paper_metrics import score_public

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results/paper_timeaudio'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def text_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def aligned_rows(rows, count, manifest=None):
    require(len(rows) == count, 'Prediction count mismatch')
    require(all(type(r.get('index')) is int for r in rows), 'Invalid index type')
    require({r['index'] for r in rows} == set(range(count)), 'Duplicate, missing, or displaced indices')
    rows = sorted(rows, key=lambda r: r['index'])
    for i, row in enumerate(rows):
        require(row.get('status') == 'ok', f'Non-ok row: {i}')
        duration = row['duration_seconds']
        require(math.isfinite(duration) and duration > 0, f'Invalid duration: {i}')
        for span in row['prediction']:
            require(len(span) == 2 and all(math.isfinite(x) for x in span)
                    and 0 <= span[0] < span[1] <= duration, f'Invalid prediction: {i}')
        iou = temporal_set_iou(row['ground_truth'], row['prediction'])
        require(math.isclose(iou, row['iou'], rel_tol=0, abs_tol=1e-10), f'Per-row IoU mismatch: {i}')
        if manifest is not None:
            item = manifest[i]
            require(row['query'] == item['caption'] and row['audio'] == Path(item['audio_path']).name
                    and row['ground_truth'] == item['annotations'], f'Manifest alignment mismatch: {i}')
    return rows


def public_metrics(rows):
    ious = [temporal_set_iou(r['ground_truth'], r['prediction']) for r in rows]
    require(bool(ious), 'Empty evaluation')
    n = len(ious)
    return {'mIoU': 100 * math.fsum(ious) / n,
            'R1@0.3': 100 * sum(i >= .3 for i in ious) / n,
            'R1@0.5': 100 * sum(i >= .5 for i in ious) / n,
            'parse_failure_count': sum(not r['prediction'] for r in rows)}


def relation_metrics(rows, manifest):
    rows = aligned_rows(rows, len(manifest), manifest)
    pairs = defaultdict(dict)
    for i, item in enumerate(manifest):
        key = (item['pair_id'], item['template'], item['variant'])
        relation = item['relation']
        require(relation not in pairs[key], f'Duplicate relation: {key}')
        pairs[key][relation] = i
    require(bool(pairs), 'No relation pairs')
    successes, swaps, joints, same = [], [], [], []
    for key, pair in sorted(pairs.items()):
        require(set(pair) == {'AB', 'BA'}, f'Incomplete pair: {key}')
        a, b = pair['AB'], pair['BA']
        require(manifest[a]['audio_path'] == manifest[b]['audio_path'], 'Pair audio mismatch')
        require(rows[a]['duration_seconds'] == rows[b]['duration_seconds'], 'Pair duration mismatch')
        pa, pb = rows[a]['prediction'], rows[b]['prediction']
        ga, gb = manifest[a]['annotations'], manifest[b]['annotations']
        # Preserve the archived diagnostic scorer's endpoint-partition summation.
        # An algebraically equivalent union-length implementation can change
        # strict floating-point ties, and therefore the historical swap count.
        ca, cb = diagnostic_set_iou(ga, pa), diagnostic_set_iou(gb, pb)
        success = ca >= .5 and cb >= .5
        swap = ca <= diagnostic_set_iou(gb, pa) or cb <= diagnostic_set_iou(ga, pb)
        successes.append(success)
        swaps.append(swap)
        joints.append(success and not swap)
        same.append(pa == pb)
    n = len(pairs)
    return {'query_count': len(rows), 'pair_count': n, 'mIoU': public_metrics(rows)['mIoU'],
            'PairAcc@0.5': 100 * sum(successes) / n, 'SwapError': 100 * sum(swaps) / n,
            'JointPairAcc@0.5': 100 * sum(joints) / n, 'same_answer_pairs': sum(same)}


def compare(actual, archived):
    for key, value in actual.items():
        require(math.isclose(value, archived[key], rel_tol=0, abs_tol=1e-8), f'Summary mismatch: {key}')


def read_rows(path):
    text = path.read_text(encoding='utf-8')
    require(text.endswith('\n'), 'Interrupted JSONL tail')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def verify():
    provenance = load(EVIDENCE / 'provenance.json')
    for item in provenance['files']:
        require(text_hash(ROOT / item['target']) == item.get('released_sha256', item['sha256']),
                'Archive hash mismatch: ' + item['target'])
    extra = load(ROOT / 'reproduction/archive/provenance.json')
    for item in extra['files']:
        require(text_hash(ROOT / item['target']) == item.get('released_sha256', item['sha256']),
                'Recovered file hash mismatch: ' + item['target'])
    spot_root = ROOT / 'results/paper_spotsound'
    spot_provenance = load(spot_root / 'provenance.json')
    for item in spot_provenance['files']:
        data = (ROOT / item['target']).read_bytes().replace(b'\r\n', b'\n')
        if item.get('terminal_newline_added'):
            require(data.endswith(b'\n'), 'Missing normalized terminal newline')
            data = data[:-1]
        require(hashlib.sha256(data).hexdigest() == item['sha256'],
                'Source hash mismatch: ' + item['target'])
    spec = load(ROOT / 'experiments/paper/release.json')
    profiles = load(ROOT / 'configs/release_profiles.json')
    checkpoint_manifest = load(ROOT / 'checkpoints/manifest.json')
    checkpoint_hashes = {f['path']: f['sha256'] for f in checkpoint_manifest['files']}
    require(checkpoint_hashes[profiles['spotsound']['checkpoint'] + '/adapter_model.safetensors']
            == profiles['spotsound']['checkpoint_sha256'] == spot_provenance['checkpoint_sha256'],
            'SpotSound release checkpoint mismatch')
    require(checkpoint_hashes[profiles['timeaudio']['checkpoint']]
            == profiles['timeaudio']['checkpoint_sha256'], 'TimeAudio release checkpoint mismatch')
    train = load(EVIDENCE / 'train_summary.json')
    require(train['delta_sha256'] == profiles['timeaudio']['checkpoint_sha256'],
            'TimeAudio train checkpoint mismatch')
    spot_train = load(spot_root / 'train_summary.json')
    for key, value in profiles['spotsound']['training'].items():
        if key != 'entry':
            require(spot_train[key] == value, 'SpotSound training profile mismatch: ' + key)
    table1 = next(r for r in spec['main_table'] if r['method'] == 'TimeAudio + RelTwin')
    report = {'verification': 'CPU rescore of archived predictions, not new inference or training',
              'paper_pdf_sha256': spec['paper']['pdf_sha256'],
              'archive_file_hashes_checked': len(provenance['files']) + len(extra['files']) + len(spot_provenance['files']),
              'table1_timeaudio': {}, 'table2_timeaudio': {},
              'table1_spotsound': {}, 'table2_spotsound': {}}
    for i, (folder, count) in enumerate([('spotsound400', 400), ('clotho1000', 6649), ('unav100', 100)]):
        path = EVIDENCE / folder
        summary = load(path / 'summary.json')
        require(summary['completed_count'] == summary['expected_count'] == count, 'Archived count mismatch')
        require(summary['delta_sha256'] == train['delta_sha256'], 'Checkpoint mismatch')
        require(summary['official_checkpoint_sha256'] == train['official_checkpoint_sha256'], 'Base checkpoint mismatch')
        require(summary['predictions_sha256'] == text_hash(path / 'predictions.jsonl'), 'Prediction hash mismatch')
        rows = aligned_rows(read_rows(path / 'predictions.jsonl'), count)
        metrics = public_metrics(rows)
        compare(metrics, summary)
        compare(score_public(rows, 'timeaudio', profile='archive'), summary)
        values = [round(metrics[m], 2) for m in spec['metric_order']]
        require(values == table1['values'][i], 'Paper Table 1 mismatch: ' + folder)
        report['table1_timeaudio'][folder] = {'queries': count, **metrics, 'paper_rounded_match': True}
    manifest = load(EVIDENCE / 'table2_manifest.json')
    for folder, name in [('official', 'Official checkpoint'), ('sft', 'Paired SFT'), ('reltwin', 'RelTwin')]:
        path = EVIDENCE / ('table2_' + folder)
        summary = load(path / 'relation_metrics_300.json')
        require(summary['manifest_sha256'] == text_hash(EVIDENCE / 'table2_manifest.json'), 'Manifest hash mismatch')
        require(summary['predictions_sha256'] == text_hash(path / 'combined_predictions.jsonl'), 'Diagnostic hash mismatch')
        metrics = relation_metrics(read_rows(path / 'combined_predictions.jsonl'), manifest)
        compare(metrics, summary)
        row = next(r for r in spec['relation_table']['rows'] if r['backbone'] == 'TimeAudio' and r['training'] == name)
        values = [round(metrics[m], 2) for m in ['mIoU', 'JointPairAcc@0.5', 'SwapError']] + [metrics['same_answer_pairs']]
        require(values == row['values'], 'Paper Table 2 mismatch: ' + name)
        report['table2_timeaudio'][folder] = {**metrics, 'paper_rounded_match': True}
    table1 = next(r for r in spec['main_table'] if r['method'] == 'SpotSound-A + RelTwin')
    for i, (folder, count) in enumerate([('spotsound400', 400), ('clotho6649', 6649), ('unav100', 100)]):
        path = spot_root / folder
        summary = load(path / 'summary.json')
        require(summary['expected_count'] == summary['completed_count'] == count, 'SpotSound count mismatch')
        # Older summaries identify the adapter by path; the 300-pair summary
        # additionally records its binary hash. All point to the same adapter.
        require(summary['adapter'].replace('\\', '/').endswith(
            '/reltwin_review_controls_20260912/no_exchange_seed0/adapter'), 'SpotSound adapter path mismatch')
        if 'adapter_sha256' in summary:
            require(summary['adapter_sha256'] == spot_provenance['checkpoint_sha256'], 'SpotSound adapter mismatch')
        require(summary.get('delta') is None, 'Unexpected bridge delta')
        rows = aligned_rows(read_rows(path / 'predictions.jsonl'), count)
        native = public_metrics(rows)
        compare({k: v for k, v in native.items() if k != 'parse_failure_count'}, summary)
        metrics = score_public(rows, 'spotsound', profile='archive')
        if folder == 'unav100':
            compare(metrics, load(path / 'metrics.json')['spotsound_protocol'])
        require([round(metrics[m], 2) for m in spec['metric_order']] == table1['values'][i],
                'Paper Table 1 mismatch: ' + folder)
        report['table1_spotsound'][folder] = {'queries': count, **metrics, 'paper_rounded_match': True}
    for folder, name in [('official', 'Official checkpoint'), ('sft256', 'Paired SFT'),
                         ('candidate256', 'Candidate supervision')]:
        path = spot_root / ('table2_' + folder)
        summary = load(path / 'summary.json')
        require(summary['expected_count'] == summary['completed_count'] == len(manifest), 'Table 2 count mismatch')
        if folder == 'candidate256':
            require(summary['adapter_sha256'] == spot_provenance['checkpoint_sha256'], 'Candidate checkpoint mismatch')
        rows = read_rows(path / 'predictions.jsonl')
        metrics = relation_metrics(rows, manifest)
        compare({k: v for k, v in public_metrics(rows).items() if k != 'parse_failure_count'}, summary)
        row = next(r for r in spec['relation_table']['rows'] if r['backbone'] == 'SpotSound-A' and r['training'] == name)
        values = [round(metrics[m], 2) for m in ['mIoU', 'JointPairAcc@0.5', 'SwapError']] + [metrics['same_answer_pairs']]
        require(values == row['values'], 'Paper Table 2 mismatch: ' + name)
        report['table2_spotsound'][folder] = {**metrics, 'paper_rounded_match': True}
    report['archived_prediction_tables_verified'] = True
    report['full_training_reproduction_certified'] = False
    return report


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
