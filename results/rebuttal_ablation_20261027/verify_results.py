"""Rescore the published predictions and paired bootstrap without a GPU or audio."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts'))
from analyze_reltwin_probe import set_iou
from interval_metrics import temporal_set_iou

ARMS = ('s0_candidate', 's0_paired_sft', 's0_candidate_js', 's1_paired_sft',
        's1_candidate', 's2_paired_sft', 's2_candidate', 's0_continue128',
        's0_setpo128', 's0_js_only')
BENCHES = ('relation600', 'spotsound400')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def recording(path):
    import re
    stem = Path(path).stem
    match = re.fullmatch(r'([A-Za-z0-9_-]{11})(?:_\d+(?:_\d+)?)?', stem)
    return 'youtube:' + match[1] if match else 'clip:' + Path(path).name


def groups_for(manifest, rows, relation=False):
    expected_count = 600 if relation else 400
    if len(rows) != expected_count or len(manifest) != expected_count:
        raise ValueError('Wrong sample count')
    if {r['index'] for r in rows} != set(range(expected_count)):
        raise ValueError('Duplicate or missing index')
    rows = sorted(rows, key=lambda r: r['index'])
    groups, pairs = defaultdict(lambda: defaultdict(list)), defaultdict(dict)
    for item, row in zip(manifest, rows):
        if (row['status'] != 'ok' or row['query'] != item['caption']
                or row['ground_truth'] != item['annotations']
                or row['audio'] != Path(item['audio_path']).name):
            raise ValueError('Prediction/manifest identity mismatch')
        # Preserve the evaluator's interval-union arithmetic at exact R1 ties.
        # The paired diagnostic below separately retains its endpoint scorer.
        iou = temporal_set_iou(item['annotations'], row['prediction'])
        if abs(iou - row['iou']) > 1e-10:
            raise ValueError('Stored IoU mismatch')
        group = Path(item['audio_path']).name if relation else recording(item['audio_path'])
        groups[group]['mIoU'].append(iou * 100)
        for threshold in (.3, .5, .7):
            groups[group][f'R1@{threshold}'].append(100. * (iou >= threshold))
        if relation:
            key = (item['pair_id'], item['template'], item.get('variant', 0))
            if item['relation'] in pairs[key]:
                raise ValueError('Duplicate query direction')
            pairs[key][item['relation']] = (item, row)
    if relation:
        if len(pairs) != 300:
            raise ValueError('Wrong inverse-pair count')
        for pair in pairs.values():
            if set(pair) != {'AB', 'BA'}:
                raise ValueError('Missing inverse direction')
            (ab, pa), (ba, pb) = pair['AB'], pair['BA']
            if ab['audio_path'] != ba['audio_path']:
                raise ValueError('Inverse pair uses different audio')
            ca = set_iou(ab['annotations'], pa['prediction'])
            cb = set_iou(ba['annotations'], pb['prediction'])
            swap = (ca <= set_iou(ba['annotations'], pa['prediction'])
                    or cb <= set_iou(ab['annotations'], pb['prediction']))
            group = Path(ab['audio_path']).name
            groups[group]['JointPairAcc@0.5'].append(100. * (ca >= .5 and cb >= .5 and not swap))
            groups[group]['swap_error'].append(100. * swap)
            groups[group]['same_answer_rate'].append(100. * (pa['prediction'] == pb['prediction']))
    return dict(groups)


def bootstrap(groups):
    names = sorted(groups)
    draws = np.random.default_rng(20261026).integers(len(names), size=(5000, len(names)))
    result = {}
    for metric in sorted(groups[names[0]]):
        counts = np.array([len(groups[k][metric]) for k in names])
        sums = np.array([sum(groups[k][metric]) for k in names])
        values = sums[draws].sum(1) / counts[draws].sum(1)
        result[metric] = dict(estimate=float(sums.sum() / counts.sum()),
                              ci95=np.quantile(values, [.025, .975]).tolist())
    return result


def contrast(groups, weights):
    first = groups[next(iter(weights))]
    if any(set(groups[k]) != set(first) for k in weights):
        raise ValueError('Unpaired audio groups')
    result = {}
    for name in sorted(first):
        result[name] = {}
        for metric in first[name]:
            arrays = [np.asarray(groups[k][name][metric]) for k in weights]
            if any(a.shape != arrays[0].shape for a in arrays):
                raise ValueError('Unpaired within-audio observations')
            result[name][metric] = sum(weights[k] * a for k, a in zip(weights, arrays)).tolist()
    return bootstrap(result)


def same(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('Report keys differ')
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('Report lengths differ')
        for a, b in zip(actual, expected):
            same(a, b)
    elif isinstance(expected, (float, int)):
        if not np.isfinite(actual) or abs(actual - expected) > 1e-8:
            raise ValueError(f'Report value differs: {actual} != {expected}')
    elif actual != expected:
        raise ValueError('Report value differs')


def main():
    inventory = read(HERE / 'release_manifest.json')
    for name, expected in inventory['files'].items():
        path = HERE / name
        if not path.resolve().is_relative_to(HERE.resolve()):
            raise ValueError('Unsafe release path')
        data = path.read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != expected['sha256_lf']:
            raise ValueError('Release checksum mismatch: ' + name)
    report = read(HERE / 'ablation_report.json')
    all_groups, rows_checked = {}, 0
    for arm in ARMS:
        for bench in BENCHES:
            manifest = read(HERE / 'manifests' / (bench + '.json'))
            rows = [json.loads(line) for line in (HERE / arm / bench / 'predictions.jsonl')
                    .read_text(encoding='utf-8').splitlines() if line.strip()]
            groups = groups_for(manifest, rows, bench == 'relation600')
            all_groups[(arm, bench)] = groups
            entry = report['arms'][arm][bench]
            same(entry['metrics'], bootstrap(groups))
            same(entry['queries'], len(rows)); same(entry['audio_groups'], len(groups))
            if bench == 'relation600':
                same(entry['same_answer_count'], round(entry['metrics']['same_answer_rate']['estimate'] * 3))
                same(entry['swap_error_count'], round(entry['metrics']['swap_error']['estimate'] * 3))
            rows_checked += len(rows)
    definitions = {f'candidate_minus_sft_seed{s}': {f's{s}_candidate': 1, f's{s}_paired_sft': -1}
                   for s in range(3)}
    definitions.update(js_without_competition={'s0_js_only': 1, 's0_paired_sft': -1},
        js_with_competition={'s0_candidate_js': 1, 's0_candidate': -1},
        factorial_interaction={'s0_candidate_js': 1, 's0_candidate': -1, 's0_js_only': -1, 's0_paired_sft': 1},
        stage2_minus_continue={'s0_setpo128': 1, 's0_continue128': -1},
        continue_minus_parent={'s0_continue128': 1, 's0_candidate_js': -1},
        stage2_minus_parent={'s0_setpo128': 1, 's0_candidate_js': -1})
    for name, weights in definitions.items():
        for bench in BENCHES:
            groups = {arm: all_groups[(arm, bench)] for arm in weights}
            same(report['contrasts'][name][bench], contrast(groups, weights))
    joint = read(HERE / 'multiseed_audio_bootstrap.json')
    for bench in BENCHES:
        weights = {f's{s}_{kind}': sign / 3 for s in range(3)
                   for kind, sign in (('candidate', 1), ('paired_sft', -1))}
        same(joint['benchmarks'][bench]['mean_gain_audio_ci'],
             contrast({arm: all_groups[(arm, bench)] for arm in weights}, weights))
        for metric, stats in report['multiseed'][bench].items():
            candidate = np.array([report['arms'][f's{s}_candidate'][bench]['metrics'][metric]['estimate']
                                  for s in range(3)])
            sft = np.array([report['arms'][f's{s}_paired_sft'][bench]['metrics'][metric]['estimate']
                            for s in range(3)])
            same(stats, dict(candidate_by_seed=candidate.tolist(), sft_by_seed=sft.tolist(),
                delta_by_seed=(candidate-sft).tolist(), candidate_mean=float(candidate.mean()),
                candidate_sd=float(candidate.std(ddof=1)), sft_mean=float(sft.mean()),
                sft_sd=float(sft.std(ddof=1)), delta_mean=float((candidate-sft).mean()),
                delta_sd=float((candidate-sft).std(ddof=1))))
    if rows_checked != 10000 or not report['complete']:
        raise ValueError('Incomplete result package')
    print('VERIFIED: 10 models, 10000 predictions, metrics, paired bootstrap and seed statistics.')


if __name__ == '__main__':
    main()
