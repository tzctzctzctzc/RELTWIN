"""Score timestamp predictions using the submitted paper's evaluation protocol."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from interval_metrics import normalize_intervals, temporal_set_iou


ROOT = Path(__file__).resolve().parents[1]


def score_public(rows, backbone, expected_count=None, *, profile='paper'):
    if backbone not in {'spotsound', 'timeaudio'}:
        raise ValueError('Unknown backbone')
    if profile not in {'paper', 'archive'}:
        raise ValueError('Unknown scoring profile')
    if not rows:
        raise ValueError('Empty evaluation')
    count = len(rows) if expected_count is None else expected_count
    if (len(rows) != count or any(type(r.get('index')) is not int for r in rows)
            or {r['index'] for r in rows} != set(range(count))):
        raise ValueError('Missing, duplicate, or displaced prediction indices')
    ious = []
    for row in rows:
        if row.get('status') != 'ok':
            raise ValueError('Non-ok prediction row')
        duration = row['duration_seconds']
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid audio duration')
        for spans in (row['prediction'], row['ground_truth']):
            for span in spans:
                if len(span) != 2 or not all(math.isfinite(x) for x in span) or span[1] <= span[0]:
                    raise ValueError('Invalid interval')
        prediction = normalize_intervals(row['prediction'], duration)
        target = (normalize_intervals(row['ground_truth'], duration)
                  if profile == 'archive' and backbone == 'spotsound' else row['ground_truth'])
        ious.append(temporal_set_iou(target, prediction))
    n = len(rows)
    return {'mIoU': 100 * math.fsum(ious) / n,
            'R1@0.3': 100 * sum(i >= .3 for i in ious) / n,
            'R1@0.5': 100 * sum(i >= .5 for i in ious) / n}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=['spotsound', 'timeaudio'], required=True)
    parser.add_argument('--benchmark', choices=['spotsound', 'clotho', 'unav100'], required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--profile', choices=['paper', 'archive'], default='paper',
                        help='paper: shared evaluation protocol; archive: stored artifact scoring settings')
    args = parser.parse_args(argv)
    profiles = json.loads((ROOT / 'configs/release_profiles.json').read_text(encoding='utf-8'))
    data = args.predictions.read_bytes().replace(b'\r\n', b'\n')
    if not data.endswith(b'\n'):
        raise ValueError('Interrupted JSONL tail')
    rows = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    result = score_public(rows, args.backbone, profiles['benchmarks'][args.benchmark], profile=args.profile)
    print(json.dumps({'backbone': args.backbone, 'benchmark': args.benchmark,
                      'scoring_profile': ('paper_prediction_clipping' if args.profile == 'paper'
                                          else profiles[args.backbone]['scoring']),
                      'queries': len(rows), 'predictions_sha256_lf': hashlib.sha256(data).hexdigest(),
                      **result}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
