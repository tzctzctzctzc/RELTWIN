# RelTwin: Contrasting Locally Valid Timestamp Answers for Query-Specific Audio Grounding

RelTwin is a two-stage fine-tuning framework for audio temporal grounding. It learns to select the occurrence requested by a query when several acoustically valid windows coexist in one recording.

[Paper](https://github.com/tzctzctzctzc/RELTWIN/blob/652c5cf420b1c0d5ab63d1620ab826a466bc00b6/paper.pdf) · [Training and evaluation](docs/PAPER_REPRODUCTION.md) · [Checkpoints](checkpoints/manifest.json)

## Method

1. **Twin construction:** reuse sound excerpts in opposite-order windows. Reversing the query exchanges the correct window.
2. **Stage 1 — complete-answer competition:** score complete timestamp answers, learn query-conditioned preferences, and enforce exchange consistency. Localization replay retains ordinary supervision.
3. **Stage 2 — quality-aware interval-set preferences:** compare six interval sets using set-IoU, threshold margin, interval count, and boundary accuracy. Mean quality and Pareto dominance define soft targets.

RelTwin supports SpotSound-A and TimeAudio. Inference directly generates timestamps from an audio recording and a text query.

## Paper results — Table 1

Each cell is **R1@0.3 / R1@0.5 / mIoU (%)**, as reported in Table 1. Baseline rows follow the SpotSound paper, Table 3.

Both adapted backbones use full two-stage RelTwin: 384 updates of query-conditioned answer learning followed by 128 updates of set-level preference learning. Exchange consistency, localization replay, and bridge adaptation are enabled. The shared training settings and backbone-specific commands are in the [reproduction guide](docs/PAPER_REPRODUCTION.md#full-reltwin-training-configuration).

| Method | SpotSound-Bench (400 queries) | Clotho-Moment (6,649 queries) | UnAV-100 subset (100 queries) |
|---|---|---|---|
| TimeAudio | 7.80 / 1.80 / 10.20 | 39.50 / 24.90 / 28.60 | 21.00 / 7.00 / 16.00 |
| TimeAudio + RelTwin | 24.25 / 13.25 / 18.36 | 45.63 / 30.92 / 34.08 | 44.00 / 25.00 / 30.48 |
| SpotSound-A | 76.50 / 59.50 / 57.90 | 93.40 / 91.00 / 85.60 | 86.00 / 74.00 / 69.80 |
| SpotSound-A + RelTwin | **78.00 / 62.00 / 59.43** | **93.61 / 91.01 / 86.68** | **89.00 / 79.00 / 74.03** |

mIoU gains over the corresponding backbone: TimeAudio **+8.16 / +5.48 / +14.48**; SpotSound-A **+1.53 / +1.08 / +4.23**.

## Relation diagnostics — Table 2

Evaluation uses **300 inverse-query pairs (600 queries)**, separate from the public benchmarks. SpotSound-A isolates Stage-1 candidate supervision after 256 updates; TimeAudio compares paired SFT with full RelTwin after 384+128 updates.

| Backbone | Training | mIoU | JointPairAcc@0.5 | Swap error | Same-answer pairs |
|---|---|---:|---:|---:|---:|
| SpotSound-A | Official checkpoint | 37.19 | 0.67 | 66.33 | 32/300 |
| SpotSound-A | Paired SFT | 76.08 | 55.33 | 36.33 | 61/300 |
| SpotSound-A | Candidate supervision | 87.70 | 81.67 | 7.00 | 2/300 |
| TimeAudio | Official checkpoint | 40.23 | 0.00 | 83.67 | 67/300 |
| TimeAudio | Paired SFT | 38.89 | 2.67 | 78.33 | 160/300 |
| TimeAudio | RelTwin | 43.86 | 10.67 | 54.33 | 34/300 |

Results use seed 0. The timing intervention retains a 20-point JointPairAcc advantage on 40 changed recordings (87.50 vs. 67.50; grouped-bootstrap CI [2.94, 33.33]).

## Reproduction and release contents

See the [guide](docs/PAPER_REPRODUCTION.md) for environment setup, training, and evaluation commands.

- Full two-stage training and direct evaluation code for **both SpotSound-A and TimeAudio**, with bridge loading, shared objectives, and metric utilities.
- **Adaptation checkpoints** for the released predictions, distributed through Git LFS: [checksum manifest](checkpoints/manifest.json). Run `git lfs pull`, then `python scripts/verify_release_checkpoints.py`.
- [SpotSound-A predictions](results/paper_spotsound/) and [TimeAudio predictions](results/paper_timeaudio/), with per-query outputs, evaluation summaries, and checksum verification.
- [Full two-stage training launcher](reproduction/train_paper.sh), a shared [paper scorer](scripts/paper_metrics.py), [checkpoint configurations](configs/release_profiles.json), and a compact [file index](docs/PAPER_ALIGNMENT.md).
- No upstream backbone weights or benchmark audio are bundled. Obtain upstream assets under their respective licenses.

Check consistency without a GPU:

```bash
python scripts/check_paper_release.py
python scripts/verify_release_checkpoints.py
python scripts/verify_paper_evidence.py
python -m pytest -q tests/test_paper_launchers.py tests/test_paper_release.py tests/test_paper_evidence.py tests/test_paper_metrics.py tests/test_interval_metrics.py tests/test_setpo_candidates.py
```

These checks validate file identities and rescore saved predictions without a GPU. For new training or model inference, follow the guide using the corresponding backbone environment.

## Licensing and dependencies

Original repository code is MIT licensed; upstream code, weights, audio, and derivative adaptation weights remain subject to applicable upstream terms. The historical SpotSound environment is pinned in [requirements.txt](requirements.txt); TimeAudio uses its [separate archived environment](reproduction/archive/timeaudio_environment.txt). Consult the guide before running experiments.
