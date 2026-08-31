# RelTwin-SetPO for SpotSound

This repository contains the code, locked protocols, and raw evidence for a relation-aware, setwise preference optimization extension of [SpotSound](https://github.com/LoieSun/SpotSound). The target task is open-vocabulary audio temporal grounding on SpotSound-Bench.

The method has two training stages:

1. **RelTwin-RBEE** (Relation-Boundary Exchange Equivariance) constructs the same audio with both `A→B` and `B→A` windows. It trains the model to prefer the query-consistent boundary sequence and makes the two inverse-query candidate distributions exchange-equivariant.
2. **SetPO** (Setwise Preference Optimization) scores six deterministic interval-set candidates. The target distribution combines temporal set IoU and symmetric soft interval F1, while a Jensen-Shannon term enforces relation exchange and ordinary localization rehearsal limits forgetting.

This is not an architecture-module stack. The contribution is a paired-data construction plus a task-specific listwise objective over structured interval sets. Its ingredients—preference learning, equivariance, and rehearsal—are established ideas; the novelty claim is their formulation for inverse audio-relation queries and multi-interval boundary quality.

## Evidence status

All headline comparisons use the same 400-row evaluator and start from the same official SpotSound-A checkpoint. We do not select the best seed. The adaptation does use 256 newly composed ESC-50 relation audios plus 512 ordinary-query rehearsal rows, so this is an **open-extra-data adaptation** comparison, not a same-training-data comparison to the original 77.6k-sample SpotSound training run (whose full corpus was not released).

| system | seeds | SpotSound mIoU | R1@.3 | R1@.5 | interpretation |
|---|---:|---:|---:|---:|---|
| SpotSound-A, paper Table 3 | 1 | 57.90 | 76.50 | 59.50 | published reference |
| SpotSound-A, reconstructed same harness | 1 | 58.317 | 77.00 | 60.75 | matched baseline |
| ordinary SFT, seed 0 | 1 | 58.313 | 76.50 | 61.75 | controlled ablation; no mIoU gain |
| E002 RBEE-256, mean ± SD | 3 | **59.166 ± 0.020** | 76.917 ± 0.629 | **62.417 ± 0.144** | all three mIoU seeds exceed the matched baseline |
| E002 RBEE-256 → SetPO-64, mean ± SD | 3 | **59.193 ± 0.068** | 76.583 ± 0.629 | **62.417 ± 0.289** | audited mIoU point-estimate SOTA; not statistically significant vs. baseline |
| E001 SetPO on unequal parents, mean of 3 | 3 | 58.593 | 76.75 | 61.417 | failed stability audit; not a SOTA result |

E002 passed its independent RelTwin development gate in every seed. On SpotSound-Bench, SetPO improves the primary mIoU point estimate over the same-harness official checkpoint by `+0.875` point, and all three seeds exceed that checkpoint. The conservative hierarchical seed-record bootstrap 95% interval is nevertheless `[-0.243, +2.043]` points (`p_nonpositive=0.063`). The preregistered significance criterion therefore **fails**: this repository claims a new audited point estimate, not a statistically significant improvement. R1@.5 also improves by `+1.667` points, while R1@.3 is `-0.417` point below the same-harness checkpoint.

SetPO is mechanistically supported on the class-disjoint RelTwin development task: relative to its matched RBEE parent it raises mIoU by `+3.358` points (95% CI `[+2.469, +4.275]`) and PairAcc@.5 by `+5.208` points (95% CI `[+2.917, +7.708]`). On the public benchmark, however, SetPO adds only `+0.027` mIoU point over RBEE (95% CI `[-0.313, +0.412]`). The robust public gain is therefore attributable primarily to RBEE; SetPO should not be presented as an independently validated public-set improvement.

E001 is intentionally retained as a negative result. Its seed-0 parent used the full 256-step/512-group RBEE stage, while seed 1 and 2 used earlier 64-step/32-group parents. The mismatch was detected after evaluation; the [original protocol](experiments/protocols/E001_public_multiseed.json) is preserved verbatim and the correction is isolated in a [post-hoc amendment](experiments/protocols/E001_posthoc_amendment.json). E002 repeats the complete pipeline under [a locked matched protocol](experiments/protocols/E002_matched_pipeline_multiseed.json).

We call a result an **open-extra-data, same-harness point-estimate SOTA** only if the locked three-seed mean exceeds both the paper value and the same-harness official baseline. The dated literature audit found no later indexed SpotSound-Bench result. Statistical uncertainty is reported separately with paired record and hierarchical seed-record bootstrap intervals; a confidence interval crossing zero is never described as a significant improvement.

## Reproduction pins

| artifact | source | pinned revision |
|---|---|---|
| SpotSound paper | [arXiv v2](https://arxiv.org/abs/2604.13023v2) | v2, 2026-08-10 |
| official code | [LoieSun/SpotSound](https://github.com/LoieSun/SpotSound) | `d60c0e214ab685d0aa916db655fc59191900fb92` |
| SpotSound-A adapter | [Loie/SpotSound](https://huggingface.co/Loie/SpotSound) | `07eca9f048599fde92a55f2376e429deaa73c21a` |
| Audio Flamingo 3 HF base | [nvidia/audio-flamingo-3-hf](https://huggingface.co/nvidia/audio-flamingo-3-hf) | `7d4bae64ee29878af6504ae6f6bb3e40492838ad` |
| SpotSound-Bench | [Loie/SpotSound-Bench](https://huggingface.co/datasets/Loie/SpotSound-Bench) | `2e64ad197b90d6bd3b43efc73738cb9f86810735` |

The benchmark contains 400 annotation triplets but 387 unique audio files: 13 additional query rows reuse audio, across 11 multiply queried files. No annotation audio is missing. Evaluation must therefore operate on all 400 rows, not deduplicate by audio filename.

## Environment

The measured environment used Python 3.10, one 32 GB NVIDIA GPU, PyTorch 2.8.0+cu128, Transformers 5.9.0, and PEFT 0.18.1.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the pinned assets with a recent Hugging Face CLI:

```bash
hf download nvidia/audio-flamingo-3-hf \
  --revision 7d4bae64ee29878af6504ae6f6bb3e40492838ad \
  --local-dir models/audio-flamingo-3-hf
hf download Loie/SpotSound \
  --revision 07eca9f048599fde92a55f2376e429deaa73c21a \
  --local-dir models/SpotSound
hf download Loie/SpotSound-Bench --repo-type dataset \
  --revision 2e64ad197b90d6bd3b43efc73738cb9f86810735 \
  --local-dir datasets/SpotSound-Bench
```

The scripts expect a normalized benchmark annotation file with the official list-of-dictionaries schema (`audio_path`, `caption`, `annotations`). The exact evaluated annotation hash and preparation audit are shipped with the result ledger.

## Build the independent RelTwin development set

Download the official [ESC-50](https://github.com/karolpiczak/ESC-50) release, then run:

```bash
python scripts/build_esc50_reltwin.py \
  --metadata data/ESC-50/meta/esc50.csv \
  --audio-dir data/ESC-50/audio \
  --output-dir data/reltwin_esc50_v1/audio \
  --train-manifest data/reltwin_esc50_v1/train.json \
  --test-manifest data/reltwin_esc50_v1/test.json \
  --rehearsal-manifest data/reltwin_esc50_v1/rehearsal.json \
  --train-audios 256 --test-pairs 40 --test-replicates 2 --seed 0
```

Ten ESC-50 classes are held out from training. The test set contains 320 inverse-relation queries; SpotSound-Bench is not used to select hyperparameters or checkpoints.

## Run the locked three-seed pipeline

```bash
PROJECT_ROOT="$PWD" \
BASE_MODEL=models/audio-flamingo-3-hf \
SPOTSOUND_ADAPTER=models/SpotSound \
RELTWIN_DATA=data/reltwin_esc50_v1 \
SPOTSOUND_ANNOTATIONS=datasets/SpotSound-Bench/annotations_processed.json \
SPOTSOUND_AUDIO_DIR=datasets/SpotSound-Bench/audio \
OUTPUT_DIR=outputs/e002_matched_pipeline \
bash scripts/run_e002_matched_pipeline.sh
```

The script first trains and evaluates RBEE and SetPO on RelTwin. Public SpotSound evaluation begins only if every SetPO seed improves its matched RBEE parent on held-out RelTwin mIoU and PairAcc@.5 and the three-seed means improve on both metrics.

Aggregate public predictions against the official same-harness prediction file:

```bash
python scripts/summarize_public_multiseed.py \
  --baseline results/official_spotsound_a/public_predictions.jsonl \
  --method \
    outputs/e002_matched_pipeline/seed_0/setpo/public_predictions.jsonl \
    outputs/e002_matched_pipeline/seed_1/setpo/public_predictions.jsonl \
    outputs/e002_matched_pipeline/seed_2/setpo/public_predictions.jsonl \
  --output outputs/e002_matched_pipeline/public_multiseed.json \
  --samples 50000 --seed 20260831
```

## Tests

The metric, candidate-construction, relation-objective, and token-KL tests do not load the base model:

```bash
python -m pytest -q
```

## Evaluation caveat and licensing

The official SpotSound repository releases inference and training code but no batch evaluator or official prediction file. The evaluator here reconstructs the paper protocol from the released response format and temporal-set IoU definition, and reproduces the paper mIoU within `+0.42` point on the official checkpoint. Consequently, claims are qualified as **same-harness reconstructed-evaluator** results.

The original code in this repository is MIT licensed. That does not relicense upstream artifacts. The SpotSound Hugging Face model card declares MIT, but the pinned official GitHub archive contains no `LICENSE` file; users should not infer a code license for that separate archive. Audio Flamingo 3 is restricted to non-commercial research by its own model terms; ESC-50 and benchmark audio retain their source licenses. No model weights or benchmark audio are committed here.
