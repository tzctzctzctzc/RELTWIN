# NOVA + RelTwin-SetPO for SpotSound

This repository contains the code, locked protocols, and raw evidence for a relation-aware adaptation and a label-free candidate router for [SpotSound](https://github.com/LoieSun/SpotSound). The target task is open-vocabulary audio temporal grounding on SpotSound-Bench.

The pipeline has two training stages and one inference stage:

1. **RelTwin-RBEE** (Relation-Boundary Exchange Equivariance) constructs the same audio with both `A→B` and `B→A` windows. It trains the model to prefer the query-consistent boundary sequence and makes the two inverse-query candidate distributions exchange-equivariant.
2. **SetPO** (Setwise Preference Optimization) scores six deterministic interval-set candidates. The target distribution combines temporal set IoU and symmetric soft interval F1, while a Jensen-Shannon term enforces relation exchange and ordinary localization rehearsal limits forgetting.
3. **NOVA** (Necessity-Oriented Verification over Audio) treats the official, SFT, and SetPO predictions as candidates. A frozen SpotSound detector checks whether each predicted interval is sufficient when kept and necessary when removed. A conservative router is fitted only on source-disjoint 60-second LongNeedle-ESC50 mixtures and retains SetPO unless a challenger has calibrated positive value.

This is not an indiscriminate architecture-module stack. The training contribution is a paired-data construction plus a task-specific listwise objective over structured interval sets. The inference contribution is a candidate-level causal audit: preserve the timeline, keep or remove the proposed interval set, and use the resulting sufficiency/necessity evidence for conservative checkpoint routing. Preference learning, equivariance, rehearsal, occlusion, and linear routing are established ingredients; the novelty claim is their formulation for inverse audio relations and set-valued audio temporal grounding.

## Evidence status

All headline comparisons use the same 400-row evaluator and start from the same official SpotSound-A checkpoint. We do not select the best seed. The adaptation does use 256 newly composed ESC-50 relation audios plus 512 ordinary-query rehearsal rows, so this is an **open-extra-data adaptation** comparison, not a same-training-data comparison to the original 77.6k-sample SpotSound training run (whose full corpus was not released).

| system | seeds | SpotSound mIoU | R1@.3 | R1@.5 | interpretation |
|---|---:|---:|---:|---:|---|
| SpotSound-A, paper Table 3 | 1 | 57.90 | 76.50 | 59.50 | published reference |
| SpotSound-A, reconstructed same harness | 1 | 58.317 | 77.00 | 60.75 | matched baseline |
| ordinary SFT, seed 0 | 1 | 58.313 | 76.50 | 61.75 | controlled ablation; no mIoU gain |
| **E004 NOVA, locked full-feature router** | deterministic | **59.329** | **78.75** | 62.25 | new point-estimate SOTA; not statistically significant |
| E002 RBEE-256, mean ± SD | 3 | **59.166 ± 0.020** | 76.917 ± 0.629 | **62.417 ± 0.144** | all three mIoU seeds exceed the matched baseline |
| E002 RBEE-256 → SetPO-64, mean ± SD | 3 | **59.193 ± 0.068** | 76.583 ± 0.629 | **62.417 ± 0.289** | previous point-estimate SOTA |
| E001 SetPO on unequal parents, mean of 3 | 3 | 58.593 | 76.75 | 61.417 | failed stability audit; not a SOTA result |

E002 passed its independent RelTwin development gate in every seed. On SpotSound-Bench, SetPO improves the primary mIoU point estimate over the same-harness official checkpoint by `+0.875` point, and all three seeds exceed that checkpoint. The conservative hierarchical seed-record bootstrap 95% interval is nevertheless `[-0.243, +2.043]` points (`p_nonpositive=0.063`). The preregistered significance criterion therefore **fails**: this repository claims a new audited point estimate, not a statistically significant improvement. R1@.5 also improves by `+1.667` points, while R1@.3 is `-0.417` point below the same-harness checkpoint.

E004 fixes SetPO seed 1 as a strong fallback and fits NOVA on LongNeedle-ESC50, whose 60-second clips have 4.76% mean target density. On the 43-row held-out development partition, NOVA reaches 24.358 mIoU versus 23.334 for the best single candidate and captures 74.9% of candidate-oracle headroom. The frozen router then reaches **59.329 mIoU** on SpotSound-Bench, `+0.095` point over the previous best single checkpoint and `+1.011` over official SpotSound-A. Its paired-record 95% interval against the previous best is `[-0.706, +0.926]` point (`p_nonpositive=0.415`), so the improvement is a **point estimate only**. It improves R1@.3 by `+1.5` points, leaves R1@.5 unchanged, and lowers R1@.7 by `-0.5` point.

The public rerun is iterative evidence, not a pristine confirmatory test: E003 public aggregates had already been observed before E004 was designed. E004 never uses SpotSound labels for coefficients, normalization, thresholds, or row selection; all fitting is on LongNeedle-ESC50. The locked full-feature result is the headline. Post-result ablations reach 59.515 with Keep-only verification and 59.536 with a seven-switch geometry guard, but are explicitly exploratory.

SetPO is mechanistically supported on the class-disjoint RelTwin development task: relative to its matched RBEE parent it raises mIoU by `+3.358` points (95% CI `[+2.469, +4.275]`) and PairAcc@.5 by `+5.208` points (95% CI `[+2.917, +7.708]`). On the public benchmark, however, SetPO adds only `+0.027` mIoU point over RBEE (95% CI `[-0.313, +0.412]`). The robust public gain is therefore attributable primarily to RBEE; SetPO should not be presented as an independently validated public-set improvement.

### Diagnostic Oracle headroom

The [Oracle analysis](results/e002/oracle_headroom.json) uses ground-truth IoU to select a prediction per benchmark row. It is a diagnostic upper bound and is **not** a deployable method or SOTA result.

| Oracle candidate pool | mIoU | R1@.3 | R1@.5 |
|---|---:|---:|---:|
| three RBEE seeds | 59.924 | 77.75 | 63.25 |
| three SetPO seeds | 60.073 | 78.00 | 63.50 |
| all six adapted models | 60.466 | 78.50 | 63.50 |
| official + ordinary SFT + all six adapted models | **62.462** | **81.25** | **65.75** |

The full Oracle improves 199 of 400 rows and never regresses because the official prediction is included as a fallback. Its largest headroom is on audio longer than 60 seconds (`+7.752` mIoU points) and rows where the official checkpoint has IoU below 0.3 (`+7.742` points). This motivated NOVA. E003 showed that a high-density two-event synthetic set reverses the public candidate ranking and fails to generalize. E004 therefore uses an independent low-density LongNeedle distribution and a SetPO fallback; SpotSound ground truth is never consumed by the fitting or application code.

E001 is intentionally retained as a negative result. Its seed-0 parent used the full 256-step/512-group RBEE stage, while seed 1 and 2 used earlier 64-step/32-group parents. The mismatch was detected after evaluation; the [original protocol](experiments/protocols/E001_public_multiseed.json) is preserved verbatim and the correction is isolated in a [post-hoc amendment](experiments/protocols/E001_posthoc_amendment.json). E002 repeats the complete pipeline under [a locked matched protocol](experiments/protocols/E002_matched_pipeline_multiseed.json).

We call a result an **open-extra-data, same-harness point-estimate SOTA** only if its locked aggregate exceeds the paper value, the same-harness official baseline, and every earlier audited result in this repository. The dated literature audit found no later indexed SpotSound-Bench result. Statistical uncertainty is reported separately; a confidence interval crossing zero is never described as a significant improvement.

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

## Build and gate NOVA on LongNeedle

LongNeedle uses unused ESC-50 source clips to build 160 dense 60-second mixtures. Targets last 1.2–2.5 seconds per occurrence and occupy 4.76% of the waveform on average.

```bash
python scripts/build_esc50_longneedle.py \
  --metadata data/ESC-50/meta/esc50.csv \
  --audio-dir data/ESC-50/audio \
  --output-dir data/longneedle_esc50_v1/audio \
  --manifest data/longneedle_esc50_v1/test.json \
  --exclude-manifest data/reltwin_esc50_v1/train.json \
  --exclude-manifest data/reltwin_esc50_v1/test.json

PROJECT_ROOT="$PWD" \
BASE_MODEL=models/audio-flamingo-3-hf \
OFFICIAL_ADAPTER=models/SpotSound \
SFT_ADAPTER=outputs/reltwin_esc50_v1/sft/adapter \
SETPO_ADAPTER=outputs/e002/seed_1/setpo/adapter \
LONGNEEDLE_DATA=data/longneedle_esc50_v1 \
OUTPUT_DIR=outputs/e004_longneedle \
bash scripts/run_e004_longneedle_gate.sh
```

The script stops with exit code 42 unless the hashed held-out partition exceeds every single candidate and captures at least 20% of SetPO-to-oracle headroom. Public routing is a separate label-free application step using the locked router; the committed [decision record](results/e004_longneedle/sota_decision.json) and raw predictions contain the exact result and caveats.

```bash
python scripts/extract_nova_features.py \
  --base models/audio-flamingo-3-hf \
  --verifier-adapter models/SpotSound \
  --manifest datasets/SpotSound-Bench/annotations_processed.json \
  --audio-dir datasets/SpotSound-Bench/audio \
  --candidate official=results/official_spotsound_a/public_predictions.jsonl \
  --candidate sft=results/controlled/sft_seed0/public_predictions.jsonl \
  --candidate setpo=results/e002/seed_1/setpo/public_predictions.jsonl \
  --output outputs/e004_public/features.jsonl

python scripts/apply_nova_router.py \
  --features outputs/e004_public/features.jsonl \
  --model outputs/e004_longneedle/router.json \
  --output outputs/e004_public/predictions.jsonl \
  --summary outputs/e004_public/summary.json
```

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

The metric, candidate-construction, relation-objective, counterfactual intervention, router, and token-KL tests do not load the base model:

```bash
python -m pytest -q
```

## Evaluation caveat and licensing

The official SpotSound repository releases inference and training code but no batch evaluator or official prediction file. The evaluator here reconstructs the paper protocol from the released response format and temporal-set IoU definition, and reproduces the paper mIoU within `+0.42` point on the official checkpoint. Consequently, claims are qualified as **same-harness reconstructed-evaluator** results.

The original code in this repository is MIT licensed. That does not relicense upstream artifacts. The SpotSound Hugging Face model card declares MIT, but the pinned official GitHub archive contains no `LICENSE` file; users should not infer a code license for that separate archive. Audio Flamingo 3 is restricted to non-commercial research by its own model terms; ESC-50 and benchmark audio retain their source licenses. No model weights or benchmark audio are committed here.
