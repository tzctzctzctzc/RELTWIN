# Training, checkpoints, and evaluation

The [submitted paper](https://github.com/tzctzctzctzc/RELTWIN/blob/652c5cf420b1c0d5ab63d1620ab826a466bc00b6/paper.pdf) defines the task and reported tables. Table 1 uses full two-stage RelTwin for both backbones. This guide provides its training and inference commands, followed by an inventory of downloadable artifacts and saved-prediction checks.

## Download

```bash
git clone https://github.com/tzctzctzctzc/RELTWIN.git
cd RELTWIN
git lfs install
git lfs pull
python scripts/verify_release_checkpoints.py
python scripts/verify_paper_evidence.py
```

The adaptation weights total approximately 208 MB. Source ZIPs can contain LFS pointers; use `git lfs pull` to obtain the weights. The checksum verifier checks file sizes and SHA256 without deserializing tensors. Upstream backbone weights and audio are obtained separately under their respective licenses.

## Environments and assets

Use separate Python environments for the backbones. SpotSound's dependencies are in [requirements.txt](../requirements.txt); its archived evaluation used PyTorch 2.8.0+cu128. TimeAudio's [environment snapshot](../reproduction/archive/timeaudio_environment.txt) records PyTorch 2.6.0+cu124, Transformers 4.47.0, PEFT 0.16.0, and NumPy 2.2.6. Install the appropriate CUDA wheels for the local machine.

SpotSound uses the Audio Flamingo 3 base and official SpotSound adapter for initialization. TimeAudio uses [upstream source](https://github.com/lysanderism/TimeAudio) at `22db7136`, Vicuna-7B-v1.5, Whisper-large-v2, BEATs, BERT-base-uncased, and `timeaudio.pth`. The official TimeAudio checkpoint SHA256 is `2d1752697035435e01fdfa75d76c6ac2c0272fd14f209582d55dd3411c975016`.

The launchers accept an asset root with the following layout:

```text
ASSETS/
  models/audio-flamingo-3-hf/
  models/SpotSound/
  models/timeaudio-public/            # Vicuna, Whisper, BEATs, BERT, timeaudio.pth
  external/TimeAudio-22db7136/
  autoresearch/06_experiments/data/   # RelTwin, rehearsal, and grounding manifests
  datasets/                         # benchmark audio and training data
```

## Full RelTwin training configuration

Both backbones use query-conditioned answer learning in Stage 1 and set-level preference learning in Stage 2. Candidate competition, exchange consistency, localization replay, and bridge adaptation are enabled. The reported configuration is recorded in [`experiments/paper/release.json`](../experiments/paper/release.json).

Following Eq. (5), reference KL regularizes the replay candidate distribution in Stage 2. The paired objective consists of sequence supervision, set preferences, and exchange consistency.

| Setting | SpotSound-A + RelTwin | TimeAudio + RelTwin |
|---|---:|---:|
| Training entry | `scripts/train_spotsound_reltwin_full.py` | `scripts/train_timeaudio_reltwin_full.py` |
| Updates | 384 Stage 1 + 128 Stage 2 | 384 Stage 1 + 128 Stage 2 |
| LoRA learning rate | 2e-6 | 2e-6 |
| Bridge learning rate | 5e-6 | 5e-6 |
| Candidate / exchange weights | 1 / 1 | 1 / 1 |
| Replay weight | 0.75 | 0.75 |
| Stage-2 replay reference KL | 0.2 | 0.2 |
| Prediction / quality-target temperature | 1 / 0.15 | 1 / 0.15 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.1 | 8 / 32 / 0.1 |
| Trainable parameters | 37,627,392 | 31,759,104 |
| Seed | 0 | 0 |

`reproduction/train_paper.sh` selects the corresponding full trainer through `reproduction/train_full.sh`. SpotSound-A saves `final_adapter/` and `final_bridge.pt`; TimeAudio saves `final_delta.pt`, containing both LoRA and temporal bridge updates.

## Training

Select the backbone's environment and use a new output directory:

```bash
bash reproduction/train_paper.sh spotsound /path/to/ASSETS /path/to/new_spotsound_run
bash reproduction/train_paper.sh spotsound /path/to/ASSETS /path/to/spotsound_audit --audit-only
bash reproduction/train_paper.sh timeaudio /path/to/ASSETS /path/to/timeaudio_audit --audit-only
bash reproduction/train_paper.sh timeaudio /path/to/ASSETS /path/to/new_timeaudio_run
```

Both full trainers use category:relation scheduling weights of 3:1 and Clotho:scale-cardinality:TEMPO:AudioGrounding replay weights of 4:3:2:1. Each update accumulates one paired item and one replay item. Evaluation manifests supplied to the trainer are used for overlap checks, not training. Data-building utilities are in `scripts/`; the launchers list the required manifest paths. For another asset layout, invoke the training entry with explicit paths (`--help`).

## Inference

Use a fresh output directory for each checkpoint and configuration. Native evaluators resume existing JSONL rows, so reuse a directory only for the same interrupted run.

Evaluate the final two-stage outputs from the training commands above. SpotSound-A loads both the adapter and bridge:

```bash
python scripts/evaluate_checkpoint.py \
  --base /path/to/ASSETS/models/audio-flamingo-3-hf \
  --adapter /path/to/new_spotsound_run/final_adapter \
  --delta /path/to/new_spotsound_run/final_bridge.pt \
  --annotations /path/to/annotations_processed.json --audio-dir /path/to/audio \
  --predictions runs/spot_new/predictions.jsonl --summary runs/spot_new/native_summary.json \
  --limit 0 --max-new-tokens 128
```

TimeAudio:

```bash
python scripts/evaluate_timeaudio_checkpoint.py \
  --timeaudio-repo /path/to/ASSETS/external/TimeAudio-22db7136 \
  --config configs/timeaudio_paper.yaml \
  --llama /path/to/ASSETS/models/timeaudio-public/vicuna-7b-v1.5 \
  --whisper /path/to/ASSETS/models/timeaudio-public/whisper-large-v2 \
  --beats /path/to/ASSETS/models/timeaudio-public/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
  --bert /path/to/ASSETS/models/timeaudio-public/bert-base-uncased \
  --checkpoint /path/to/ASSETS/models/timeaudio-public/timeaudio.pth \
  --delta /path/to/new_timeaudio_run/final_delta.pt \
  --annotations /path/to/annotations_processed.json --audio-dir /path/to/audio \
  --predictions runs/ta_new/predictions.jsonl --summary runs/ta_new/native_summary.json \
  --limit 0 --max-new-tokens 128
```

These commands use direct timestamp generation with one beam, no sampling, and at most 128 new tokens. TimeAudio's paper settings are explicit in `configs/timeaudio_paper.yaml`.

## Downloadable checkpoints and saved predictions

| Artifact | Path | Recorded training |
|---|---|---|
| SpotSound-A candidate-supervision adapter | `checkpoints/spotsound/adapter` | 256 updates; Table 2 candidate control |
| TimeAudio full RelTwin delta | `checkpoints/timeaudio/delta.pt` | 384 Stage 1 + 128 Stage 2 |

The [checkpoint profiles](../configs/release_profiles.json) list each downloadable artifact's training settings and checksum. To evaluate the SpotSound-A candidate control, use its adapter path without `--delta`; to evaluate the TimeAudio artifact, use its delta path.

## Scoring

```bash
python scripts/paper_metrics.py --backbone spotsound --benchmark unav100 \
  --predictions runs/spot_new/predictions.jsonl
python scripts/paper_metrics.py --backbone timeaudio --benchmark unav100 \
  --predictions runs/ta_new/predictions.jsonl
```

Choose `spotsound`, `clotho`, or `unav100` for the benchmark. The scorer requires exactly 400, 6,649, or 100 unique indexed predictions, including empty answers. Both backbones use the same evaluation rule: predictions are clipped to audio duration, annotations are preserved, and an empty union scores zero. It reports R1@0.3, R1@0.5, and mean set-IoU in percent.

`--profile paper` is the default. For stored-record checks, `verify_paper_evidence.py` uses the corresponding archive profiles automatically; `paper_metrics.py --profile archive` also exposes these settings explicitly. Scoring reports the selected profile and input hash without modifying predictions.

Table 2 uses the shared `results/paper_timeaudio/table2_manifest.json`: 300 inverse-query pairs, 600 queries. SpotSound compares SFT and candidate supervision at 256 updates (candidate weight 0/1, exchange 0, replay 0.5). TimeAudio compares SFT with candidate competition disabled (`lambda=0`) against full RelTwin (`lambda=1`) at 384+128 updates. Recorded run settings are included with the prediction artifacts. JointPairAcc requires both targets to reach IoU 0.5 without a swap; ties count as swaps. Same-answer counts compare the predicted interval lists exactly.

## CPU checks

```bash
python scripts/check_paper_release.py
python scripts/check_paper_release.py --submitted-pdf /path/to/finalpaper.pdf
python scripts/verify_release_checkpoints.py
python scripts/verify_paper_evidence.py
python -m pytest -q tests/test_paper_launchers.py tests/test_paper_release.py tests/test_paper_evidence.py tests/test_paper_metrics.py tests/test_interval_metrics.py tests/test_setpo_candidates.py
```

These CPU checks validate submission identity, file checksums, query alignment, and stored metrics. Model-runtime tests (`test_spotsound_full_reltwin.py`, `test_timeaudio_reltwin*.py`, and `test_rbee_exchange_control.py`) use the corresponding model environment.
