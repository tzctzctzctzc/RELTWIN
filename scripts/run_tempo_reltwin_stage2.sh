#!/usr/bin/env bash
set -euo pipefail

project="${PROJECT:-/root/autodl-tmp/SpotSound-ICASSP}"
code="${CODE:-$project/spotsound_reltwin_crossbench_20260914}"
python_bin="${PYTHON_BIN:-$project/env-conda/bin/python}"
base="${TEMPO_BASE:-$project/models/tempo-sft-stage1-2}"
data_root="${TEMPO_DATA:-$project/datasets/TEMPO_trainval}"
out="${OUT:-$project/outputs/tempo_reltwin_lora_20260915}"

export PYTHONPATH="$code/scripts"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

while screen -ls 2>/dev/null | grep -Eq '[.]tempo_(ckpt|data)_download'; do
  sleep 20
done

test -s "$base/model-00004-of-00004.safetensors"
test -s "$base/time_proj.pt"
test -s "$data_root/data/audio_grounding/val/part-00000.parquet"
for part in 0 1 2 3 4; do
  test -s "$data_root/data/audio_grounding/sft_stage2/part-0000${part}.parquet"
done

mkdir -p "$data_root/materialized/val" "$data_root/materialized/train" "$out"

"$python_bin" "$code/scripts/prepare_tempo_grounding.py" \
  --parquet-dir "$data_root/data/audio_grounding/val" \
  --audio-dir "$data_root/materialized/val/audio" \
  --annotations "$data_root/materialized/val/annotations.json" \
  > "$data_root/materialized/val/prepare.log" 2>&1

"$python_bin" "$code/scripts/prepare_tempo_grounding.py" \
  --parquet-dir "$data_root/data/audio_grounding/sft_stage2" \
  --audio-dir "$data_root/materialized/train/audio" \
  --annotations "$data_root/materialized/train/annotations.json" \
  > "$data_root/materialized/train/prepare.log" 2>&1

mkdir -p "$out/base_val500"
"$python_bin" "$code/scripts/evaluate_tempo_checkpoint.py" \
  --base "$base" \
  --time-projector "$base/time_proj.pt" \
  --annotations "$data_root/materialized/val/annotations.json" \
  --audio-dir "$data_root/materialized/val/audio" \
  --predictions "$out/base_val500/predictions.jsonl" \
  --summary "$out/base_val500/inference.json" \
  > "$out/base_val500/inference.log" 2>&1

"$python_bin" "$code/scripts/score_tempo_grounding.py" \
  --predictions "$out/base_val500/predictions.jsonl" \
  --summary "$out/base_val500/metrics.json" \
  > "$out/base_val500/scoring.log" 2>&1

mkdir -p "$out/reltwin_rbee_seed0"
"$python_bin" "$code/scripts/train_tempo_reltwin_lora.py" \
  --base "$base" \
  --time-projector "$base/time_proj.pt" \
  --manifest "$project/autoresearch/06_experiments/data/reltwin_esc50_v1/train.json" \
  --audio-dir "$project/autoresearch/06_experiments/data/reltwin_esc50_v1/audio" \
  --rehearsal-manifest "$data_root/materialized/train/annotations.json" \
  --rehearsal-audio-dir "$data_root/materialized/train/audio" \
  --output-adapter "$out/reltwin_rbee_seed0/adapter" \
  --summary "$out/reltwin_rbee_seed0/train_summary.json" \
  --mode rbee \
  --steps 256 \
  --learning-rate 5e-6 \
  --rehearsal-weight 0.25 \
  --seed 0 \
  > "$out/reltwin_rbee_seed0/train.log" 2>&1

mkdir -p "$out/reltwin_rbee_seed0/val500"
"$python_bin" "$code/scripts/evaluate_tempo_checkpoint.py" \
  --base "$base" \
  --time-projector "$base/time_proj.pt" \
  --adapter "$out/reltwin_rbee_seed0/adapter" \
  --annotations "$data_root/materialized/val/annotations.json" \
  --audio-dir "$data_root/materialized/val/audio" \
  --predictions "$out/reltwin_rbee_seed0/val500/predictions.jsonl" \
  --summary "$out/reltwin_rbee_seed0/val500/inference.json" \
  > "$out/reltwin_rbee_seed0/val500/inference.log" 2>&1

"$python_bin" "$code/scripts/score_tempo_grounding.py" \
  --predictions "$out/reltwin_rbee_seed0/val500/predictions.jsonl" \
  --summary "$out/reltwin_rbee_seed0/val500/metrics.json" \
  > "$out/reltwin_rbee_seed0/val500/scoring.log" 2>&1

# Freeze the validation decision before touching the 5,151-example test set.
if "$python_bin" - "$out/base_val500/metrics.json" "$out/reltwin_rbee_seed0/val500/metrics.json" <<'PY'
import json
import sys

base, adapted = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:])
miou_gain = adapted["symmetric_mIoU"] - base["symmetric_mIoU"]
f1_gain = adapted["F1_IoU_0.5"] - base["F1_IoU_0.5"]
keep = (miou_gain > 0 or f1_gain > 0) and miou_gain >= -0.5 and f1_gain >= -0.5
raise SystemExit(0 if keep else 1)
PY
then
  printf 'continue_full_test\n' > "$out/gate_decision.txt"
  mkdir -p "$out/reltwin_rbee_seed0/test5151"
  "$python_bin" "$code/scripts/evaluate_tempo_checkpoint.py" \
    --base "$base" \
    --time-projector "$base/time_proj.pt" \
    --adapter "$out/reltwin_rbee_seed0/adapter" \
    --annotations "$project/datasets/TEMPO/annotations_processed.json" \
    --audio-dir "$project/datasets/TEMPO/audio" \
    --predictions "$out/reltwin_rbee_seed0/test5151/predictions.jsonl" \
    --summary "$out/reltwin_rbee_seed0/test5151/inference.json" \
    > "$out/reltwin_rbee_seed0/test5151/inference.log" 2>&1
  "$python_bin" "$code/scripts/score_tempo_grounding.py" \
    --predictions "$out/reltwin_rbee_seed0/test5151/predictions.jsonl" \
    --summary "$out/reltwin_rbee_seed0/test5151/metrics.json" \
    > "$out/reltwin_rbee_seed0/test5151/scoring.log" 2>&1
else
  printf 'stop_after_validation\n' > "$out/gate_decision.txt"
fi
