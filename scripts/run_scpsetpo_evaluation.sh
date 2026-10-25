#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:?set PROJECT_ROOT}
python_bin=${PYTHON_BIN:?set PYTHON_BIN}
base_model=${BASE_MODEL:?set BASE_MODEL}
run_root=${RUN_ROOT:?set RUN_ROOT}
spotsound_root=${SPOTSOUND_ROOT:?set SPOTSOUND_ROOT}
clotho_root=${CLOTHO_ROOT:?set CLOTHO_ROOT}
output_root="$run_root/evaluation"
export PYTHONPATH="$project_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$output_root"

for _ in $(seq 1 480); do
  if [[ -s "$run_root/sc_psetpo/train_summary.json" && -s "$run_root/scalar_control/train_summary.json" ]]; then
    break
  fi
  if [[ -s "$run_root/sc_psetpo/train.exit" ]] && [[ "$(cat "$run_root/sc_psetpo/train.exit")" != 0 ]]; then
    exit 2
  fi
  if [[ -s "$run_root/scalar_control/train.exit" ]] && [[ "$(cat "$run_root/scalar_control/train.exit")" != 0 ]]; then
    exit 3
  fi
  sleep 30
done
test -s "$run_root/sc_psetpo/train_summary.json"
test -s "$run_root/scalar_control/train_summary.json"

evaluate() {
  local gpu=$1
  local benchmark=$2
  local name=$3
  local adapter=$4
  local annotations=$5
  local audio_dir=$6
  local root="$output_root/$benchmark/$name"
  mkdir -p "$root"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$project_root/scripts/evaluate_checkpoint.py" \
    --base "$base_model" --adapter "$adapter" \
    --annotations "$annotations" --audio-dir "$audio_dir" \
    --predictions "$root/predictions.jsonl" --summary "$root/inference_summary.json" \
    > "$root/inference.log" 2>&1
  "$python_bin" "$project_root/scripts/score_interval_predictions.py" \
    --predictions "$root/predictions.jsonl" --summary "$root/metrics.json" \
    > "$root/scoring.log" 2>&1
}

for benchmark in spotsound clotho; do
  if [[ "$benchmark" == spotsound ]]; then
    annotations="$spotsound_root/annotations_processed.json"
    audio_dir="$spotsound_root/audio"
  else
    annotations="$clotho_root/annotations_processed.json"
    audio_dir="$clotho_root/audio"
  fi
  evaluate 0 "$benchmark" sc_psetpo "$run_root/sc_psetpo/adapter" "$annotations" "$audio_dir" &
  main_pid=$!
  evaluate 1 "$benchmark" scalar_control "$run_root/scalar_control/adapter" "$annotations" "$audio_dir" &
  control_pid=$!
  wait "$main_pid"
  wait "$control_pid"
done
