#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:?set PROJECT_ROOT}
dataset_root=${DATASET_ROOT:?set DATASET_ROOT}
audio_dir=${AUDIO_DIR:-$dataset_root/wds}
base_model=${BASE_MODEL:?set BASE_MODEL}
official_adapter=${OFFICIAL_ADAPTER:?set OFFICIAL_ADAPTER}
setpo_adapter=${SETPO_ADAPTER:?set SETPO_ADAPTER}
output_root=${OUTPUT_ROOT:?set OUTPUT_ROOT}
python_bin=${PYTHON_BIN:-python}
limit=${LIMIT:-100}

export PYTHONPATH="$project_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$output_root/setpo" "$output_root/official"

run_one() {
  local gpu=$1
  local name=$2
  local adapter=$3
  local output_dir="$output_root/$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$project_root/scripts/evaluate_checkpoint.py" \
    --base "$base_model" \
    --adapter "$adapter" \
    --annotations "$dataset_root/annotations_processed.json" \
    --audio-dir "$audio_dir" \
    --predictions "$output_dir/predictions.jsonl" \
    --summary "$output_dir/inference_summary.json" \
    --prompt-mode spotsound \
    --limit "$limit" \
    --max-new-tokens 128 \
    > "$output_dir/inference.log" 2>&1
  "$python_bin" "$project_root/scripts/score_interval_predictions.py" \
    --predictions "$output_dir/predictions.jsonl" \
    --summary "$output_dir/metrics.json" \
    > "$output_dir/scoring.log" 2>&1
}

run_one 0 setpo "$setpo_adapter" &
setpo_pid=$!
run_one 1 official "$official_adapter" &
official_pid=$!
set +e
wait "$setpo_pid"; setpo_exit=$?
wait "$official_pid"; official_exit=$?
set -e
printf '%s\n' "$setpo_exit" > "$output_root/setpo.exit"
printf '%s\n' "$official_exit" > "$output_root/official.exit"
if (( setpo_exit != 0 || official_exit != 0 )); then
  exit 1
fi
