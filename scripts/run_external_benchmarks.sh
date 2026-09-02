#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:?set PROJECT_ROOT}
dataset_root=${DATASET_ROOT:?set DATASET_ROOT}
base_model=${BASE_MODEL:?set BASE_MODEL}
official_adapter=${OFFICIAL_ADAPTER:?set OFFICIAL_ADAPTER}
setpo_adapter=${SETPO_ADAPTER:?set SETPO_ADAPTER}
output_root=${OUTPUT_ROOT:?set OUTPUT_ROOT}
python_bin=${PYTHON_BIN:-python}

export PYTHONPATH="$project_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$output_root/clotho_moment" "$output_root/aegbench"

wait_for_gpu() {
  local gpu=$1
  while true; do
    local first second
    first=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' ')
    sleep 10
    second=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' ')
    local first_mem=${first%,*} first_util=${first#*,}
    local second_mem=${second%,*} second_util=${second#*,}
    if (( first_mem < 1000 && first_util < 5 && second_mem < 1000 && second_util < 5 )); then
      return
    fi
    printf 'waiting_for_gpu=%s first=%s second=%s\n' "$gpu" "$first" "$second"
    sleep 50
  done
}

run_checkpoint() {
  local gpu=$1
  local benchmark=$2
  local adapter_name=$3
  local adapter_path=$4
  local manifest=$5
  local audio_dir=$6
  local prompt_mode=$7
  local max_new_tokens=$8
  local output_dir="$output_root/$benchmark/$adapter_name"
  mkdir -p "$output_dir"
  wait_for_gpu "$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$project_root/scripts/evaluate_checkpoint.py" \
    --base "$base_model" \
    --adapter "$adapter_path" \
    --annotations "$manifest" \
    --audio-dir "$audio_dir" \
    --predictions "$output_dir/predictions.jsonl" \
    --summary "$output_dir/inference_summary.json" \
    --prompt-mode "$prompt_mode" \
    --max-new-tokens "$max_new_tokens" \
    > "$output_dir/inference.log" 2>&1
  "$python_bin" "$project_root/scripts/score_interval_predictions.py" \
    --predictions "$output_dir/predictions.jsonl" \
    --summary "$output_dir/metrics.json" \
    > "$output_dir/scoring.log" 2>&1
}

run_clotho() {
  "$project_root/scripts/download_external_benchmarks.sh" "$dataset_root" clotho
  local root="$dataset_root/Clotho-Moment"
  "$python_bin" "$project_root/scripts/prepare_public_benchmarks.py" \
    --benchmark clotho-moment \
    --source "$root/clotho_moment_test_release.jsonl" \
    --output "$root/annotations_processed.json"
  run_checkpoint 0 clotho_moment setpo "$setpo_adapter" \
    "$root/annotations_processed.json" "$root/audio" spotsound 128
  run_checkpoint 0 clotho_moment official "$official_adapter" \
    "$root/annotations_processed.json" "$root/audio" spotsound 128
}

run_aegbench() {
  "$project_root/scripts/download_external_benchmarks.sh" "$dataset_root" aeg
  local root="$dataset_root/AEGBench"
  "$python_bin" "$project_root/scripts/prepare_public_benchmarks.py" \
    --benchmark aegbench \
    --source "$root/manifest.json" \
    --output "$root/annotations_processed.json"
  run_checkpoint 1 aegbench setpo "$setpo_adapter" \
    "$root/annotations_processed.json" "$root" aegbench 256
  run_checkpoint 1 aegbench official "$official_adapter" \
    "$root/annotations_processed.json" "$root" aegbench 256
}

run_clotho > "$output_root/clotho_moment/pipeline.log" 2>&1 &
clotho_pid=$!
run_aegbench > "$output_root/aegbench/pipeline.log" 2>&1 &
aeg_pid=$!
set +e
wait "$clotho_pid"; clotho_exit=$?
wait "$aeg_pid"; aeg_exit=$?
set -e

printf '%s\n' "$clotho_exit" > "$output_root/clotho_moment/pipeline.exit"
printf '%s\n' "$aeg_exit" > "$output_root/aegbench/pipeline.exit"
if (( clotho_exit != 0 || aeg_exit != 0 )); then
  exit 1
fi
