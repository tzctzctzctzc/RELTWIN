#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:?set PROJECT_ROOT}
annotations=${ANNOTATIONS:?set ANNOTATIONS}
audio_dir=${AUDIO_DIR:?set AUDIO_DIR}
incumbent=${INCUMBENT_PREDICTIONS:?set INCUMBENT_PREDICTIONS}
benchmark_name=${BENCHMARK_NAME:?set BENCHMARK_NAME}
expected_rows=${EXPECTED_ROWS:?set EXPECTED_ROWS}
base_model=${BASE_MODEL:?set BASE_MODEL}
feature_adapter=${FEATURE_ADAPTER:?set FEATURE_ADAPTER}
spantool_checkpoint=${SPANTOOL_CHECKPOINT:?set SPANTOOL_CHECKPOINT}
boundary_model=${BOUNDARY_MODEL:?set BOUNDARY_MODEL}
output_dir=${OUTPUT_DIR:?set OUTPUT_DIR}
python_bin=${PYTHON_BIN:-python}
radius_ratio=${RADIUS_RATIO:-0.625}
radius_cap_seconds=${RADIUS_CAP_SECONDS:-0.25}
margin=${MARGIN:-0.02}
shards=${SHARDS:-8}
boundary_context_seconds=${BOUNDARY_CONTEXT_SECONDS:-2.0}

mkdir -p "$output_dir/evidence/logits" "$output_dir/evidence/logs" "$output_dir/local"

"$python_bin" "$project_root/scripts/prepare_long_boundary_benchmark.py" \
  --annotations "$annotations" \
  --predictions "$incumbent" \
  --benchmark-name "$benchmark_name" \
  --expected-rows "$expected_rows" \
  --boundary-context-seconds "$boundary_context_seconds" \
  --output "$output_dir/evidence/manifest.json" \
  > "$output_dir/evidence/logs/prepare.log" 2>&1

sha256sum \
  "$annotations" \
  "$incumbent" \
  "$base_model/config.json" \
  "$feature_adapter/adapter_config.json" \
  "$spantool_checkpoint/spantool_config.json" \
  "$boundary_model" \
  "$output_dir/evidence/manifest.json" \
  > "$output_dir/input_hashes.sha256"
git -C "$project_root" rev-parse HEAD > "$output_dir/runner_commit.txt"

"$python_bin" "$project_root/scripts/run_metric_iou_pilot.py" export \
  --manifest "$output_dir/evidence/manifest.json" \
  --audio-dir "$audio_dir" \
  --base "$base_model" \
  --adapter "$feature_adapter" \
  --spantool-checkpoint "$spantool_checkpoint" \
  --output "$output_dir/evidence/logits/boundary_logits.jsonl" \
  > "$output_dir/evidence/logs/export.log" 2>&1

actual_rows=$(wc -l < "$output_dir/evidence/logits/boundary_logits.jsonl")
if [[ "$actual_rows" -ne "$expected_rows" ]]; then
  printf 'incomplete logit export: %s != %s\n' "$actual_rows" "$expected_rows" >&2
  exit 2
fi

PROJECT_ROOT="$project_root" \
INPUT_DIR="$output_dir/evidence" \
BOUNDARY_MODEL="$boundary_model" \
OUTPUT_DIR="$output_dir/local" \
RADIUS_RATIO="$radius_ratio" \
RADIUS_CAP_SECONDS="$radius_cap_seconds" \
MARGIN="$margin" \
SHARDS="$shards" \
PYTHON_BIN="$python_bin" \
bash "$project_root/scripts/run_adaptive_radius_decode.sh" \
  > "$output_dir/local/pipeline.log" 2>&1

"$python_bin" "$project_root/scripts/restore_long_boundary_predictions.py" \
  --predictions "$output_dir/local/predictions.jsonl" \
  --output "$output_dir/predictions.jsonl" \
  > "$output_dir/restore.log" 2>&1

"$python_bin" "$project_root/scripts/run_metric_iou_pilot.py" evaluate \
  --predictions "$output_dir/predictions.jsonl" \
  --report "$output_dir/report.json" \
  --failures "$output_dir/failures.jsonl" \
  --bootstrap-samples 10000 \
  --seed 20260903 \
  > "$output_dir/evaluate.log" 2>&1

"$python_bin" - "$output_dir/report.json" <<'PY'
import json
import sys

print(json.dumps(json.load(open(sys.argv[1], encoding="utf-8")), ensure_ascii=False, indent=2))
PY
