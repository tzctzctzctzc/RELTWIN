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

mkdir -p "$output_dir/logs" "$output_dir/logits"

"$python_bin" "$project_root/scripts/prepare_boundary_benchmark.py" \
  --annotations "$annotations" \
  --predictions "$incumbent" \
  --benchmark-name "$benchmark_name" \
  --incumbent-name official \
  --expected-rows "$expected_rows" \
  --output "$output_dir/manifest.json" \
  > "$output_dir/logs/prepare.log" 2>&1

sha256sum \
  "$annotations" \
  "$incumbent" \
  "$base_model/config.json" \
  "$feature_adapter/adapter_config.json" \
  "$spantool_checkpoint/spantool_config.json" \
  "$boundary_model" \
  "$output_dir/manifest.json" \
  > "$output_dir/input_hashes.sha256"

git -C "$project_root" rev-parse HEAD > "$output_dir/runner_commit.txt"

"$python_bin" "$project_root/scripts/run_metric_iou_pilot.py" export \
  --manifest "$output_dir/manifest.json" \
  --audio-dir "$audio_dir" \
  --base "$base_model" \
  --adapter "$feature_adapter" \
  --spantool-checkpoint "$spantool_checkpoint" \
  --output "$output_dir/logits/boundary_logits.jsonl" \
  > "$output_dir/logs/export.log" 2>&1

actual_rows=$(wc -l < "$output_dir/logits/boundary_logits.jsonl")
if [[ "$actual_rows" -ne "$expected_rows" ]]; then
  printf 'incomplete logit export: %s != %s\n' "$actual_rows" "$expected_rows" >&2
  exit 2
fi

"$python_bin" "$project_root/scripts/boundary_utility.py" decode \
  --manifest "$output_dir/manifest.json" \
  --logits "$output_dir/logits/boundary_logits.jsonl" \
  --model "$boundary_model" \
  --output "$output_dir/predictions.jsonl" \
  > "$output_dir/logs/decode.log" 2>&1

"$python_bin" "$project_root/scripts/run_metric_iou_pilot.py" evaluate \
  --predictions "$output_dir/predictions.jsonl" \
  --report "$output_dir/report.json" \
  --failures "$output_dir/failures.jsonl" \
  --bootstrap-samples 10000 \
  --seed 20260903 \
  > "$output_dir/logs/evaluate.log" 2>&1

"$python_bin" - "$output_dir/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps(report, ensure_ascii=False, indent=2))
PY
