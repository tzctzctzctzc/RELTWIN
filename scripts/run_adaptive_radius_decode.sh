#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:?set PROJECT_ROOT}
input_dir=${INPUT_DIR:?set INPUT_DIR}
boundary_model=${BOUNDARY_MODEL:?set BOUNDARY_MODEL}
output_dir=${OUTPUT_DIR:?set OUTPUT_DIR}
radius_ratio=${RADIUS_RATIO:?set RADIUS_RATIO}
radius_cap_seconds=${RADIUS_CAP_SECONDS:-0.25}
margin=${MARGIN:-0.02}
shards=${SHARDS:-16}
python_bin=${PYTHON_BIN:-python}

manifest="$input_dir/manifest.json"
logits="$input_dir/logits/boundary_logits.jsonl"
mkdir -p "$output_dir/shards" "$output_dir/logs"

git -C "$project_root" rev-parse HEAD > "$output_dir/runner_commit.txt"
sha256sum "$manifest" "$logits" "$boundary_model" > "$output_dir/input_hashes.sha256"

pids=()
for ((shard = 0; shard < shards; shard++)); do
  "$python_bin" "$project_root/scripts/boundary_utility.py" decode \
    --manifest "$manifest" \
    --logits "$logits" \
    --model "$boundary_model" \
    --radius-ratio "$radius_ratio" \
    --radius-cap-seconds "$radius_cap_seconds" \
    --margin-override "$margin" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --output "$output_dir/shards/$shard.jsonl" \
    > "$output_dir/logs/decode_$shard.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more decode shards failed" >&2
  exit 2
fi

merge_args=()
for ((shard = 0; shard < shards; shard++)); do
  merge_args+=(--inputs "$output_dir/shards/$shard.jsonl")
done
"$python_bin" "$project_root/scripts/boundary_utility.py" merge \
  --manifest "$manifest" \
  "${merge_args[@]}" \
  --output "$output_dir/predictions.jsonl" \
  > "$output_dir/logs/merge.log" 2>&1

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

print(json.dumps(json.load(open(sys.argv[1], encoding="utf-8")), ensure_ascii=False, indent=2))
PY
