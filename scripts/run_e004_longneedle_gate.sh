#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-$(pwd)}
cd "$project_root"
python_bin=${PYTHON_BIN:-python}
base_model=${BASE_MODEL:-models/audio-flamingo-3-hf}
official_adapter=${OFFICIAL_ADAPTER:-models/SpotSound}
sft_adapter=${SFT_ADAPTER:?Set SFT_ADAPTER}
setpo_adapter=${SETPO_ADAPTER:?Set SETPO_ADAPTER}
data=${LONGNEEDLE_DATA:-data/longneedle_esc50_v1}
output=${OUTPUT_DIR:-results/e004_longneedle}
mkdir -p "$output"/{official,sft,setpo}
trap 'printf "%s\n" "$?" > "$output/exit_code.txt"' EXIT

evaluate_candidate() {
  local name=$1
  local adapter=$2
  "$python_bin" scripts/evaluate_checkpoint.py \
    --base "$base_model" --adapter "$adapter" \
    --annotations "$data/test.json" --audio-dir "$data/audio" \
    --predictions "$output/$name/predictions.jsonl" \
    --summary "$output/$name/summary.json"
}

evaluate_candidate official "$official_adapter"
evaluate_candidate sft "$sft_adapter"
evaluate_candidate setpo "$setpo_adapter"

"$python_bin" scripts/extract_nova_features.py \
  --base "$base_model" --verifier-adapter "$official_adapter" \
  --manifest "$data/test.json" --audio-dir "$data/audio" \
  --candidate "official=$output/official/predictions.jsonl" \
  --candidate "sft=$output/sft/predictions.jsonl" \
  --candidate "setpo=$output/setpo/predictions.jsonl" \
  --output "$output/features.jsonl"

"$python_bin" scripts/fit_nova_router.py \
  --features "$output/features.jsonl" --official setpo \
  --min-heldout-capture 0.2 \
  --output-model "$output/router.json" \
  --output-report "$output/gate_report.json"

"$python_bin" - "$output/gate_report.json" <<'PY'
import json
import sys

if not json.load(open(sys.argv[1], encoding="utf-8"))["gate"]["passed"]:
    raise SystemExit(42)
PY
