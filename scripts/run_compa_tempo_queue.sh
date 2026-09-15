#!/usr/bin/env bash
set -euo pipefail

project="${PROJECT:-/root/autodl-tmp/SpotSound-ICASSP}"
code="${CODE:-$project/spotsound_reltwin_crossbench_20260914}"
python_bin="${PYTHON_BIN:-$project/env-conda/bin/python}"
base="${BASE:-$project/models/audio-flamingo-3-hf}"
adapter="${ADAPTER:-$project/outputs/reltwin_review_controls_20260912/no_exchange_seed0/adapter}"
out="${OUT:-$project/outputs/compa_tempo_eval_20260915}"

export PYTHONPATH="$code/scripts"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$out/compa_full400" "$out/tempo_full5151"

# The separately launched two-instance smoke test validates the generative
# likelihood path before this queue commits GPU time to the public benchmark.
while screen -ls 2>/dev/null | grep -q '[.]compa_smoke_wait'; do
  sleep 10
done
test -s "$out/compa_smoke2/summary.json"

"$python_bin" "$code/scripts/evaluate_compa_order.py" \
  --base "$base" \
  --adapter "$adapter" \
  --manifest "$project/datasets/CompA-Order/manifest.json" \
  --audio-dir "$project/datasets/CompA-Order/audio" \
  --predictions "$out/compa_full400/predictions.jsonl" \
  --summary "$out/compa_full400/summary.json" \
  > "$out/compa_full400/run.log" 2>&1

# Pre-declared 1,000-example gate. The same resumable file is continued only
# when the public TEMPO metrics show a competitive signal.
"$python_bin" "$code/scripts/evaluate_checkpoint.py" \
  --base "$base" \
  --adapter "$adapter" \
  --annotations "$project/datasets/TEMPO/annotations_processed.json" \
  --audio-dir "$project/datasets/TEMPO/audio" \
  --predictions "$out/tempo_full5151/predictions.jsonl" \
  --summary "$out/tempo_full5151/inference_gate1000.json" \
  --prompt-mode aegbench \
  --max-new-tokens 128 \
  --limit 1000 \
  > "$out/tempo_full5151/inference_gate1000.log" 2>&1

"$python_bin" "$code/scripts/score_tempo_grounding.py" \
  --predictions "$out/tempo_full5151/predictions.jsonl" \
  --summary "$out/tempo_full5151/metrics_gate1000.json" \
  > "$out/tempo_full5151/scoring_gate1000.log" 2>&1

if "$python_bin" - "$out/tempo_full5151/metrics_gate1000.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
# TEMPO reports 49.4 mIoU / 46.5 F1. Continue when the 1k gate is within
# plausible full-set reach on either primary metric.
raise SystemExit(
    0
    if metrics["symmetric_mIoU"] >= 47.0 or metrics["F1_IoU_0.5"] >= 44.0
    else 1
)
PY
then
  printf 'continue\n' > "$out/tempo_full5151/gate_decision.txt"
  "$python_bin" "$code/scripts/evaluate_checkpoint.py" \
    --base "$base" \
    --adapter "$adapter" \
    --annotations "$project/datasets/TEMPO/annotations_processed.json" \
    --audio-dir "$project/datasets/TEMPO/audio" \
    --predictions "$out/tempo_full5151/predictions.jsonl" \
    --summary "$out/tempo_full5151/inference_full.json" \
    --prompt-mode aegbench \
    --max-new-tokens 128 \
    > "$out/tempo_full5151/inference_full.log" 2>&1
  "$python_bin" "$code/scripts/score_tempo_grounding.py" \
    --predictions "$out/tempo_full5151/predictions.jsonl" \
    --summary "$out/tempo_full5151/metrics_full.json" \
    > "$out/tempo_full5151/scoring_full.log" 2>&1
else
  printf 'stop_after_gate\n' > "$out/tempo_full5151/gate_decision.txt"
fi
