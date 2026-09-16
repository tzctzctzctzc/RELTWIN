#!/usr/bin/env bash
set -euo pipefail

project=/root/autodl-tmp/SpotSound-ICASSP
repo="$project/spotsound_timeaudio_reltwin_20260916"
python_bin=/root/miniconda3/bin/python
output="$project/outputs/spotsound_full_reltwin_20260916"
base="$project/models/audio-flamingo-3-hf"
official="$project/models/SpotSound"

mkdir -p "$output"
export PYTHONPATH="$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo"

train_args=(
  --base "$base"
  --adapter "$official"
  --twin-source event_twin "$project/autoresearch/06_experiments/data/reltwin_esc50_v1/rehearsal.json" "$project/autoresearch/06_experiments/data/reltwin_esc50_v1/audio" 3
  --twin-source relation_twin "$project/autoresearch/06_experiments/data/reltwin_esc50_v1/train.json" "$project/autoresearch/06_experiments/data/reltwin_esc50_v1/audio" 1
  --replay-source clotho_train "$project/datasets/Clotho-Moment/train_subset/materialized_full/train.json" "$project/datasets/Clotho-Moment/train_subset/materialized_full" 4
  --replay-source scale_cardinality "$project/autoresearch/06_experiments/data/sc_rehearsal_v1/train.json" "$project/autoresearch/06_experiments/data/sc_rehearsal_v1" 3
  --replay-source tempo "$project/datasets/TEMPO_trainval/materialized/train/annotations.json" "$project/datasets/TEMPO_trainval/materialized/train/audio" 2
  --replay-source audiogrounding "$project/autoresearch/06_experiments/data/spantool_v1/audiogrounding_train.json" "$project/datasets/AudioGrounding-v2" 1
  --eval-manifest "$project/datasets/SpotSound-Bench/annotations_processed.json"
  --eval-manifest "$project/datasets/UnAV-100/annotations_processed.json"
  --eval-manifest "$project/datasets/Clotho-Moment/annotations_processed.json"
  --stage1-adapter "$output/stage1_adapter"
  --stage1-delta "$output/stage1_bridge.pt"
  --output-adapter "$output/final_adapter"
  --output-delta "$output/final_bridge.pt"
  --summary "$output/train_summary.json"
  --rbee-steps 384
  --setpo-steps 128
  --lora-learning-rate 2e-6
  --bridge-learning-rate 5e-6
  --warmup-steps 32
  --min-lr-ratio 0.1
  --prediction-temperature 1.0
  --target-temperature 0.15
  --method-weight 1.0
  --exchange-weight 1.0
  --rbee-replay-weight 0.75
  --setpo-replay-weight 0.75
  --relation-reference-kl-weight 0.2
  --replay-reference-kl-weight 0.2
  --reference-temperature 1.0
  --jitter-ratio 0.15
  --pareto-weight 0.5
  --cache-audios 12
  --seed 0
)

"$python_bin" scripts/train_spotsound_reltwin_full.py "${train_args[@]}" --audit-only \
  > "$output/data_audit.json"
"$python_bin" scripts/train_spotsound_reltwin_full.py "${train_args[@]}" \
  2>&1 | tee "$output/train.log"

evaluate() {
  local name=$1
  local annotations=$2
  local audio_dir=$3
  local limit=$4
  mkdir -p "$output/$name"
  "$python_bin" scripts/evaluate_checkpoint.py \
    --base "$base" \
    --adapter "$output/final_adapter" \
    --delta "$output/final_bridge.pt" \
    --annotations "$annotations" \
    --audio-dir "$audio_dir" \
    --predictions "$output/$name/predictions.jsonl" \
    --summary "$output/$name/summary.json" \
    --limit "$limit" \
    --max-new-tokens 128 \
    2>&1 | tee "$output/$name/evaluate.log"
}

evaluate spotsound400 \
  "$project/datasets/SpotSound-Bench/annotations_processed.json" \
  "$project/datasets/SpotSound-Bench/audio" 0
evaluate unav100 \
  "$project/datasets/UnAV-100/annotations_processed.json" \
  "$project/datasets/UnAV-100/audio" 0
evaluate clotho \
  "$project/datasets/Clotho-Moment/annotations_processed.json" \
  "$project/datasets/Clotho-Moment/audio" 1000

"$python_bin" - "$output" "$project" <<'PY'
import json
import sys
from pathlib import Path

output, project = map(Path, sys.argv[1:])
specs = {
    "spotsound400": project / "outputs/reltwin_review_controls_20260912/no_exchange_seed0/public_summary.json",
    "unav100": project / "outputs/reltwin_crossbench_direct_20260915/unav100/summary.json",
    "clotho": project / "outputs/reltwin_crossbench_direct_20260915/clotho_gate1000/summary.json",
}
report = {}
for name, baseline_path in specs.items():
    current = json.loads((output / name / "summary.json").read_text())
    baseline = json.loads(baseline_path.read_text())
    report[name] = {
        "current": {key: current[key] for key in ("completed_count", "mIoU", "R1@0.3", "R1@0.5")},
        "previous_reltwin": {key: baseline[key] for key in ("completed_count", "mIoU", "R1@0.3", "R1@0.5")},
        "delta": {key: current[key] - baseline[key] for key in ("mIoU", "R1@0.3", "R1@0.5")},
    }
gate = report["clotho"]["delta"]
report["clotho_full_gate"] = {
    "passed": gate["mIoU"] > 0 and gate["R1@0.3"] >= -0.5 and gate["R1@0.5"] >= -0.5,
    "rule_fixed_before_training": "mIoU > previous RelTwin and both recalls no worse than 0.5 point",
}
(output / "gate_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not report["clotho_full_gate"]["passed"]:
    raise SystemExit(4)
PY

# The evaluator is resumable: the second call retains the fixed first 1,000 rows
# and computes only the remaining Clotho-Moment examples.
evaluate clotho \
  "$project/datasets/Clotho-Moment/annotations_processed.json" \
  "$project/datasets/Clotho-Moment/audio" 0

"$python_bin" - "$output" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
paths = {
    "train": output / "train_summary.json",
    "spotsound400": output / "spotsound400/summary.json",
    "unav100": output / "unav100/summary.json",
    "clotho6649": output / "clotho/summary.json",
}
result = {name: json.loads(path.read_text()) for name, path in paths.items()}
(output / "final_results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({name: value.get("mIoU", value.get("final_event")) for name, value in result.items()}, indent=2))
PY
