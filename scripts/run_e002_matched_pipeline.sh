#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:-$(pwd)}
cd "$project_root"
python_bin=${PYTHON_BIN:-python}
script_dir=${SCRIPT_DIR:-scripts}
base_model=${BASE_MODEL:-models/audio-flamingo-3-hf}
official_adapter=${SPOTSOUND_ADAPTER:-models/SpotSound}
data=${RELTWIN_DATA:-data/reltwin_esc50_v1}
public_annotations=${SPOTSOUND_ANNOTATIONS:-datasets/SpotSound-Bench/annotations_processed.json}
public_audio=${SPOTSOUND_AUDIO_DIR:-datasets/SpotSound-Bench/audio}
suite=${OUTPUT_DIR:-outputs/e002_matched_pipeline}
seeds=${SEEDS:-"0 1 2"}
mkdir -p "$suite"
trap 'printf "%s\n" "$?" > "$suite/exit_code.txt"' EXIT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

evaluate_reltwin() {
  local adapter=$1
  local root=$2
  "$python_bin" "$script_dir/evaluate_checkpoint.py" \
    --base "$base_model" --adapter "$adapter" \
    --annotations "$data/test.json" --audio-dir "$data/audio" \
    --predictions "$root/reltwin_predictions.jsonl" --summary "$root/reltwin_summary.json"
  "$python_bin" "$script_dir/analyze_reltwin_probe.py" \
    --manifest "$data/test.json" --predictions "$root/reltwin_predictions.jsonl" \
    --output "$root/reltwin_analysis.json"
  "$python_bin" "$script_dir/score_interval_predictions.py" \
    --predictions "$root/reltwin_predictions.jsonl" --summary "$root/reltwin_dual_metrics.json"
}

evaluate_public() {
  local adapter=$1
  local root=$2
  "$python_bin" "$script_dir/evaluate_checkpoint.py" \
    --base "$base_model" --adapter "$adapter" \
    --annotations "$public_annotations" \
    --audio-dir "$public_audio" \
    --predictions "$root/public_predictions.jsonl" --summary "$root/public_summary.json"
  "$python_bin" "$script_dir/score_interval_predictions.py" \
    --predictions "$root/public_predictions.jsonl" --summary "$root/public_dual_metrics.json"
}

for seed in $seeds; do
  seed_root=$suite/seed_${seed}
  rbee_root=$seed_root/rbee
  setpo_root=$seed_root/setpo
  mkdir -p "$rbee_root" "$setpo_root"

  "$python_bin" "$script_dir/train_reltwin_micro.py" \
    --base "$base_model" --adapter "$official_adapter" \
    --manifest "$data/train.json" --rehearsal-manifest "$data/rehearsal.json" \
    --audio-dir "$data/audio" --output-adapter "$rbee_root/adapter" \
    --summary "$rbee_root/train_summary.json" --mode rbee \
    --steps 256 --learning-rate 5e-6 --temperature 1.0 \
    --method-weight 1.0 --rehearsal-weight 0.5 --cache-groups 512 \
    --margin 0.2 --seed "$seed"
  evaluate_reltwin "$rbee_root/adapter" "$rbee_root"

  "$python_bin" "$script_dir/train_reltwin_setpo.py" \
    --base "$base_model" --adapter "$rbee_root/adapter" \
    --manifest "$data/train.json" --rehearsal-manifest "$data/rehearsal.json" \
    --audio-dir "$data/audio" --output-adapter "$setpo_root/adapter" \
    --summary "$setpo_root/train_summary.json" \
    --steps 64 --learning-rate 2e-6 --seed "$seed" \
    --method-weight 0.5 --exchange-weight 1.0 \
    --rehearsal-weight 0.5 --rehearsal-setpo-every 4 \
    --prediction-temperature 1.0 --target-temperature 0.15 \
    --jitter-ratio 0.15 --cache-groups 4
  evaluate_reltwin "$setpo_root/adapter" "$setpo_root"
done

"$python_bin" - "$suite" <<'PY'
import json
import sys
from pathlib import Path

suite = Path(sys.argv[1])
pairs = []
for seed in (0, 1, 2):
    base = suite / f"seed_{seed}"
    pairs.append((json.loads((base / "rbee/reltwin_analysis.json").read_text()), json.loads((base / "setpo/reltwin_analysis.json").read_text())))

per_seed = []
for seed, (rbee, setpo) in enumerate(pairs):
    per_seed.append({
        "seed": seed,
        "rbee_mIoU": rbee["query_mIoU"],
        "setpo_mIoU": setpo["query_mIoU"],
        "rbee_pair_acc_0.5": rbee["pair_acc_0.5"],
        "setpo_pair_acc_0.5": setpo["pair_acc_0.5"],
        "passes": setpo["query_mIoU"] > rbee["query_mIoU"] and setpo["pair_acc_0.5"] > rbee["pair_acc_0.5"],
    })
mean_rbee_miou = sum(x[0]["query_mIoU"] for x in pairs) / 3
mean_setpo_miou = sum(x[1]["query_mIoU"] for x in pairs) / 3
mean_rbee_pair = sum(x[0]["pair_acc_0.5"] for x in pairs) / 3
mean_setpo_pair = sum(x[1]["pair_acc_0.5"] for x in pairs) / 3
gate = all(row["passes"] for row in per_seed) and mean_setpo_miou > mean_rbee_miou and mean_setpo_pair > mean_rbee_pair
report = {"per_seed": per_seed, "means": {"rbee_mIoU": mean_rbee_miou, "setpo_mIoU": mean_setpo_miou, "rbee_pair_acc_0.5": mean_rbee_pair, "setpo_pair_acc_0.5": mean_setpo_pair}, "development_gate_passed": gate}
(suite / "development_gate.json").write_text(json.dumps(report, indent=2) + "\n")
if not gate:
    raise SystemExit(42)
PY

for seed in 0 1 2; do
  seed_root=$suite/seed_${seed}
  evaluate_public "$seed_root/rbee/adapter" "$seed_root/rbee"
  evaluate_public "$seed_root/setpo/adapter" "$seed_root/setpo"
done
