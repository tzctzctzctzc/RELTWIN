#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/SpotSound-ICASSP}
WORK=${WORK:-$ROOT/spotsound_timeaudio_reltwin_20260916}
PY=${PY:-$ROOT/env-timeaudio/bin/python}
OUT=${OUT:-$ROOT/outputs/timeaudio_full_reltwin_20260916}
TRAIN="$WORK/scripts/train_timeaudio_reltwin_full.py"
EVAL="$WORK/scripts/evaluate_timeaudio_checkpoint.py"
DELTA="$OUT/seed0/final_delta.pt"
RBEE_STEPS=${RBEE_STEPS:-384}
SETPO_STEPS=${SETPO_STEPS:-128}

mkdir -p "$OUT/seed0"
export PYTHONPATH="$WORK/scripts"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

common_model=(
  --timeaudio-repo "$ROOT/external/TimeAudio-22db7136"
  --config "$ROOT/external/TimeAudio-22db7136/configs/infer_config.yaml"
  --llama "$ROOT/models/timeaudio-public/vicuna-7b-v1.5"
  --whisper "$ROOT/models/timeaudio-public/whisper-large-v2"
  --beats "$ROOT/models/timeaudio-public/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
  --bert "$ROOT/models/timeaudio-public/bert-base-uncased"
  --checkpoint "$ROOT/models/timeaudio-public/timeaudio.pth"
)

train_args=(
  "${common_model[@]}"
  --twin-source event_twin "$ROOT/autoresearch/06_experiments/data/reltwin_esc50_v1/rehearsal.json" "$ROOT/autoresearch/06_experiments/data/reltwin_esc50_v1/audio" 3
  --twin-source relation_twin "$ROOT/autoresearch/06_experiments/data/reltwin_esc50_v1/train.json" "$ROOT/autoresearch/06_experiments/data/reltwin_esc50_v1/audio" 1
  --replay-source clotho_train "$ROOT/datasets/Clotho-Moment/train_subset/materialized_full/train.json" "$ROOT/datasets/Clotho-Moment/train_subset/materialized_full" 4
  --replay-source scale_cardinality "$ROOT/autoresearch/06_experiments/data/sc_rehearsal_v1/train.json" "$ROOT/autoresearch/06_experiments/data/sc_rehearsal_v1" 3
  --replay-source tempo_train "$ROOT/datasets/TEMPO_trainval/materialized/train/annotations.json" "$ROOT/datasets/TEMPO_trainval/materialized/train/audio" 2
  --replay-source audiogrounding_train "$ROOT/autoresearch/06_experiments/data/spantool_v1/audiogrounding_train.json" "$ROOT/datasets/AudioGrounding-v2" 1
  --eval-manifest "$ROOT/datasets/SpotSound-Bench/annotations_processed.json"
  --eval-manifest "$ROOT/datasets/UnAV-100/annotations_processed.json"
  --eval-manifest "$ROOT/datasets/Clotho-Moment/annotations_processed.json"
  --stage1-delta "$OUT/seed0/rbee_delta.pt"
  --output-delta "$DELTA"
  --summary "$OUT/seed0/train_summary.json"
  --rbee-steps "$RBEE_STEPS"
  --setpo-steps "$SETPO_STEPS"
  --lora-learning-rate 2e-6
  --temporal-learning-rate 5e-6
  --warmup-steps 32
  --method-weight 1.0
  --exchange-weight 1.0
  --rbee-replay-weight 0.75
  --setpo-replay-weight 0.75
  --replay-reference-kl-weight 0.2
  --seed 0
)

printf '%s PREFLIGHT_START\n' "$(date -Is)" >> "$OUT/queue.log"
"$PY" -m pytest -q "$WORK/tests" > "$OUT/preflight_pytest.log" 2>&1
"$PY" "$TRAIN" "${train_args[@]}" --audit-only > "$OUT/data_audit.json" 2> "$OUT/data_audit.err"
if [[ "${AUDIT_ONLY:-0}" == "1" ]]; then
  printf '%s AUDIT_ONLY_DONE\n' "$(date -Is)" >> "$OUT/queue.log"
  exit 0
fi
printf '%s TRAIN_START\n' "$(date -Is)" >> "$OUT/queue.log"
"$PY" "$TRAIN" "${train_args[@]}" > "$OUT/seed0/train.log" 2>&1
printf '%s TRAIN_DONE\n' "$(date -Is)" >> "$OUT/queue.log"
if [[ "${TRAIN_ONLY:-0}" == "1" ]]; then
  printf '%s TRAIN_ONLY_DONE\n' "$(date -Is)" >> "$OUT/queue.log"
  exit 0
fi

eval_one() {
  local name="$1" annotations="$2" audio_dir="$3" limit="$4"
  local dir="$OUT/eval/$name"
  mkdir -p "$dir"
  args=(
    "${common_model[@]}"
    --delta "$DELTA"
    --annotations "$annotations"
    --audio-dir "$audio_dir"
    --predictions "$dir/predictions.jsonl"
    --summary "$dir/summary.json"
    --model-label "TimeAudio + Full RelTwin"
    --max-new-tokens 128
  )
  if [[ "$limit" -gt 0 ]]; then
    args+=(--limit "$limit")
  fi
  printf '%s EVAL_START %s limit=%s\n' "$(date -Is)" "$name" "$limit" >> "$OUT/queue.log"
  "$PY" "$EVAL" "${args[@]}" > "$dir/run.log" 2>&1
  printf '%s EVAL_DONE %s limit=%s\n' "$(date -Is)" "$name" "$limit" >> "$OUT/queue.log"
}

eval_one spotsound400 "$ROOT/datasets/SpotSound-Bench/annotations_processed.json" "$ROOT/datasets/SpotSound-Bench/audio" 0
eval_one unav100 "$ROOT/datasets/UnAV-100/annotations_processed.json" "$ROOT/datasets/UnAV-100/audio" 0
eval_one clotho1000 "$ROOT/datasets/Clotho-Moment/annotations_processed.json" "$ROOT/datasets/Clotho-Moment/audio" 1000

"$PY" - "$ROOT" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root, out = map(Path, sys.argv[1:])
load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
official_root = root / "outputs/timeaudio_reltwin_20260916"
official = {
    "SpotSound-Bench": load(official_root / "baseline_official_spot400/summary.json"),
    "UnAV-100": load(official_root / "baseline_official_unav100/summary.json"),
    "Clotho-Moment-1000": load(official_root / "baseline_official_clotho1000/summary.json"),
}
adapted = {
    "SpotSound-Bench": load(out / "eval/spotsound400/summary.json"),
    "UnAV-100": load(out / "eval/unav100/summary.json"),
    "Clotho-Moment-1000": load(out / "eval/clotho1000/summary.json"),
}
metrics = ("mIoU", "R1@0.3", "R1@0.5")
gains = {
    dataset: {metric: adapted[dataset][metric] - official[dataset][metric] for metric in metrics}
    for dataset in official
}
passed = all(gains[dataset]["mIoU"] > 0 for dataset in gains)
payload = {
    "gate": "pass" if passed else "fail",
    "rule": "mIoU must exceed the same-harness official TimeAudio checkpoint on all three benchmarks",
    "official": {dataset: {metric: value[metric] for metric in metrics} for dataset, value in official.items()},
    "adapted": {dataset: {metric: value[metric] for metric in metrics} for dataset, value in adapted.items()},
    "gains": gains,
}
(out / "strict_probe_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
raise SystemExit(0 if passed else 20)
PY

eval_one clotho1000 "$ROOT/datasets/Clotho-Moment/annotations_processed.json" "$ROOT/datasets/Clotho-Moment/audio" 0
printf '%s QUEUE_DONE\n' "$(date -Is)" >> "$OUT/queue.log"
