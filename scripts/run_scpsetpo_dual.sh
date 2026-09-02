#!/usr/bin/env bash
set -euo pipefail

project_root=${PROJECT_ROOT:?set PROJECT_ROOT}
python_bin=${PYTHON_BIN:?set PYTHON_BIN}
base_model=${BASE_MODEL:?set BASE_MODEL}
start_adapter=${START_ADAPTER:?set START_ADAPTER}
reltwin_data=${RELTWIN_DATA:?set RELTWIN_DATA}
sc_manifest=${SC_MANIFEST:?set SC_MANIFEST}
output_root=${OUTPUT_ROOT:?set OUTPUT_ROOT}
steps=${STEPS:-128}
seed=${SEED:-1}
mkdir -p "$output_root"
export PYTHONPATH="$project_root/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

extra_args=(--extra-rehearsal-manifest "$sc_manifest")
if [[ -n "${CLOTHO_TRAIN_MANIFEST:-}" && -s "${CLOTHO_TRAIN_MANIFEST}" ]]; then
  extra_args+=(--extra-rehearsal-manifest "$CLOTHO_TRAIN_MANIFEST")
fi

train_arm() {
  local gpu=$1
  local name=$2
  local quality_mode=$3
  local root="$output_root/$name"
  mkdir -p "$root"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$project_root/scripts/train_reltwin_setpo.py" \
    --base "$base_model" \
    --adapter "$start_adapter" \
    --manifest "$reltwin_data/train.json" \
    --rehearsal-manifest "$reltwin_data/rehearsal.json" \
    "${extra_args[@]}" \
    --audio-dir "$reltwin_data/audio" \
    --output-adapter "$root/adapter" \
    --summary "$root/train_summary.json" \
    --steps "$steps" --learning-rate 2e-6 --seed "$seed" \
    --method-weight 0.5 --exchange-weight 1.0 \
    --rehearsal-weight 0.7 --rehearsal-setpo-every 2 \
    --prediction-temperature 1.0 --target-temperature 0.25 \
    --jitter-ratio 0.15 --cache-groups 3 \
    --candidate-mode scale-cardinality \
    --quality-mode "$quality_mode" --pareto-weight 0.5 \
    --rehearsal-sampling balanced \
    --rehearsal-reference-kl-weight 0.1 --long-scale-threshold 0.3 \
    > "$root/train.log" 2>&1
  printf '%s\n' "$?" > "$root/train.exit"
}

train_arm 0 sc_psetpo pareto &
pid_main=$!
train_arm 1 scalar_control scalar &
pid_control=$!
set +e
wait "$pid_main"; main_exit=$?
wait "$pid_control"; control_exit=$?
set -e
printf '%s\n' "$main_exit" > "$output_root/sc_psetpo/train.exit"
printf '%s\n' "$control_exit" > "$output_root/scalar_control/train.exit"
if (( main_exit != 0 || control_exit != 0 )); then
  exit 1
fi
