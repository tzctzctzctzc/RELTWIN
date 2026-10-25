#!/usr/bin/env bash
# Full two-stage RelTwin training for both paper backbones.
set -euo pipefail
backbone=${1:?Usage: bash reproduction/train_full.sh timeaudio|spotsound ASSETS NEW_OUTPUT [--audit-only]}
assets=${2:?Missing asset root}
output=${3:?Missing new output directory}
reltwin_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}

case "$backbone" in timeaudio|spotsound) ;; *) echo 'Unknown backbone' >&2; exit 2;; esac
if [[ -e "$output" ]]; then
  echo 'Choose a new output directory for this training run.' >&2
  exit 2
fi
args=(
  --twin-source event_twin "$assets/autoresearch/06_experiments/data/reltwin_esc50_v1/rehearsal.json" "$assets/autoresearch/06_experiments/data/reltwin_esc50_v1/audio" 3
  --twin-source relation_twin "$assets/autoresearch/06_experiments/data/reltwin_esc50_v1/train.json" "$assets/autoresearch/06_experiments/data/reltwin_esc50_v1/audio" 1
  --replay-source clotho_train "$assets/datasets/Clotho-Moment/train_subset/materialized_full/train.json" "$assets/datasets/Clotho-Moment/train_subset/materialized_full" 4
  --replay-source scale_cardinality "$assets/autoresearch/06_experiments/data/sc_rehearsal_v1/train.json" "$assets/autoresearch/06_experiments/data/sc_rehearsal_v1" 3
  --replay-source tempo_train "$assets/datasets/TEMPO_trainval/materialized/train/annotations.json" "$assets/datasets/TEMPO_trainval/materialized/train/audio" 2
  --replay-source audiogrounding_train "$assets/autoresearch/06_experiments/data/spantool_v1/audiogrounding_train.json" "$assets/datasets/AudioGrounding-v2" 1
  --eval-manifest "$assets/datasets/SpotSound-Bench/annotations_processed.json"
  --eval-manifest "$assets/datasets/UnAV-100/annotations_processed.json"
  --eval-manifest "$assets/datasets/Clotho-Moment/annotations_processed.json"
  --summary "$output/train_summary.json"
  --rbee-steps 384 --setpo-steps 128 --seed 0
  --lora-learning-rate 2e-6 --warmup-steps 32 --min-lr-ratio 0.1
  --prediction-temperature 1 --target-temperature 0.15
  --method-weight 1 --exchange-weight 1
  --rbee-replay-weight 0.75 --setpo-replay-weight 0.75
  --replay-reference-kl-weight 0.2 --reference-temperature 1
  --jitter-ratio 0.15 --pareto-weight 0.5 --cache-audios 12
)
if [[ "$backbone" == timeaudio ]]; then
  trainer="$reltwin_repo/scripts/train_timeaudio_reltwin_full.py"
  args+=(
    --timeaudio-repo "$assets/external/TimeAudio-22db7136"
    --config "$reltwin_repo/configs/timeaudio_paper.yaml"
    --llama "$assets/models/timeaudio-public/vicuna-7b-v1.5"
    --whisper "$assets/models/timeaudio-public/whisper-large-v2"
    --beats "$assets/models/timeaudio-public/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    --bert "$assets/models/timeaudio-public/bert-base-uncased"
    --checkpoint "$assets/models/timeaudio-public/timeaudio.pth"
    --temporal-learning-rate 5e-6
    --stage1-delta "$output/stage1_delta.pt" --output-delta "$output/final_delta.pt"
  )
else
  trainer="$reltwin_repo/scripts/train_spotsound_reltwin_full.py"
  args+=(
    --base "$assets/models/audio-flamingo-3-hf" --adapter "$assets/models/SpotSound"
    --bridge-learning-rate 5e-6 --relation-reference-kl-weight 0
    --stage1-adapter "$output/stage1_adapter" --stage1-delta "$output/stage1_bridge.pt"
    --output-adapter "$output/final_adapter" --output-delta "$output/final_bridge.pt"
  )
fi
# Extra arguments support --audit-only and explicit experimental overrides.
exec "$python_bin" "$trainer" "${args[@]}" "${@:4}"
