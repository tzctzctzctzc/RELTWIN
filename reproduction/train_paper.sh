#!/usr/bin/env bash
# Train the full two-stage RelTwin configuration for either paper backbone.
set -euo pipefail
backbone=${1:?Usage: bash reproduction/train_paper.sh spotsound|timeaudio ASSETS NEW_OUTPUT}
assets=${2:?Missing asset root}
output=${3:?Missing new output directory}
reltwin_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$reltwin_repo/reproduction/train_full.sh" "$backbone" "$assets" "$output" "${@:4}"
