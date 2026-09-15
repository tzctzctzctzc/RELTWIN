#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/SpotSound-ICASSP
WORK="$ROOT/spotsound_timeaudio_reltwin_20260916"
WORK_MIN_COMMIT=7317aa56fba5ab9e25e826cea9bc558f80ad7917
EXTERNAL="$ROOT/external/TimeAudio-22db7136"
ENV_DIR="$ROOT/env-timeaudio"
MODEL_DIR="$ROOT/models/timeaudio-public"
HF_HOME="$ROOT/hf-timeaudio"
export HF_HOME

mkdir -p "$ROOT/external" "$MODEL_DIR" "$HF_HOME"

[[ -d "$WORK/.git" ]]
git -C "$WORK" merge-base --is-ancestor "$WORK_MIN_COMMIT" HEAD

if [[ ! -d "$EXTERNAL/.git" ]]; then
  git clone https://github.com/lysanderism/TimeAudio.git "$EXTERNAL"
fi
[[ "$(git -C "$EXTERNAL" rev-parse HEAD)" == "22db7136ff3bad3b6557332a6475998c84f7e3f8" ]]

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  /root/miniconda3/bin/conda create -y -p "$ENV_DIR" python=3.10 pip
fi

"$ENV_DIR/bin/pip" install \
  torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
"$ENV_DIR/bin/pip" install \
  transformers==4.47.0 peft==0.16.0 accelerate==1.6.0 \
  sentencepiece==0.2.0 soundfile librosa omegaconf tensorboardX huggingface_hub

"$ENV_DIR/bin/python" - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path('/root/autodl-tmp/SpotSound-ICASSP/models/timeaudio-public')
snapshot_download('lmsys/vicuna-7b-v1.5', local_dir=root / 'vicuna-7b-v1.5')
snapshot_download('openai/whisper-large-v2', local_dir=root / 'whisper-large-v2')
snapshot_download('google-bert/bert-base-uncased', local_dir=root / 'bert-base-uncased')
hf_hub_download(
    'lysanderism/TimeAudio', 'timeaudio.pth', local_dir=root
)
hf_hub_download(
    'WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2',
    'BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
    local_dir=root,
)
PY

"$ENV_DIR/bin/python" - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('/root/autodl-tmp/SpotSound-ICASSP/models/timeaudio-public')
paths = [path for path in root.rglob('*') if path.is_file() and '/.cache/' not in str(path)]
records = []
for path in sorted(paths):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    records.append({
        'path': str(path.relative_to(root)),
        'bytes': path.stat().st_size,
        'sha256': digest.hexdigest(),
    })
payload = {
    'timeaudio_git_commit': '22db7136ff3bad3b6557332a6475998c84f7e3f8',
    'files': records,
}
destination = root / 'asset_manifest.json'
destination.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
print(destination)
PY

"$ENV_DIR/bin/python" - <<'PY'
import torch, transformers, peft
assert torch.__version__.startswith('2.6.0')
assert transformers.__version__ == '4.47.0'
assert peft.__version__ == '0.16.0'
print({'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__})
PY

echo SETUP_COMPLETE
