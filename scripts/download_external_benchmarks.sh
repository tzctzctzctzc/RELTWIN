#!/usr/bin/env bash
set -euo pipefail

root=${1:?usage: download_external_benchmarks.sh DATASET_ROOT}
benchmark=${2:-all}
mirror=${HF_MIRROR:-https://hf-mirror.com}
workers=${DOWNLOAD_WORKERS:-8}
python_bin=${PYTHON_BIN:-python}
clotho_revision=${CLOTHO_REVISION:-f22447945c9cad8239aa6876f7ec61e621681b22}
aeg_revision=${AEG_REVISION:-49a1d919b6df6717c4a34ef9c01e75aa4b3fc8a5}

clotho_root="$root/Clotho-Moment"
aeg_root="$root/AEGBench"
mkdir -p "$clotho_root/test_shards" "$clotho_root/audio" "$aeg_root/data"

download() {
  local url=$1
  local output=$2
  wget -c -q --show-progress \
    --timeout=60 --tries=20 \
    --retry-on-http-error=429,500,502,503,504 \
    -O "$output" "$url"
}
export -f download

if [[ "$benchmark" == all || "$benchmark" == clotho ]]; then
  export mirror clotho_root clotho_revision
  seq -f '%03g' 0 142 | xargs -P "$workers" -I{} bash -c '
    download \
      "$mirror/datasets/lighthouse-emnlp2024/Clotho-Moment/resolve/$clotho_revision/test/test-{}.tar" \
      "$clotho_root/test_shards/test-{}.tar"
  '
  find "$clotho_root/test_shards" -maxdepth 1 -name 'test-*.tar' -print0 \
    | sort -z \
    | xargs -0 -P 4 -I{} tar -xf '{}' -C "$clotho_root/audio"
  printf 'clotho_wav=%s\n' "$(find "$clotho_root/audio" -maxdepth 1 -name '*.wav' | wc -l)"
fi

if [[ "$benchmark" == all || "$benchmark" == aeg ]]; then
  export mirror aeg_root aeg_revision
  seq -f '%05g' 0 3 | xargs -P 4 -I{} bash -c '
    download \
      "$mirror/datasets/zihan-audio/AEGBench/resolve/$aeg_revision/data/train-{}-of-00004.parquet" \
      "$aeg_root/data/train-{}-of-00004.parquet"
  '
  "$python_bin" "$aeg_root/eval/materialize_audio.py" --repo "$aeg_root" --out "$aeg_root"
  printf 'aeg_audio=%s\n' "$(find "$aeg_root/audio" -type f | wc -l)"
fi
