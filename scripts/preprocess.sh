#!/usr/bin/env bash
set -euo pipefail

python3 src/common/preprocess_videos.py \
  --config configs/base.yaml \
  --datasets mandatory \
  "$@"
