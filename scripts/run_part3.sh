#!/usr/bin/env bash
set -euo pipefail

python3 src/part3/run_explore.py \
  --config configs/base.yaml \
  --dataset all
