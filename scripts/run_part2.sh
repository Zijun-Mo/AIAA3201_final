#!/usr/bin/env bash
set -euo pipefail

python3 src/part2/run_sota.py \
  --config configs/base.yaml \
  --dataset all
