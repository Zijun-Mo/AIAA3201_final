#!/usr/bin/env bash
set -euo pipefail

python3 src/part1/run_baseline.py \
  --config configs/base.yaml \
  --dataset all
