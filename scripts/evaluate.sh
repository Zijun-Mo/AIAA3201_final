#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  EXP_ID="phase0_$(date +%Y%m%d_%H%M%S)"
  python3 src/common/evaluate_experiment.py \
    --config configs/base.yaml \
    --exp-id "$EXP_ID" \
    --datasets mandatory \
    --pred-root data/processed \
    --gt-root data/gt \
    --allow-missing-gt true
else
  python3 src/common/evaluate_experiment.py "$@"
fi
