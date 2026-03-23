#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  EXP_ID="phase2_$(date +%Y%m%d_%H%M%S)"
  python3 src/part2/run_sota.py \
    --config configs/base.yaml \
    --datasets mandatory \
    --exp-id "$EXP_ID"
else
  python3 src/part2/run_sota.py "$@"
fi
