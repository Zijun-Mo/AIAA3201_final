#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -eq 0 ]; then
  EXP_ID="phase5_$(date +%Y%m%d_%H%M%S)"
  python3 src/part3/run_diffusion.py \
    --config configs/base.yaml \
    --datasets mandatory \
    --exp-id "$EXP_ID"
else
  python3 src/part3/run_diffusion.py "$@"
fi
