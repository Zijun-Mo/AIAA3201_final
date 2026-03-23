#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: bash scripts/check_phase2.sh --exp-id <exp_id> [--config configs/base.yaml] [--strict-dual-run true|false]"
  exit 1
fi

python3 src/common/check_phase2.py "$@"
