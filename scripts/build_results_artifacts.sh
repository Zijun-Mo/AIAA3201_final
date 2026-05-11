#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  echo "Usage: bash scripts/build_results_artifacts.sh [--metrics-out outputs/metrics/final_results] [--figures-out outputs/figures/final_results] [--phase1-exp-id ID ... --phase6-exp-id ID]"
  exit 0
fi

conda run -n aiaa3201 python3 src/common/build_results_artifacts.py "$@"
