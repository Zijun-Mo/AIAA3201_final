#!/bin/bash
# Run all figure generation scripts from this directory
set -e
cd "$(dirname "$0")"
python gen_pipeline_figure.py
python gen_metrics_figure.py
python gen_ablation_figure.py
echo "All figures saved to ../tex_workspace/figures/"
