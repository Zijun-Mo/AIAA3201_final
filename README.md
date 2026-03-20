# AIAA3201 Project 3: Video Object Removal & Inpainting

This repository contains the implementation and experiments for removing dynamic objects in video and restoring clean background using temporal information.

## Project Structure

```text
.
├── AGENT.md
├── PLAN.md
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── gt/
├── docs/
├── outputs/
│   ├── masks/
│   ├── videos/
│   ├── figures/
│   ├── metrics/
│   └── logs/
├── scripts/
├── src/
│   ├── common/
│   ├── part1/
│   ├── part2/
│   └── part3/
└── notebooks/
```

## Quick Start

1. Put datasets under `data/raw/` and optional GT under `data/gt/`.
2. Edit `configs/base.yaml` if needed.
3. Run baseline and SOTA pipelines:
   - `bash scripts/run_part1.sh`
   - `bash scripts/run_part2.sh`
   - `bash scripts/run_part3.sh`
4. Run evaluation:
   - `bash scripts/evaluate.sh`

## Mandatory Datasets

- Wild video
- bmx-trees
- tennis

## Core Metrics

- Mask quality: JM (IoU mean), JR (IoU recall)
- Video quality (with GT): PSNR, SSIM

## Notes

- Follow repository conventions in `AGENT.md`.
- Follow execution schedule and milestones in `PLAN.md`.
