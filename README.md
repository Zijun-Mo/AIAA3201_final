# AIAA3201 Project 3: Video Object Removal & Inpainting

This repository contains the implementation and experiments for removing dynamic objects in video and restoring clean background using temporal information.

## Project Structure

```text
.
├── AGENT.md
├── PLAN.md
├── environment.yml
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

## Phase 0 Workflow (Engineering-Ready)

Standardize data and run unified evaluation before model experiments:

```bash
# 1) Normalize mandatory datasets into standardized frame folders
bash scripts/preprocess.sh --datasets mandatory --overwrite

# 2) Unified evaluation entry (works even when GT is missing)
bash scripts/evaluate.sh --config configs/base.yaml --exp-id <exp_id> --datasets mandatory --pred-root outputs/videos --gt-root data/gt --allow-missing-gt true

# 3) Day-1 gate check for Phase 0 completeness
bash scripts/check_phase0.sh --exp-id <exp_id> --config configs/base.yaml
```

Input contract:
- Raw videos: `data/raw/wild.mp4`, `data/raw/bmx-trees.mp4`, `data/raw/tennis.mp4`
- Predictions: `outputs/videos/<dataset>/frames/` and optional `outputs/videos/<dataset>/masks/`
- GT (optional): `data/gt/<dataset>/frames/` and optional `data/gt/<dataset>/masks/`

## Phase 1 Workflow (A1-A5 Full Baseline)

Phase 1 now runs a complete staged baseline:
`YOLOv8-seg / MaskRCNN -> optical-flow dynamic filter -> mask dilation -> temporal borrow -> cv2.inpaint`.

```bash
# Default: run full A1-A5 on mandatory datasets
bash scripts/run_part1.sh

# Explicit run with custom exp id
python3 src/part1/run_baseline.py \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id phase1_manual_20260320 \
  --seed 42 \
  --wild-fallback-mask true

# Optional: override a specific setting (single-value run)
python3 src/part1/run_baseline.py \
  --config configs/base.yaml \
  --datasets wild \
  --exp-id phase1_wild_debug \
  --flow-threshold 1.2 \
  --dilation-kernel 7 \
  --inpaint-method telea \
  --temporal-window 1

# Phase 1 acceptance gate
bash scripts/check_phase1.sh --exp-id <exp_id> --config configs/base.yaml
```

Main outputs:
- A-best predictions: `outputs/videos/<exp_id>/<dataset>/{frames,masks}`
- Candidate predictions: `outputs/videos/<exp_id>/_candidates/A1..A5/...`
- Final A-best metrics: `outputs/metrics/<exp_id>/summary.json` and `per_dataset.csv`
- Phase 1 ablation table: `outputs/metrics/<exp_id>/phase1_ablation.csv`
- Phase 1 selection metadata: `outputs/metrics/<exp_id>/phase1_selection.json`
- Failure case index: `outputs/figures/<exp_id>/failure_cases/failure_cases.csv`
- Failure case explained index: `outputs/figures/<exp_id>/failure_cases/failure_cases_explained.csv`
- Acceptance report: `outputs/metrics/<exp_id>/phase1_acceptance_report.md`

Notes:
- If `ultralytics` is missing and `part1.runtime.auto_install_missing=true`, script auto-installs it and continues.
- If YOLO install/load fails, pipeline logs fallback and continues with available segmentation model(s).
- `--seed` defaults to `project.seed` in config and is recorded in `phase1_run_meta.json`.
- For `wild`, fallback mask generation can be enabled with `--wild-fallback-mask true` (or config fallback flag).

## Phase 2 Workflow (B1-B5 SOTA Mainline)

Phase 2 now runs staged SOTA reproduction:
`SAM2 / TrackAnything (dual run) -> ProPainter`, with auto-install, fallback, and B-best selection.

```bash
# Default: run full B1-B5 on mandatory datasets
bash scripts/run_part2.sh

# Explicit run with custom exp id
python3 src/part2/run_sota.py \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id phase2_manual_20260323 \
  --mask-models sam2,trackanything \
  --prompt-detector yolo \
  --seed 42 \
  --strict-dual-run true

# Quick smoke (single dataset, limited frames)
python3 src/part2/run_sota.py \
  --config configs/base.yaml \
  --datasets wild \
  --stages B1 \
  --max-frames 16 \
  --exp-id phase2_smoke_wild

# Phase 2 acceptance gate
bash scripts/check_phase2.sh --exp-id <exp_id> --config configs/base.yaml --strict-dual-run true
```

Main outputs:
- B-best predictions: `outputs/videos/<exp_id>/<dataset>/{frames,masks}`
- Candidate predictions: `outputs/videos/<exp_id>/_candidates/B1..B5/...`
- Final B-best metrics: `outputs/metrics/<exp_id>/summary.json` and `per_dataset.csv`
- Phase 2 ablation table: `outputs/metrics/<exp_id>/phase2_ablation.csv`
- Phase 2 selection metadata: `outputs/metrics/<exp_id>/phase2_selection.json`
- A-best vs B-best comparison: `outputs/metrics/<exp_id>/phase2_a_vs_b.csv`
- Failure case explained index: `outputs/figures/<exp_id>/failure_cases/failure_cases_explained.csv`
- Acceptance report: `outputs/metrics/<exp_id>/phase2_acceptance_report.md`

Notes:
- External assets are auto-managed under `outputs/external/part2/`.
- Backend policy: `run_sota.py` now tries local official SAM2/Track-Anything inference first, and only falls back when official runtime/checkpoints fail.
- If ProPainter fails (including OOM), pipeline retries degraded profile and can fallback to CV2 inpaint while recording status.
- Backend fallback is non-silent: reason is logged and persisted in candidate metadata (`official_error` / `fallback_reason`).
- `--strict-dual-run true` enforces both `sam2` and `trackanything` in B1 stage.

## Conda Environment Setup

Use a single staged environment named `aiaa3201` with Python 3.10.

### 1) Create and activate environment (base)

```bash
conda env create -f environment.yml
conda activate aiaa3201
python -V
```

Expected: `Python 3.10.x`

### 2) Stage A (required core dependencies)

```bash
pip install -r requirements.txt
python -c "import numpy, cv2, yaml, skimage, matplotlib; print('core ok')"
```

### 3) Stage B (on-demand advanced GPU stack)

Install this stage when running SAM2/TrackAnything/ProPainter pipelines.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "einops>=0.7,<1" "omegaconf>=2.3,<3" "hydra-core>=1.3,<2" "imageio>=2.34,<3" "imageio-ffmpeg>=0.5,<1"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 4) Stage C (optional diffusion stack)

Install this stage only for Route G (diffusion-based inpainting).

```bash
pip install "diffusers>=0.30,<1" "transformers>=4.44,<5" "accelerate>=0.33,<1" "xformers>=0.0.27"
```

## Environment Validation

```bash
conda env list | grep aiaa3201
python -V
bash scripts/run_part1.sh
bash scripts/run_part2.sh
```

## Troubleshooting

- CUDA mismatch:
  - Ensure NVIDIA driver is available (`nvidia-smi`).
  - Reinstall torch packages with cu121 index URL exactly as shown above.
- pip index / network issues:
  - Retry with `pip --default-timeout=120 install ...`.
  - Upgrade pip first: `python -m pip install --upgrade pip`.
- version conflict rollback:
  - Remove env and recreate from scratch:
    - `conda deactivate`
    - `conda env remove -n aiaa3201`
    - `conda env create -f environment.yml`

## Mandatory Datasets

- Wild video
- bmx-trees
- tennis

### How To Download (Official + Reproducible)

#### 1) `bmx-trees` / `tennis` from DAVIS 2017 (480p)

Official download URL:
- `https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip`

```bash
# Download DAVIS package (about 795MB)
mkdir -p data/external/davis
wget -c -O data/external/davis/DAVIS-2017-trainval-480p.zip \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

# Extract only required sequences and annotations
unzip -o data/external/davis/DAVIS-2017-trainval-480p.zip \
  'DAVIS/JPEGImages/480p/bmx-trees/*' \
  'DAVIS/JPEGImages/480p/tennis/*' \
  'DAVIS/Annotations/480p/bmx-trees/*' \
  'DAVIS/Annotations/480p/tennis/*' \
  -d data/external/davis
```

Convert DAVIS frame sequences to this repo's standard layout (`data/raw/*.mp4`, `data/gt/*/frames`, `data/gt/*/masks`):

```bash
conda run -n aiaa3201 python - <<'PY'
from pathlib import Path
import cv2

repo = Path('.').resolve()
davis_root = repo / 'data' / 'external' / 'davis' / 'DAVIS'

for seq in ['bmx-trees', 'tennis']:
    rgb_dir = davis_root / 'JPEGImages' / '480p' / seq
    ann_dir = davis_root / 'Annotations' / '480p' / seq
    rgb_files = sorted(rgb_dir.glob('*.jpg'))
    ann_files = sorted(ann_dir.glob('*.png'))

    first = cv2.imread(str(rgb_files[0]))
    h, w = first.shape[:2]
    out_video = repo / 'data' / 'raw' / f'{seq}.mp4'
    out_video.parent.mkdir(parents=True, exist_ok=True)

    vw = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*'mp4v'), 24.0, (w, h))
    for f in rgb_files:
        vw.write(cv2.imread(str(f)))
    vw.release()

    gt_frames = repo / 'data' / 'gt' / seq / 'frames'
    gt_masks = repo / 'data' / 'gt' / seq / 'masks'
    gt_frames.mkdir(parents=True, exist_ok=True)
    gt_masks.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(rgb_files):
        cv2.imwrite(str(gt_frames / f'frame_{i:06d}.png'), cv2.imread(str(f)))
    for i, f in enumerate(ann_files):
        m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(gt_masks / f'frame_{i:06d}.png'), m)
PY
```

#### 2) `wild` video

`wild.mp4` is not from DAVIS. Capture it yourself (campus/corridor/street) or generate it with a text-to-video model, then place it at:

```bash
data/raw/wild.mp4
```

#### 3) Re-run Phase 0 preprocess after data download

```bash
bash scripts/preprocess.sh --datasets mandatory --overwrite
```

## Core Metrics

- Mask quality: JM (IoU mean), JR (IoU recall)
- Video quality (with GT): PSNR, SSIM

## Notes

- Follow repository conventions in `AGENT.md`.
- Follow execution schedule and milestones in `PLAN.md`.
