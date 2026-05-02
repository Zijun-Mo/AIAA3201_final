# AIAA3201 Project 3: Video Object Removal & Inpainting

This repository contains the implementation and experiments for removing dynamic objects in video and restoring clean background using temporal information.

## Project Structure

```text
.
├── AGENTS.md
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

## Acceptance Runbook (Phase 0-5, Safe Defaults)

This runbook is the default reproducible path for fresh clones.  
Goal: pass Phase 0-5 acceptance checks with lowest risk.

### 0) Prepare Environment

Use a single staged conda environment:

```bash
conda env create -f environment.yml
conda activate aiaa3201
python -V
```

Expected: `Python 3.10.x`

Install Stage A (required):

```bash
pip install -r requirements.txt
python -c "import numpy, cv2, yaml, skimage, matplotlib; print('core ok')"
```

Install Stage B for Phase 2/3:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "einops>=0.7,<1" "omegaconf>=2.3,<3" "hydra-core>=1.3,<2" "imageio>=2.34,<3" "imageio-ffmpeg>=0.5,<1"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Optional Stage C (Route G only):

```bash
pip install "diffusers>=0.30,<1" "transformers>=4.44,<5" "accelerate>=0.33,<1" "xformers>=0.0.27"
```

### 1) Define Experiment IDs Once

```bash
export P0_EXP="phase0_$(date +%Y%m%d_%H%M%S)"
export P1_EXP="phase1_$(date +%Y%m%d_%H%M%S)"
export P2_EXP="phase2_$(date +%Y%m%d_%H%M%S)"
export P3_EXP="phase3_$(date +%Y%m%d_%H%M%S)"
export P4_EXP="phase4_$(date +%Y%m%d_%H%M%S)"
export P5_EXP="phase5_$(date +%Y%m%d_%H%M%S)"
```

### 2) Preflight Directories and Mandatory Data

```bash
mkdir -p data/{raw,processed,gt} outputs/{videos,masks,figures,metrics,logs}
ls data/raw/wild.mp4 data/raw/bmx-trees.mp4 data/raw/tennis.mp4
```

### 3) Phase 0 (Standardize + Unified Evaluation + Gate)

```bash
bash scripts/preprocess.sh --datasets mandatory --overwrite

bash scripts/evaluate.sh \
  --config configs/base.yaml \
  --exp-id "$P0_EXP" \
  --datasets mandatory \
  --pred-root data/processed \
  --gt-root data/gt \
  --allow-missing-gt true

bash scripts/check_phase0.sh --exp-id "$P0_EXP" --config configs/base.yaml
```

### 4) Phase 1 (A1-A5 + Gate)

```bash
python3 src/part1/run_baseline.py \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id "$P1_EXP" \
  --seed 42 \
  --wild-fallback-mask true

bash scripts/check_phase1.sh --exp-id "$P1_EXP" --config configs/base.yaml
```

### 5) Phase 2 (B1-B5 + Gate, Non-Strict Default)

`--phase1-exp-id "$P1_EXP"` is required in this safe path to ensure `phase2_a_vs_b.csv` is generated.
Phase 2 mask propagation uses one global backend selected with mask-first scoring and the default `bidirectional_no_wrap` policy, so SAM2/Track Anything propagate from prompt anchors forward and backward without connecting the last frame to the first frame.

```bash
python3 src/part2/run_sota.py \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id "$P2_EXP" \
  --phase1-exp-id "$P1_EXP" \
  --seed 42 \
  --strict-dual-run false

bash scripts/check_phase2.sh \
  --exp-id "$P2_EXP" \
  --config configs/base.yaml \
  --strict-dual-run false
```

### 6) Phase 3 (E1,E2,E4 + Gate, Non-Strict Default)

```bash
python3 src/part3/run_explore.py \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id "$P3_EXP" \
  --phase2-exp-id "$P2_EXP" \
  --stages E1,E2,E4 \
  --seed 42 \
  --strict-sam3-permission true

bash scripts/check_phase3.sh \
  --exp-id "$P3_EXP" \
  --config configs/base.yaml \
  --strict-sam3-permission true
```

### 7) Phase 4 (F1-F5 + Gate, VGGT4D Prior)

Phase 4 Route F exports **VGGT4D prior** as the final video/metric result. `prior` means the VGGT4D raw mask converted to prompt anchors and passed through the same global mask backend and `bidirectional_no_wrap` propagation policy used by B-best. The propagation policy treats videos as non-cyclic, so no backend may connect the last frame back to the first frame.

F-stage semantics:
- `F1`: VGGT4D prior → B-best mask backend + bidirectional no-wrap → ProPainter (final Route F output).
- `F2`: unchanged B-best baseline (reference).
- `F3/F4`: YOLO, VGGT4D raw/prior, and prior-fusion ablations for honest comparison only.
- `F-best`: forced to a pure VGGT4D-prior candidate.

Preflight:

```bash
conda run -n vggt4d python -V
conda run -n vggt4d python data/external/vggt4d/run_vggt4d_chunked.py --help
ls -lh data/external/vggt4d/ckpts/model_tracker_fixed_e20.pt
```

```bash
conda run -n aiaa3201 python src/part3/run_explore.py \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id "$P4_EXP" \
  --phase2-exp-id "$P2_EXP" \
  --stages F1,F2,F3,F4,F5 \
  --phase phase4 \
  --seed 42

bash scripts/check_phase4.sh --exp-id "$P4_EXP" --config configs/base.yaml
```

### 8) Phase 5 (Route G + Gate, Diffusion Inpainting)

Phase 5 implementation is under `src/part3/` (migrated from legacy `src/part4/` path).
By default, only the final selected result (`outputs/videos/<exp_id>/...`) is kept.  
Intermediate variant videos (`<exp_id>__G-*`) are cleaned after scoring.

Phase 5 is based on Phase 2 (`B-best`) only:
- `--phase2-exp-id "$P2_EXP"` provides both the fixed mask/base input source and the `B-best` metric reference.

```bash
python3 src/part3/run_diffusion.py \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id "$P5_EXP" \
  --phase2-exp-id "$P2_EXP" \
  --seed 42 \
  --device cuda

bash scripts/check_phase5.sh --exp-id "$P5_EXP" --config configs/base.yaml
```

If you need to keep all variant videos for debugging/visual comparison, add:

```bash
--keep-variant-videos
```

## CLI Contract for Acceptance

- Phase 2 safe path requires `--phase1-exp-id`.
- Phase 3 safe path requires `--phase2-exp-id`.
- Phase 5 safe path requires `--phase2-exp-id`.
- Safe-path gate commands use:
  - `check_phase2 --strict-dual-run false`
  - `check_phase3 --strict-sam3-permission false`

## Output Contract (Default `video_only=true`)

Stable output locations:

- Restored prediction video: `outputs/videos/<exp_id>/<dataset>/restored_h264.mp4`
- Predicted mask video: `outputs/videos/<exp_id>/<dataset>/mask_h264.mp4`
- Metrics root: `outputs/metrics/<exp_id>/`

Important behavior:

- Intermediate `_candidates/` may be auto-cleaned.
- `frames/` and `masks/` directories may be auto-cleaned when video-only output is enabled.

Phase 4 naming (migrated):
- Prior naming is now `vggt4d` / `vggt4d_yolo` (replacing `vggt` / `vggt_yolo`).
- `phase4_mask_priors.csv` and Phase 4 logs/reports use the new naming.

## Optional: Strict Mode

Use strict mode only when environment and permissions are fully ready.

Phase 2 strict:

```bash
python3 src/part2/run_sota.py \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id "$P2_EXP" \
  --phase1-exp-id "$P1_EXP" \
  --seed 42 \
  --strict-dual-run true

bash scripts/check_phase2.sh \
  --exp-id "$P2_EXP" \
  --config configs/base.yaml \
  --strict-dual-run true
```

Phase 3 strict:

```bash
python3 src/part3/run_explore.py \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id "$P3_EXP" \
  --phase2-exp-id "$P2_EXP" \
  --stages E1,E2,E3,E4 \
  --seed 42 \
  --sam3-env-name sam3 \
  --strict-sam3-permission true

bash scripts/check_phase3.sh \
  --exp-id "$P3_EXP" \
  --config configs/base.yaml \
  --strict-sam3-permission true
```

## Optional: Alias Management (`A-best` / `B-best`)

Aliases are not required for the default acceptance runbook.  
Use this only when you want stable alias folders for analysis or downstream references.

```bash
bash scripts/sync_best_aliases.sh \
  --a-exp-id "$P1_EXP" \
  --b-exp-id "$P2_EXP"
```

Outputs:

- `outputs/videos/A-best`, `outputs/metrics/A-best`, `outputs/figures/A-best`
- `outputs/videos/B-best`, `outputs/metrics/B-best`, `outputs/figures/B-best`
- `outputs/metrics/best_alias_map.json`

## Troubleshooting

- CUDA mismatch:
  - Ensure NVIDIA driver is available: `nvidia-smi`
  - Reinstall torch packages with the exact `cu121` index URL above.
- pip index / network issues:
  - Retry with `pip --default-timeout=120 install ...`
  - Upgrade pip: `python -m pip install --upgrade pip`
- Environment reset:
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
- Removal quality: ROS, TCF, BES

## Notes

- Follow repository conventions in `AGENTS.md`.
- Follow execution schedule and milestones in `PLAN.md`.
