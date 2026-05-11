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

## Acceptance Runbook (Phase 0-6, Safe Defaults)

This runbook is the default reproducible path for fresh clones.  
Goal: pass Phase 0-5 acceptance checks, run the Phase 6 stacking gate, and rebuild the final tables/figures with lowest risk.

### Phase 0-5 Batch Run

```bash
bash scripts/run_phase0_to5.sh
```

Common overrides:

```bash
# Dry-run (print commands only)
bash scripts/run_phase0_to5.sh --dry-run true

# Debug/permissive run when SAM3 access is not ready; do not use this as the final accepted run
bash scripts/run_phase0_to5.sh --strict-dual-run false --strict-sam3-permission false
```

Generated artifacts:
- Unified experiment IDs: `phase0_<tag>` ... `phase5_<tag>`
- Full log: `outputs/logs/run_phase0_to5_<tag>.log`
- Summary: `outputs/logs/run_phase0_to5_<tag>_summary.txt`

After Phase 0-5, run the Phase 6 gate in section 8 and rebuild final artifacts in section 9.

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

### Model Weights and External Checkouts

Most external repositories and checkpoints are not vendored in git. The expected local paths are configured in `configs/base.yaml` and used by the accepted runs:

- Part 1: `yolov8n-seg.pt` at repo root for YOLOv8-seg; Mask R-CNN weights are resolved by torchvision.
- Part 2: SAM2 checkpoint at `outputs/external/part2/checkpoints/sam2.1_hiera_tiny.pt`.
- Part 2: Track Anything checkpoints at `outputs/external/part2/checkpoints/sam_vit_h_4b8939.pth`, `XMem-s012.pth`, and `E2FGVI-HQ-CVPR22.pth`.
- Part 2/3/6: ProPainter external checkout is auto-managed under `outputs/external/part2/repos/ProPainter`; this workspace also keeps ProPainter-related weights in `weights/`.
- Part 3/6: SAM3 checkpoint at `outputs/external/part3/checkpoints/sam3.pt` with the SAM3 subprocess environment named `sam3`.
- Part 4/F route: VGGT4D checkout at `data/external/vggt4d` and checkpoint at `data/external/vggt4d/ckpts/model_tracker_fixed_e20.pt`, run through the `vggt4d` conda environment.
- Phase 5/G route: `stable-diffusion-v1-5/stable-diffusion-inpainting` is loaded through the diffusers cache.
- FAST-VQA: official checkout expected at `outputs/external/quality/FAST-VQA-and-FasterVQA` with official `FAST-VQA-M` weights.

The public GitHub repository should document or link how to obtain these files. Do not commit large checkpoints unless the course explicitly asks for them.

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

### 7) Phase 4 (F1-F5 + Gate, VGGT4D Prior Backend Ablation)

Phase 4 Route F evaluates VGGT4D prior with the B-best backend, the alternate backend, and trajectory refinement candidates, then exports the best global candidate by `MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR`. `prior` means the VGGT4D raw mask converted to prompt anchors and passed through a video mask backend with the same `bidirectional_no_wrap` propagation policy used by B-best. The propagation policy treats videos as non-cyclic, so no backend may connect the last frame back to the first frame.

F-stage semantics:
- `F1`: VGGT4D prior → B-best mask backend + bidirectional no-wrap → ProPainter.
- `F2`: unchanged B-best baseline (reference).
- `F3`: VGGT4D prior → backend not selected by Phase2 B-best. If B-best uses Track Anything, F3 uses SAM2; if B-best uses SAM2, F3 uses Track Anything.
- `F4`: trajectory/bidirectional refinement on the best earlier F mask.
- `F-best`: selected globally by MaskScore across successful F-stage candidates.
- Legacy YOLO/VGGT fusion-prior comparisons are not part of the final Phase 4 reporting scope.

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

### 8) Phase 6 (E/F Core Stacking Gate)

Phase 6 is a narrow confirmation gate, not a broad comparison search. H0 re-runs only three traceable references, `B-best`, `E-best`, and `F-best`; H1 tests the single stacked core change, `F-best prior + SAM3 refinement`. The final Phase6-best is selected by `MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR`; color TCF and FAST-VQA are reported as video-quality side metrics.
`scripts/run_phase6.sh` defaults to running the driver in the `aiaa3201` conda environment; pass `--conda-env current` to use the current shell Python.
For a new Phase 0-5 run, pass the actual Phase 2/3/4 experiment IDs from `outputs/logs/run_phase0_to5_<tag>_summary.txt`. The command below uses the accepted final IDs.

```bash
bash scripts/run_phase6.sh \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-id "phase6_core_maskscore_fastvqa_$(date +%Y%m%d_%H%M%S)_pl220" \
  --phase2-exp-id phase2_maskscore_fastvqa_20260510_023457_pl220 \
  --phase3-exp-id phase3_maskscore_fastvqa_20260510_023457_pl220 \
  --phase4-exp-id phase4_maskscore_fastvqa_altbackend_20260510_pl220 \
  --sam3-env-name sam3 \
  --strict-sam3-permission true \
  --seed 42
```

Expected Phase 6 outputs include:
- `outputs/metrics/<exp_id>/phase6_ablation.csv`
- `outputs/metrics/<exp_id>/phase6_selection.json`
- `outputs/metrics/<exp_id>/phase6_b_vs_h.csv`
- `outputs/metrics/<exp_id>/phase6_efh_jmjr.csv`
- `outputs/metrics/<exp_id>/phase6_pareto.csv`
- `outputs/metrics/<exp_id>/phase6_run_meta.json`
- `outputs/metrics/<exp_id>/phase6_acceptance_report.md`

Phase 6 candidates:
- `H0/b_best_ref`: unchanged B-best mask.
- `H0/e_best_ref`: B-best mask with the E-best morphology + SAM3 refinement path.
- `H0/f_best_ref`: current F-best, `VGGT4D prior + SAM2`.
- `H1/f_best_then_sam3`: current F-best followed by SAM3 refinement.

To run the gate separately:

```bash
bash scripts/check_phase6.sh \
  --exp-id "$PHASE6_EXP" \
  --config configs/base.yaml \
  --strict-sam3-permission true
```

For a quick smoke test, add `--max-frames 5`.

Accepted Phase 6 run:
- Exp ID: `phase6_core_maskscore_fastvqa_20260511_235610_pl220`
- Gate: PASS with strict SAM3 permission.
- Phase6-best: `H1/f_best_then_sam3`
- Aggregate: `GT_Coverage=0.9741`, `JM=0.7075`, `JR=0.7688`, `MaskScore=0.8561`, `TCF(color)=0.0697`, `FAST_VQA=0.1234`
- Note: this confirms the F-best core change plus SAM3 slightly improves the global mask score while preserving the same JR and essentially the same JM as F-best.

Phase 6 accepted candidates:

| Candidate | GT_Coverage | JM | JR | MaskScore | TCF | FAST_VQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `H0/b_best_ref` | 0.9196 | 0.6378 | 0.7500 | 0.8067 | 0.0649 | 0.1190 |
| `H0/e_best_ref` | 0.9531 | 0.6328 | 0.7625 | 0.8254 | 0.0647 | 0.1195 |
| `H0/f_best_ref` | 0.9722 | 0.7074 | 0.7688 | 0.8552 | 0.0698 | 0.1244 |
| `H1/f_best_then_sam3` | 0.9741 | 0.7075 | 0.7688 | 0.8561 | 0.0697 | 0.1234 |

### 9) Final Tables And Figures

After Phase 1/2/3/4/5/6 outputs exist, build paper-ready tables and figures:

```bash
bash scripts/build_results_artifacts.sh
```

Outputs:
- `outputs/metrics/final_results/table1_main_performance.{csv,md,tex}`
- `outputs/metrics/final_results/table2_a_ablation.{csv,md,tex}`
- `outputs/metrics/final_results/table3_bef_phase6_ablation.{csv,md,tex}`
- `outputs/metrics/final_results/table4_phase5_qualitative.{csv,md,tex}`
- `outputs/metrics/final_results/final_results_summary.md`
- `outputs/figures/final_results/figure*.png`

Current main results:

| Method | GT_Coverage | JM | JR | MaskScore | TCF(color) | FAST_VQA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A-best | 0.9222 | 0.6059 | 0.7188 | 0.7923 | 0.0608 | 0.0835 |
| B-best | 0.9208 | 0.6387 | 0.7562 | 0.8091 | 0.0640 | 0.1060 |
| B+E | 0.9531 | 0.6328 | 0.7625 | 0.8254 | 0.0647 | 0.1258 |
| B+F | 0.9722 | 0.7074 | 0.7688 | 0.8552 | 0.0698 | 0.1201 |
| Phase6-best | 0.9741 | 0.7075 | 0.7688 | 0.8561 | 0.0697 | 0.1234 |

Representative visual result artifacts:
- `outputs/figures/final_results/figure2_a_vs_b_global.png`
- `outputs/figures/final_results/figure3_b_e_f_phase6_visual.png`
- `outputs/figures/final_results/figure4_b_e_f_phase6_boundary.png`
- `outputs/figures/final_results/figure8_phase5_failure_strip.png`

`outputs/` is gitignored because these files can be large. For the final public repository, include a small approved visual subset in the report, release artifacts, or another tracked/lightweight location and keep the table numbers traceable to `outputs/metrics/final_results/`.

### 10) Phase 5 (Route G + Gate, Diffusion Inpainting)

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
  - `check_phase3 --strict-sam3-permission true`
  - `check_phase6 --strict-sam3-permission true`
- `--strict-sam3-permission false` is only for debugging or access-limited runs; mark such results as non-final.

## Output Contract (Default `video_only=true`)

Stable output locations:

- Restored prediction video: `outputs/videos/<exp_id>/<dataset>/restored_h264.mp4`
- Predicted mask video: `outputs/videos/<exp_id>/<dataset>/mask_h264.mp4`
- Predicted mask overlay on the original input: `outputs/videos/<exp_id>/<dataset>/mask_overlay_h264.mp4`
- GT mask videos: `outputs/videos/gt/<dataset>/mask_h264.mp4`
- GT mask overlay on the original input: `outputs/videos/gt/<dataset>/mask_overlay_h264.mp4`
- Metrics root: `outputs/metrics/<exp_id>/`

Important behavior:

- Intermediate `_candidates/` may be auto-cleaned.
- `frames/` and `masks/` directories may be auto-cleaned when video-only output is enabled.
- `outputs/videos/gt/wild/gt_missing.json` is expected because the wild video has no GT mask.

To regenerate overlay videos for the accepted final experiments:

```bash
bash scripts/export_video_overlays.sh \
  --config configs/base.yaml \
  --datasets mandatory \
  --exp-ids final
```

Phase 4 backend prior audit:
- `phase4_backend_priors.csv` records `vggt4d_raw`, `b_best_backend`, and `alternate_backend` mask ratios.
- F3 is the alternate-backend VGGT4D-prior test, not a YOLO/VGGT fusion prior comparison.

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

- Mask quality: `GT_Coverage = |pred ∩ gt| / |gt|`, JM (IoU mean), JR (IoU recall), and `MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR`
- Video quality: color TCF (lower is better), FAST-VQA (higher is better when configured)
- PSNR/SSIM from the project brief should be reported only when clean background GT frames are available. The mandatory DAVIS-derived sequences provide object-mask GT, not clean removed-background GT, so the accepted final tables use color TCF + FAST-VQA for video quality and explain this in the report.
- Removed legacy internal metrics: ROS and BES are no longer reported or used for ranking.

TCF is a temporal smoothness proxy, not a standalone inpainting-quality metric. Blurry, low-frequency, or wrong-but-stable fills can produce a lower TCF than sharper and more realistic fills because TCF only measures color-frame differences after optical-flow warping in the predicted-mask ROI. In the accepted runs, this explains why A-best and the qualitative G route can have low TCF despite worse visual quality. FAST-VQA aligns better with human inspection in our results and is used together with qualitative figures to discuss generated-video quality.

FAST-VQA is configured through `scripts/fast_vqa_score.py`, which calls the official `FAST-VQA-and-FasterVQA` checkout under `outputs/external/quality/` with the official `FAST-VQA-M` weights. The evaluator writes `FAST_VQA` per dataset and caches `fast_vqa_score.json` next to each output video.

## Final Submission Packaging

Local accepted deliverables are:

- Final method videos: `outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/{wild,bmx-trees,tennis}/restored_h264.mp4`
- Final masks/overlays: same folders, `mask_h264.mp4` and `mask_overlay_h264.mp4`
- Final tables: `outputs/metrics/final_results/table*.{csv,md,tex}`
- Final figures: `outputs/figures/final_results/figure*.png`
- Submission TODO/checklist: `docs/SUBMISSION_TODO.md`

Create `videos.zip` from the accepted Phase6-best restored videos:

```bash
mkdir -p outputs/submission_videos
cp outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/wild/restored_h264.mp4 outputs/submission_videos/wild_restored_h264.mp4
cp outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/bmx-trees/restored_h264.mp4 outputs/submission_videos/bmx-trees_restored_h264.mp4
cp outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/tennis/restored_h264.mp4 outputs/submission_videos/tennis_restored_h264.mp4
zip -j videos.zip outputs/submission_videos/*.mp4
```

## Notes

- Follow repository conventions in `AGENTS.md`.
- Follow execution schedule and milestones in `PLAN.md`.
