# Experiment Log Template

## Metadata
- Experiment ID:
- Date:
- Owner:
- Route: A / B / E / F / G
- Dataset: Wild / bmx-trees / tennis / DAVIS
- Input version (raw file hash or source tag):

## Objective
- What is tested:
- Why this matters:

## Phase 0 Required Configuration
- Preprocess config:
  - target_fps:
  - target_resolution (W x H):
  - frame_format:
  - frame_name_template:
- Evaluation config:
  - jr_iou_threshold:
  - allow_missing_gt:
  - save_visualization:

## Runtime Configuration
- Model / method:
- Key parameters:
- Input resolution / fps:
- Hardware:
- Runtime:
- GPU memory:

## Artifacts
- Processed manifest path:
- Metrics summary path:
- Metrics CSV path:
- Figure output path:
- Log path:

## Results (if available)
- JM:
- JR:
- PSNR:
- SSIM:

## Qualitative Findings
- Boundary artifacts:
- Temporal flicker:
- Texture quality:
- Failure cases:

## Conclusion
- Keep / discard this setting:
- Next step:

## Reproducible Commands
```bash
# Phase 0 preprocess
bash scripts/preprocess.sh --datasets mandatory --overwrite

# Unified evaluation (set your exp id)
bash scripts/evaluate.sh --config configs/base.yaml --exp-id <exp_id> --datasets mandatory --pred-root outputs/videos --gt-root data/gt --allow-missing-gt true

# Phase 0 gate check
bash scripts/check_phase0.sh --exp-id <exp_id> --config configs/base.yaml
```
