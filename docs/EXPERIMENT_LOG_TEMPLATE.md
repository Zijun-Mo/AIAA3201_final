# Experiment Log Template

## Metadata
- Experiment ID:
- Date:
- Owner:
- Route: A / B / E / F / G / Phase6
- Dataset: Wild / bmx-trees / tennis / DAVIS
- Input version (raw file hash or source tag):
- Seed:
- Git commit:

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
- Acceptance report path:
- Run metadata path:
- Figure output path:
- Log path:
- Restored video path:
- Mask/overlay video path:

## Results (if available)
- GT_Coverage:
- JM:
- JR:
- MaskScore:
- TCF(color):
- FAST_VQA:
- PSNR/SSIM (only if clean background GT exists):
- Selection status: selected / rejected / qualitative-only / failed
- Dataset aggregation notes:

## Qualitative Findings
- Boundary artifacts:
- Temporal flicker:
- Texture quality:
- Failure cases:
- TCF interpretation caveat:

## Conclusion
- Keep / discard this setting:
- Next step:
- Report wording needed:

## Reproducible Commands
```bash
# Phase 0 preprocess
bash scripts/preprocess.sh --datasets mandatory --overwrite

# Unified evaluation (set your exp id)
bash scripts/evaluate.sh --config configs/base.yaml --exp-id <exp_id> --datasets mandatory --pred-root outputs/videos --gt-root data/gt --allow-missing-gt true

# Phase 0 gate check
bash scripts/check_phase0.sh --exp-id <exp_id> --config configs/base.yaml

# Example route gate checks
bash scripts/check_phase1.sh --exp-id <phase1_exp_id> --config configs/base.yaml
bash scripts/check_phase2.sh --exp-id <phase2_exp_id> --config configs/base.yaml --strict-dual-run false
bash scripts/check_phase3.sh --exp-id <phase3_exp_id> --config configs/base.yaml --strict-sam3-permission true
bash scripts/check_phase4.sh --exp-id <phase4_exp_id> --config configs/base.yaml
bash scripts/check_phase5.sh --exp-id <phase5_exp_id> --config configs/base.yaml
bash scripts/check_phase6.sh --exp-id <phase6_exp_id> --config configs/base.yaml --strict-sam3-permission true
```

## Final Metric Policy
- Use `MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR` for mask-best selection.
- Report color TCF and FAST-VQA as video-quality side metrics.
- Do not report ROS/BES as final metrics.
- Report PSNR/SSIM only for experiments with clean removed-background GT; otherwise explain why they are not applicable.
