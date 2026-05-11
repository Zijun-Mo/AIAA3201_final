# Submission TODO

Current local documentation state: 2026-05-12. This file tracks final packaging and report risks; it should not be treated as proof that external Canvas/GitHub/arXiv steps are complete.

## External Submission Items

- [ ] Public GitHub repository is available and README renders correctly online.
- [ ] CVPR-format PDF is complete, 6-8 body pages excluding references.
- [ ] GitHub repository link is included at the end of the abstract.
- [ ] arXiv upload is complete with camera-ready template settings.
- [ ] `videos.zip` is created and contains one processed restored video for each mandatory dataset: `wild`, `bmx-trees`, `tennis`.

## Local Evidence Already Generated

- [x] Mandatory Phase6-best restored videos exist:
  - `outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/wild/restored_h264.mp4`
  - `outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/bmx-trees/restored_h264.mp4`
  - `outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/tennis/restored_h264.mp4`
- [x] Final quantitative tables exist in `outputs/metrics/final_results/`.
- [x] Final qualitative figures exist in `outputs/figures/final_results/`.
- [x] Accepted Phase6 gate passed: `phase6_core_maskscore_fastvqa_20260511_235610_pl220`.

## Report Checklist

- [ ] Main table uses the final metric schema: `GT_Coverage`, JM, JR, `MaskScore`, color TCF, FAST-VQA.
- [ ] Report explicitly states that `MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR`.
- [ ] Report states that `wild` has no GT mask and is excluded from `GT_Coverage/JM/JR/MaskScore` aggregation, while still included in video outputs.
- [ ] Report does not use legacy ROS/BES numbers as final metrics.
- [ ] If PSNR/SSIM are not reported, report explains that mandatory data have object-mask GT rather than clean removed-background GT; otherwise add PSNR/SSIM only for experiments with clean background GT.
- [ ] Qualitative section includes A-best vs B-best, B/E/F/Phase6 comparison, boundary detail, hard-scene analysis, and Phase 5 diffusion failure strip.
- [ ] Failure cases discuss A-route blur/boundary residuals, TCF limitations, and G-route flicker/edge seams/structure hallucination.
- [ ] Related work cites all papers from `project3.txt` Section 4 plus any additional methods used.

## Accepted Final IDs

| Role | Exp ID | Notes |
| --- | --- | --- |
| A-best | `phase1_maskscore_fastvqa_20260510_023457_pl220` | Mask R-CNN + flow + Telea + temporal borrowing |
| B-best | `phase2_maskscore_fastvqa_20260510_023457_pl220` | Track Anything coarse + ProPainter |
| B+E | `phase3_maskscore_fastvqa_20260510_023457_pl220` | SAM3 refinement |
| B+F | `phase4_maskscore_fastvqa_altbackend_20260510_pl220` | VGGT4D prior + SAM2 alternate backend |
| Phase6-best | `phase6_core_maskscore_fastvqa_20260511_235610_pl220` | VGGT4D prior + SAM2 -> SAM3 |
| G qualitative | `phase5_maskscore_fastvqa_20260510_023457_pl220` | Diffusion failure/qualitative analysis only |

## Packaging Command

```bash
mkdir -p outputs/submission_videos
cp outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/wild/restored_h264.mp4 outputs/submission_videos/wild_restored_h264.mp4
cp outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/bmx-trees/restored_h264.mp4 outputs/submission_videos/bmx-trees_restored_h264.mp4
cp outputs/videos/phase6_core_maskscore_fastvqa_20260511_235610_pl220/tennis/restored_h264.mp4 outputs/submission_videos/tennis_restored_h264.mp4
zip -j videos.zip outputs/submission_videos/*.mp4
```
