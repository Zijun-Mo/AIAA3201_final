import unittest
from pathlib import Path

import numpy as np

from src.part3.run_explore import (
    CandidateResult,
    CandidateSpec,
    apply_morph_profile,
    build_prior_prompt_from_masks,
    decide_e3_flow,
    select_f_route_best,
    select_best,
    temporal_smooth_masks,
)


class TestPhase3Utils(unittest.TestCase):
    def test_build_prior_prompt_from_masks_uses_largest_prior_frame(self):
        masks = []
        for _ in range(3):
            masks.append(np.zeros((32, 32), dtype=np.uint8))
        masks[0][2:5, 2:5] = 255
        masks[2][10:20, 12:24] = 255

        frame_idx, boxes, instances, meta = build_prior_prompt_from_masks(
            masks_u8=masks,
            frame_shape=(32, 32),
            max_prompts=2,
            min_area_ratio=0.0,
        )

        self.assertEqual(frame_idx, 2)
        self.assertEqual(boxes[0], (12, 10, 23, 19))
        self.assertEqual(len(instances), 3)
        self.assertEqual(meta["prompt_source"], "vggt4d_connected_components")

    def test_apply_morph_profile_binary_output(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        mask[0, 0] = 255
        out = apply_morph_profile(
            masks_u8=[mask],
            opening_kernel=3,
            closing_kernel=5,
            dilation_kernel=3,
            smoothing_kernel=3,
        )[0]
        self.assertTrue(set(np.unique(out)).issubset({0, 255}))
        self.assertGreater(int((out > 0).sum()), 0)

    def test_temporal_smooth_removes_single_frame_noise(self):
        m0 = np.zeros((32, 32), dtype=np.uint8)
        m1 = np.zeros((32, 32), dtype=np.uint8)
        m2 = np.zeros((32, 32), dtype=np.uint8)
        m1[10:14, 10:14] = 255

        out = temporal_smooth_masks([m0, m1, m2], temporal_window=1)
        self.assertEqual(len(out), 3)
        self.assertEqual(int((out[1] > 0).sum()), 0)

    def test_selection_excludes_wild(self):
        spec_a = CandidateSpec(
            stage="E1",
            name="a",
            source_stage="B",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )
        spec_b = CandidateSpec(
            stage="E1",
            name="b",
            source_stage="B",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )

        res_a = CandidateResult(
            spec=spec_a,
            candidate_root=Path("/tmp/a"),
            eval_exp_id="exp_a",
            summary_path=Path("/tmp/a/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": 0.95, "JR": 0.95, "Q_REMOVE": 0.90}},
                "bmx-trees": {"metrics": {"JM": 0.70, "JR": 0.75, "Q_REMOVE": 0.65}},
                "tennis": {"metrics": {"JM": 0.68, "JR": 0.74, "Q_REMOVE": 0.64}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.1},
                "bmx-trees": {"mean_mask_ratio": 0.1},
                "tennis": {"mean_mask_ratio": 0.1},
            },
            stage_mask_meta={},
            propainter_meta={},
        )
        res_b = CandidateResult(
            spec=spec_b,
            candidate_root=Path("/tmp/b"),
            eval_exp_id="exp_b",
            summary_path=Path("/tmp/b/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": 0.10, "JR": 0.12, "Q_REMOVE": 0.20}},
                "bmx-trees": {"metrics": {"JM": 0.80, "JR": 0.83, "Q_REMOVE": 0.82}},
                "tennis": {"metrics": {"JM": 0.79, "JR": 0.82, "Q_REMOVE": 0.81}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.1},
                "bmx-trees": {"mean_mask_ratio": 0.1},
                "tennis": {"mean_mask_ratio": 0.1},
            },
            stage_mask_meta={},
            propainter_meta={},
        )

        best = select_best(stage="E1", entries=[res_a, res_b], score_datasets=["bmx-trees", "tennis"])
        self.assertEqual(best.spec.name, "b")

    def test_decide_e3_flow_split(self):
        self.assertEqual(decide_e3_flow(permission_passed=False, runtime_error=None), "abort")
        self.assertEqual(decide_e3_flow(permission_passed=True, runtime_error="oom"), "skip")
        self.assertEqual(decide_e3_flow(permission_passed=True, runtime_error=None), "continue")

    def test_select_best_respects_coverage_constraints(self):
        spec_hi = CandidateSpec(
            stage="E4",
            name="high_q",
            source_stage="E3",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )
        spec_ok = CandidateSpec(
            stage="E4",
            name="covered",
            source_stage="E3",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )

        hi = CandidateResult(
            spec=spec_hi,
            candidate_root=Path("/tmp/high_q"),
            eval_exp_id="exp_high_q",
            summary_path=Path("/tmp/high_q/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": None, "JR": None, "Q_REMOVE": 0.99}},
                "bmx-trees": {"metrics": {"JM": 0.62, "JR": 0.8, "Q_REMOVE": 0.91}},
                "tennis": {"metrics": {"JM": 0.80, "JR": 1.0, "Q_REMOVE": 0.97}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.0001, "active_frame_ratio": 0.01},
                "bmx-trees": {"mean_mask_ratio": 0.04, "active_frame_ratio": 0.8},
                "tennis": {"mean_mask_ratio": 0.02, "active_frame_ratio": 0.7},
            },
            stage_mask_meta={},
            propainter_meta={},
        )
        covered = CandidateResult(
            spec=spec_ok,
            candidate_root=Path("/tmp/covered"),
            eval_exp_id="exp_covered",
            summary_path=Path("/tmp/covered/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": None, "JR": None, "Q_REMOVE": 0.95}},
                "bmx-trees": {"metrics": {"JM": 0.61, "JR": 0.79, "Q_REMOVE": 0.90}},
                "tennis": {"metrics": {"JM": 0.79, "JR": 0.99, "Q_REMOVE": 0.96}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.01, "active_frame_ratio": 0.6},
                "bmx-trees": {"mean_mask_ratio": 0.04, "active_frame_ratio": 0.8},
                "tennis": {"mean_mask_ratio": 0.02, "active_frame_ratio": 0.7},
            },
            stage_mask_meta={},
            propainter_meta={},
        )

        best = select_best(
            stage="E4",
            entries=[hi, covered],
            score_datasets=["wild", "bmx-trees", "tennis"],
            coverage_constraints={"wild": {"min_mean_mask_ratio": 0.002, "min_active_frame_ratio": 0.25}},
            enforce_if_candidate_available=True,
        )
        self.assertEqual(best.spec.name, "covered")

    def test_select_best_supports_primary_metric_override(self):
        spec_quality = CandidateSpec(
            stage="E4",
            name="quality_first",
            source_stage="E3",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )
        spec_mask = CandidateSpec(
            stage="E4",
            name="mask_first",
            source_stage="E3",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )

        quality_first = CandidateResult(
            spec=spec_quality,
            candidate_root=Path("/tmp/quality_first"),
            eval_exp_id="exp_quality_first",
            summary_path=Path("/tmp/quality_first/summary.json"),
            aggregate={},
            per_dataset={
                "bmx-trees": {"metrics": {"JM": 0.62, "JR": 0.70, "Q_REMOVE": 0.95}},
                "tennis": {"metrics": {"JM": 0.61, "JR": 0.69, "Q_REMOVE": 0.94}},
            },
            mask_stats={
                "bmx-trees": {"mean_mask_ratio": 0.04, "active_frame_ratio": 1.0},
                "tennis": {"mean_mask_ratio": 0.04, "active_frame_ratio": 1.0},
            },
            stage_mask_meta={},
            propainter_meta={},
        )
        mask_first = CandidateResult(
            spec=spec_mask,
            candidate_root=Path("/tmp/mask_first"),
            eval_exp_id="exp_mask_first",
            summary_path=Path("/tmp/mask_first/summary.json"),
            aggregate={},
            per_dataset={
                "bmx-trees": {"metrics": {"JM": 0.72, "JR": 0.80, "Q_REMOVE": 0.88}},
                "tennis": {"metrics": {"JM": 0.71, "JR": 0.79, "Q_REMOVE": 0.87}},
            },
            mask_stats={
                "bmx-trees": {"mean_mask_ratio": 0.04, "active_frame_ratio": 1.0},
                "tennis": {"mean_mask_ratio": 0.04, "active_frame_ratio": 1.0},
            },
            stage_mask_meta={},
            propainter_meta={},
        )

        best_quality = select_best(
            stage="E4",
            entries=[quality_first, mask_first],
            score_datasets=["bmx-trees", "tennis"],
            primary_metric="quality",
        )
        self.assertEqual(best_quality.spec.name, "quality_first")

        best_mask = select_best(
            stage="E4",
            entries=[quality_first, mask_first],
            score_datasets=["bmx-trees", "tennis"],
            primary_metric="mask",
        )
        self.assertEqual(best_mask.spec.name, "mask_first")

    def test_select_f_route_best_uses_jm_jr_priority(self):
        spec_f1 = CandidateSpec(
            stage="F1",
            name="f1_vggt_replace",
            source_stage="B",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
            f_source_key="vggt4d",
        )
        spec_f3 = CandidateSpec(
            stage="F3",
            name="f3_fused",
            source_stage="F2",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )

        f1 = CandidateResult(
            spec=spec_f1,
            candidate_root=Path("/tmp/f1"),
            eval_exp_id="exp_f1",
            summary_path=Path("/tmp/f1/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": 0.75, "JR": 0.82, "Q_REMOVE": 0.70}},
                "bmx-trees": {"metrics": {"JM": 0.78, "JR": 0.84, "Q_REMOVE": 0.71}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.10},
                "bmx-trees": {"mean_mask_ratio": 0.11},
            },
            stage_mask_meta={},
            propainter_meta={},
        )
        f3 = CandidateResult(
            spec=spec_f3,
            candidate_root=Path("/tmp/f3"),
            eval_exp_id="exp_f3",
            summary_path=Path("/tmp/f3/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": 0.65, "JR": 0.70, "Q_REMOVE": 0.95}},
                "bmx-trees": {"metrics": {"JM": 0.66, "JR": 0.71, "Q_REMOVE": 0.96}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.10},
                "bmx-trees": {"mean_mask_ratio": 0.11},
            },
            stage_mask_meta={},
            propainter_meta={},
        )

        best = select_f_route_best(
            entries=[f1, f3],
            score_datasets=["wild", "bmx-trees"],
            coverage_constraints={},
            enforce_if_candidate_available=True,
        )
        self.assertEqual(best.spec.name, "f1_vggt_replace")

    def test_select_f_route_best_forces_vggt4d_prior_final(self):
        spec_vggt = CandidateSpec(
            stage="F1",
            name="bbest_vggt4d_replace_yolo",
            source_stage="B",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
            f_source_key="vggt4d",
        )
        spec_yolo = CandidateSpec(
            stage="F2",
            name="bbest_baseline",
            source_stage="B",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )

        vggt = CandidateResult(
            spec=spec_vggt,
            candidate_root=Path("/tmp/vggt"),
            eval_exp_id="exp_vggt",
            summary_path=Path("/tmp/vggt/summary.json"),
            aggregate={},
            per_dataset={
                "bmx-trees": {"metrics": {"JM": 0.50, "JR": 0.60, "Q_REMOVE": 0.70}},
                "tennis": {"metrics": {"JM": 0.51, "JR": 0.61, "Q_REMOVE": 0.71}},
            },
            mask_stats={
                "bmx-trees": {"mean_mask_ratio": 0.03, "active_frame_ratio": 1.0},
                "tennis": {"mean_mask_ratio": 0.03, "active_frame_ratio": 1.0},
            },
            stage_mask_meta={},
            propainter_meta={},
        )
        yolo = CandidateResult(
            spec=spec_yolo,
            candidate_root=Path("/tmp/yolo"),
            eval_exp_id="exp_yolo",
            summary_path=Path("/tmp/yolo/summary.json"),
            aggregate={},
            per_dataset={
                "bmx-trees": {"metrics": {"JM": 0.90, "JR": 0.95, "Q_REMOVE": 0.96}},
                "tennis": {"metrics": {"JM": 0.91, "JR": 0.96, "Q_REMOVE": 0.97}},
            },
            mask_stats={
                "bmx-trees": {"mean_mask_ratio": 0.10, "active_frame_ratio": 1.0},
                "tennis": {"mean_mask_ratio": 0.10, "active_frame_ratio": 1.0},
            },
            stage_mask_meta={},
            propainter_meta={},
        )

        best = select_f_route_best(
            entries=[vggt, yolo],
            score_datasets=["bmx-trees", "tennis"],
            coverage_constraints={},
            enforce_if_candidate_available=True,
        )
        self.assertEqual(best.spec.name, "bbest_vggt4d_replace_yolo")

    def test_select_f_route_best_does_not_export_fusion_candidate(self):
        spec_prior = CandidateSpec(
            stage="F1",
            name="bbest_vggt4d_replace_yolo",
            source_stage="B",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
            f_source_key="vggt4d",
        )
        spec_fusion = CandidateSpec(
            stage="F3",
            name="vggt4d_guided",
            source_stage="F2",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
            f_fusion_method="vggt4d_guided",
        )

        prior = CandidateResult(
            spec=spec_prior,
            candidate_root=Path("/tmp/prior"),
            eval_exp_id="exp_prior",
            summary_path=Path("/tmp/prior/summary.json"),
            aggregate={},
            per_dataset={
                "bmx-trees": {"metrics": {"JM": 0.40, "JR": 0.50, "Q_REMOVE": 0.60}},
                "tennis": {"metrics": {"JM": 0.41, "JR": 0.51, "Q_REMOVE": 0.61}},
            },
            mask_stats={
                "bmx-trees": {"mean_mask_ratio": 0.04, "active_frame_ratio": 1.0},
                "tennis": {"mean_mask_ratio": 0.04, "active_frame_ratio": 1.0},
            },
            stage_mask_meta={},
            propainter_meta={},
        )
        fusion = CandidateResult(
            spec=spec_fusion,
            candidate_root=Path("/tmp/fusion"),
            eval_exp_id="exp_fusion",
            summary_path=Path("/tmp/fusion/summary.json"),
            aggregate={},
            per_dataset={
                "bmx-trees": {"metrics": {"JM": 0.90, "JR": 0.95, "Q_REMOVE": 0.96}},
                "tennis": {"metrics": {"JM": 0.91, "JR": 0.96, "Q_REMOVE": 0.97}},
            },
            mask_stats={
                "bmx-trees": {"mean_mask_ratio": 0.10, "active_frame_ratio": 1.0},
                "tennis": {"mean_mask_ratio": 0.10, "active_frame_ratio": 1.0},
            },
            stage_mask_meta={},
            propainter_meta={},
        )

        best = select_f_route_best(
            entries=[prior, fusion],
            score_datasets=["bmx-trees", "tennis"],
            coverage_constraints={},
            enforce_if_candidate_available=True,
        )
        self.assertEqual(best.spec.name, "bbest_vggt4d_replace_yolo")


if __name__ == "__main__":
    unittest.main()
