import unittest

import numpy as np

from src.part2.run_sota import (
    build_auto_prompts,
    classify_failure_case,
    compute_mean_mask_ratio,
    refine_masks,
)


class TestPhase2Utils(unittest.TestCase):
    def test_compute_mean_mask_ratio(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m1[0:5, 0:5] = 255
        m2[0:2, 0:2] = 255
        ratio = compute_mean_mask_ratio([m1, m2])
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)

    def test_build_auto_prompts_from_instances(self):
        h, w = 64, 64
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(4)]
        instances = [[], [], [], []]

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:40, 10:30] = 1
        instances[2] = [{"mask": mask}]

        prompts = build_auto_prompts(instances_per_frame=instances, frames=frames, max_prompts=3)
        self.assertEqual(prompts["frame_idx"], 2)
        self.assertTrue(len(prompts["boxes"]) >= 1)
        x1, y1, x2, y2 = prompts["boxes"][0]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)

    def test_refine_masks_temporal(self):
        masks = []
        for i in range(5):
            m = np.zeros((32, 32), dtype=np.uint8)
            m[10:15, 10 + i : 14 + i] = 255
            masks.append(m)

        refined = refine_masks(masks_u8=masks, morph_kernel=3, temporal_window=1)
        self.assertEqual(len(refined), len(masks))
        self.assertTrue(any(int((m > 0).sum()) > 0 for m in refined))

    def test_failure_classification_priority(self):
        self.assertEqual(
            classify_failure_case(
                dataset="tennis",
                psnr=30.0,
                edge_diff=0.01,
                texture_ratio=1.0,
                backend_fallback=True,
                propainter_fallback=False,
            ),
            "mask_backend_fallback",
        )
        self.assertEqual(
            classify_failure_case(
                dataset="tennis",
                psnr=30.0,
                edge_diff=0.01,
                texture_ratio=1.0,
                backend_fallback=False,
                propainter_fallback=True,
            ),
            "propainter_profile_fallback",
        )


if __name__ == "__main__":
    unittest.main()
