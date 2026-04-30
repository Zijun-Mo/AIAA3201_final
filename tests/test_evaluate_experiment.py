import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.common.evaluate_experiment import compute_mask_metrics


class TestEvaluateExperimentMasks(unittest.TestCase):
    def test_compute_mask_metrics_unions_recursive_gt_parts(self):
        pred = np.zeros((12, 12), dtype=np.uint8)
        pred[1:5, 1:5] = 255
        pred[7:10, 7:11] = 255

        with tempfile.TemporaryDirectory() as tmp:
            gt_root = Path(tmp)
            part_a = gt_root / "object_a"
            part_b = gt_root / "object_b"
            part_a.mkdir()
            part_b.mkdir()

            gt_a = np.zeros_like(pred)
            gt_b = np.zeros_like(pred)
            gt_a[1:5, 1:5] = 255
            gt_b[7:10, 7:11] = 255
            cv2.imwrite(str(part_a / "frame_000000.png"), gt_a)
            cv2.imwrite(str(part_b / "frame_000000.png"), gt_b)

            metrics, note = compute_mask_metrics(
                pred_masks=[pred],
                pred_frame_names=["frame_000000.png"],
                gt_mask_dir=gt_root,
                threshold=0.5,
            )

        self.assertIsNone(note)
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["JM"], 1.0, places=6)
        self.assertAlmostEqual(metrics["JR"], 1.0, places=6)
        self.assertEqual(metrics["gt_mask_parts_merged"], 2)


if __name__ == "__main__":
    unittest.main()
