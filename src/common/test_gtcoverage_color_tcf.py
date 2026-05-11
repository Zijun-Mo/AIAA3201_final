#!/usr/bin/env python3
"""Smoke tests for mask scoring, GT_Coverage, and color TCF."""
from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.evaluate_experiment import compute_mask_metrics
from src.common.mask_ranking import select_maskscore_best
from src.common.remove_quality import compute_tcf_per_frame


def test_gt_coverage_union_mask() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gt_dir = root / "gt"
        gt_dir.mkdir()
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[0, 0] = 38
        gt[1, 1] = 75
        cv2.imwrite(str(gt_dir / "frame_000000_a.png"), gt)

        pred = np.zeros((4, 4), dtype=np.uint8)
        pred[0, 0] = 255
        pred[3, 3] = 255
        metrics, err = compute_mask_metrics([pred], ["frame_000000.png"], gt_dir, threshold=0.5)
        assert err is None
        assert metrics is not None
        assert abs(float(metrics["GT_Coverage"]) - 0.5) < 1e-6
        assert abs(float(metrics["JM"]) - (1.0 / 3.0)) < 1e-6


def test_maskscore_ranking() -> None:
    rows = [
        {"name": "high_quality_low_coverage", "GT_Coverage": 0.67, "JM": 0.95, "JR": 0.95, "TCF": 0.01},
        {"name": "lower_quality_high_coverage", "GT_Coverage": 0.70, "JM": 0.50, "JR": 0.50, "TCF": 0.10},
    ]
    best = select_maskscore_best(rows, lambda x: x)
    assert best["name"] == "high_quality_low_coverage"

    near_rows = [
        {"name": "balanced", "GT_Coverage": 0.80, "JM": 0.70, "JR": 0.70, "TCF": 0.10},
        {"name": "coverage_heavy", "GT_Coverage": 0.95, "JM": 0.40, "JR": 0.40, "TCF": 0.01},
    ]
    best_near = select_maskscore_best(near_rows, lambda x: x)
    assert best_near["name"] == "balanced"


def test_color_tcf_uses_chroma_change() -> None:
    f0 = np.zeros((8, 8, 3), dtype=np.uint8)
    f1 = np.zeros((8, 8, 3), dtype=np.uint8)
    f0[:, :] = (0, 0, 255)
    f1[:, :] = (255, 0, 0)
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    vals, empty = compute_tcf_per_frame([f0, f1], [mask, mask], dilate_kernel=1)
    assert empty == 0
    assert len(vals) == 2
    assert vals[1] > 0.1


if __name__ == "__main__":
    test_gt_coverage_union_mask()
    test_maskscore_ranking()
    test_color_tcf_uses_chroma_change()
    print("gtcoverage/color-tcf smoke ok")
