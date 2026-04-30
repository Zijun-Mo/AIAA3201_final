"""External dynamic cue loader for Route F."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_external_masks(
    external_dir: Path,
    dataset: str,
    target_size: tuple[int, int],
    target_count: int,
) -> list[np.ndarray] | None:
    ds_dir = external_dir / dataset
    if not ds_dir.is_dir():
        logger.warning("External mask directory missing: %s", ds_dir)
        return None

    mask_paths = sorted(p for p in ds_dir.iterdir() if p.suffix.lower() == ".png")
    if not mask_paths:
        logger.warning("No png masks found in %s", ds_dir)
        return None

    h, w = target_size
    out: list[np.ndarray] = []
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            out.append(np.zeros((h, w), dtype=np.uint8))
            continue
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out.append(((m > 127).astype(np.uint8) * 255))

    if len(out) > target_count:
        out = out[:target_count]
    while len(out) < target_count:
        out.append(np.zeros((h, w), dtype=np.uint8))

    return out
