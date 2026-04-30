"""Mask fusion strategies for Phase 4 Route F."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .motion_flow import compute_motion_binary_mask

logger = logging.getLogger(__name__)


def fuse_intersection(semantic_mask: np.ndarray, motion_binary_mask: np.ndarray) -> np.ndarray:
    return (((semantic_mask > 0) & (motion_binary_mask > 0)).astype(np.uint8) * 255)


def fuse_union(semantic_mask: np.ndarray, motion_binary_mask: np.ndarray) -> np.ndarray:
    return (((semantic_mask > 0) | (motion_binary_mask > 0)).astype(np.uint8) * 255)


def fuse_weighted(
    semantic_mask: np.ndarray,
    motion_map: np.ndarray,
    alpha: float = 0.4,
    threshold: float = 0.5,
) -> np.ndarray:
    semantic_norm = (semantic_mask > 0).astype(np.float32)
    motion_max = float(motion_map.max()) if float(motion_map.max()) > 0 else 1.0
    motion_norm = np.clip(motion_map / motion_max, 0.0, 1.0)
    blended = (1.0 - float(alpha)) * semantic_norm + float(alpha) * motion_norm
    return ((blended >= float(threshold)).astype(np.uint8) * 255)


def fuse_instance_filter(
    semantic_mask: np.ndarray,
    motion_score: float,
    threshold: float = 2.0,
) -> np.ndarray:
    if float(motion_score) >= float(threshold):
        return ((semantic_mask > 0).astype(np.uint8) * 255)
    return np.zeros_like(semantic_mask, dtype=np.uint8)


def fuse_vggt4d_guided(
    semantic_mask: np.ndarray,
    motion_map: np.ndarray,
    external_mask: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.3,
    threshold: float = 0.5,
) -> np.ndarray:
    semantic_norm = (semantic_mask > 0).astype(np.float32)
    motion_max = float(motion_map.max()) if float(motion_map.max()) > 0 else 1.0
    motion_norm = np.clip(motion_map / motion_max, 0.0, 1.0)
    ext_norm = (external_mask > 0).astype(np.float32)

    gamma = max(1.0 - float(alpha) - float(beta), 0.0)
    blended = gamma * semantic_norm + float(alpha) * motion_norm + float(beta) * ext_norm
    return ((blended >= float(threshold)).astype(np.uint8) * 255)


def fuse_video_intersection(
    semantic_masks: list[np.ndarray],
    motion_maps: list[np.ndarray],
    pixel_motion_threshold: float = 1.5,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for sem, mm in zip(semantic_masks, motion_maps):
        motion_bin = compute_motion_binary_mask(mm, float(pixel_motion_threshold))
        out.append(fuse_intersection(sem, motion_bin))
    return out


def fuse_video_union(
    semantic_masks: list[np.ndarray],
    motion_maps: list[np.ndarray],
    pixel_motion_threshold: float = 1.5,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for sem, mm in zip(semantic_masks, motion_maps):
        motion_bin = compute_motion_binary_mask(mm, float(pixel_motion_threshold))
        out.append(fuse_union(sem, motion_bin))
    return out


def fuse_video_weighted(
    semantic_masks: list[np.ndarray],
    motion_maps: list[np.ndarray],
    alpha: float = 0.4,
    threshold: float = 0.5,
) -> list[np.ndarray]:
    return [fuse_weighted(sem, mm, alpha=float(alpha), threshold=float(threshold)) for sem, mm in zip(semantic_masks, motion_maps)]


def fuse_video_instance_filter(
    semantic_masks: list[np.ndarray],
    motion_scores: list[float],
    instance_filter_threshold: float = 2.0,
) -> list[np.ndarray]:
    return [
        fuse_instance_filter(sem, score, threshold=float(instance_filter_threshold))
        for sem, score in zip(semantic_masks, motion_scores)
    ]


def fuse_video_vggt4d_guided(
    semantic_masks: list[np.ndarray],
    motion_maps: list[np.ndarray],
    external_masks: list[np.ndarray],
    alpha: float = 0.3,
    beta: float = 0.3,
    threshold: float = 0.5,
) -> list[np.ndarray]:
    return [
        fuse_vggt4d_guided(sem, mm, ext, alpha=float(alpha), beta=float(beta), threshold=float(threshold))
        for sem, mm, ext in zip(semantic_masks, motion_maps, external_masks)
    ]


SUPPORTED_FUSION_METHODS = {
    "intersection",
    "union",
    "weighted",
    "instance_filter",
    "vggt4d_guided",
}


def apply_fusion(
    semantic_masks: list[np.ndarray],
    motion_maps: list[np.ndarray],
    motion_scores: list[float],
    method: str,
    fusion_cfg: dict[str, Any] | None = None,
    external_masks: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    cfg = fusion_cfg or {}
    pixel_thr = float(cfg.get("pixel_motion_threshold", 1.5))
    alpha = float(cfg.get("weighted_alpha", 0.4))
    weighted_thr = float(cfg.get("weighted_threshold", 0.5))
    inst_thr = float(cfg.get("instance_filter_threshold", 2.0))

    meta: dict[str, Any] = {"method": method}

    if method == "intersection":
        fused = fuse_video_intersection(semantic_masks, motion_maps, pixel_motion_threshold=pixel_thr)
        meta["pixel_motion_threshold"] = pixel_thr
    elif method == "union":
        fused = fuse_video_union(semantic_masks, motion_maps, pixel_motion_threshold=pixel_thr)
        meta["pixel_motion_threshold"] = pixel_thr
    elif method == "weighted":
        fused = fuse_video_weighted(semantic_masks, motion_maps, alpha=alpha, threshold=weighted_thr)
        meta["alpha"] = alpha
        meta["threshold"] = weighted_thr
    elif method == "instance_filter":
        fused = fuse_video_instance_filter(
            semantic_masks,
            motion_scores,
            instance_filter_threshold=inst_thr,
        )
        meta["instance_filter_threshold"] = inst_thr
    elif method == "vggt4d_guided":
        if external_masks is None:
            raise ValueError("vggt4d_guided fusion requires external_masks")
        vggt_alpha = float(cfg.get("vggt4d_alpha", 0.3))
        vggt_beta = float(cfg.get("vggt4d_beta", 0.3))
        vggt_thr = float(cfg.get("vggt4d_threshold", 0.5))
        fused = fuse_video_vggt4d_guided(
            semantic_masks,
            motion_maps,
            external_masks,
            alpha=vggt_alpha,
            beta=vggt_beta,
            threshold=vggt_thr,
        )
        meta["vggt4d_alpha"] = vggt_alpha
        meta["vggt4d_beta"] = vggt_beta
        meta["vggt4d_threshold"] = vggt_thr
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    semantic_active = sum(1 for m in semantic_masks if int((m > 0).sum()) > 0)
    fused_active = sum(1 for m in fused if int((m > 0).sum()) > 0)
    meta["semantic_active_frames"] = semantic_active
    meta["fused_active_frames"] = fused_active
    if semantic_active > 0 and fused_active == 0:
        logger.warning("Fusion method=%s zeroed all masks (semantic active=%d)", method, semantic_active)

    return fused, meta
