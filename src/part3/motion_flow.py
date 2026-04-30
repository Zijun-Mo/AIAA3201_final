"""Motion and trajectory utilities for Phase 4 Route F."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


def compute_dense_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    farneback_cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = farneback_cfg or {}
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=float(cfg.get("pyr_scale", 0.5)),
        levels=int(cfg.get("levels", 3)),
        winsize=int(cfg.get("winsize", 15)),
        iterations=int(cfg.get("iterations", 3)),
        poly_n=int(cfg.get("poly_n", 5)),
        poly_sigma=float(cfg.get("poly_sigma", 1.2)),
        flags=0,
    )


def compute_motion_map(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)


def subtract_global_motion(motion_map: np.ndarray) -> np.ndarray:
    median = float(np.median(motion_map))
    return np.maximum(motion_map - median, 0.0).astype(np.float32)


def compute_motion_binary_mask(
    motion_map: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return ((motion_map > threshold).astype(np.uint8) * 255)


def compute_video_motion_maps(
    frames_bgr: list[np.ndarray],
    flow_cfg: dict[str, Any] | None = None,
    compensate_global: bool = True,
) -> list[np.ndarray]:
    if not frames_bgr:
        return []

    cfg = flow_cfg or {}
    farneback_cfg = cfg.get("farneback", {}) or {}

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    h, w = grays[0].shape[:2]
    maps: list[np.ndarray] = [np.zeros((h, w), dtype=np.float32)]

    for idx in range(1, len(grays)):
        flow = compute_dense_flow(grays[idx - 1], grays[idx], farneback_cfg)
        mm = compute_motion_map(flow)
        if compensate_global:
            mm = subtract_global_motion(mm)
        maps.append(mm)

    return maps


def compute_instance_motion_score(
    motion_map: np.ndarray,
    instance_mask: np.ndarray,
    aggregation: str = "median",
) -> float:
    roi = instance_mask > 0
    if not np.any(roi):
        return 0.0
    values = motion_map[roi]

    mode = aggregation.lower().strip()
    if mode == "mean":
        return float(np.mean(values))
    if mode == "p75":
        return float(np.percentile(values, 75))
    return float(np.median(values))


def compute_video_instance_motion_scores(
    frames_bgr: list[np.ndarray],
    masks_per_frame: list[np.ndarray],
    flow_cfg: dict[str, Any] | None = None,
    compensate_global: bool = True,
) -> list[float]:
    if not frames_bgr:
        return []
    cfg = flow_cfg or {}
    aggregation = str(cfg.get("aggregation", "median")).strip().lower()
    motion_maps = compute_video_motion_maps(frames_bgr, flow_cfg=cfg, compensate_global=compensate_global)

    scores: list[float] = []
    for mm, mask in zip(motion_maps, masks_per_frame):
        scores.append(compute_instance_motion_score(mm, mask, aggregation=aggregation))
    return scores


def compute_bidirectional_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    farneback_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fwd = compute_dense_flow(prev_gray, curr_gray, farneback_cfg)
    bwd = compute_dense_flow(curr_gray, prev_gray, farneback_cfg)
    return fwd, bwd


def compute_flow_consistency_map(
    fwd_flow: np.ndarray,
    bwd_flow: np.ndarray,
) -> np.ndarray:
    h, w = fwd_flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    wx = grid_x + fwd_flow[..., 0]
    wy = grid_y + fwd_flow[..., 1]

    bwd_at_warped = cv2.remap(
        bwd_flow,
        wx,
        wy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    round_trip = fwd_flow + bwd_at_warped
    return np.sqrt(round_trip[..., 0] ** 2 + round_trip[..., 1] ** 2).astype(np.float32)


def compute_video_flow_consistency(
    frames_bgr: list[np.ndarray],
    farneback_cfg: dict[str, Any] | None = None,
) -> list[np.ndarray]:
    if not frames_bgr:
        return []

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    h, w = grays[0].shape[:2]
    maps: list[np.ndarray] = [np.zeros((h, w), dtype=np.float32)]

    for idx in range(1, len(grays)):
        fwd, bwd = compute_bidirectional_flow(grays[idx - 1], grays[idx], farneback_cfg)
        maps.append(compute_flow_consistency_map(fwd, bwd))

    return maps


def mask_flow_reliability(
    motion_map: np.ndarray,
    consistency_map: np.ndarray,
    max_consistency_error: float = 3.0,
) -> np.ndarray:
    reliable = consistency_map < float(max_consistency_error)
    return (motion_map * reliable.astype(np.float32)).astype(np.float32)


@dataclass
class TrackSegment:
    start: int
    end: int
    mean_motion: float

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def _smooth_1d(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    w = max(1, int(window))
    if w <= 1:
        return [float(v) for v in values]

    arr = np.array(values, dtype=np.float32)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.convolve(padded, kernel, mode="valid")
    return [float(x) for x in out[: len(values)]]


def _active_segments(active: list[bool], smoothed_scores: list[float]) -> list[TrackSegment]:
    segments: list[TrackSegment] = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and active[i + 1]:
            i += 1
        end = i
        seg_scores = smoothed_scores[start : end + 1] or [0.0]
        segments.append(
            TrackSegment(start=start, end=end, mean_motion=float(np.mean(np.array(seg_scores, dtype=np.float32))))
        )
        i += 1
    return segments


def apply_trajectory_consistency(
    masks_u8: list[np.ndarray],
    motion_scores: list[float],
    trajectory_cfg: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    cfg = trajectory_cfg or {}
    min_track_length = int(cfg.get("min_track_length", 3))
    track_motion_threshold = float(cfg.get("track_motion_threshold", 1.5))
    smooth_window = int(cfg.get("motion_smooth_window", 3))
    iou_link_threshold = float(cfg.get("iou_link_threshold", 0.3))
    min_kept_active_ratio = float(cfg.get("min_kept_active_ratio", 0.9))
    min_output_active_ratio = float(cfg.get("min_output_active_ratio", 0.0))

    if not masks_u8:
        return [], {
            "min_track_length": min_track_length,
            "track_motion_threshold": track_motion_threshold,
            "motion_smooth_window": smooth_window,
            "iou_link_threshold": iou_link_threshold,
            "min_kept_active_ratio": min_kept_active_ratio,
            "min_output_active_ratio": min_output_active_ratio,
            "segments": [],
            "kept_segment_count": 0,
            "dropped_segment_count": 0,
        }

    smoothed = _smooth_1d(motion_scores[: len(masks_u8)], window=smooth_window)
    active = [bool(int((np.asarray(m) > 0).sum()) > 0) for m in masks_u8]
    segments = _active_segments(active, smoothed)

    keep_ranges: list[tuple[int, int]] = []
    segment_rows: list[dict[str, Any]] = []
    for seg in segments:
        keep = seg.length >= min_track_length and seg.mean_motion >= track_motion_threshold
        if keep:
            keep_ranges.append((seg.start, seg.end))
        segment_rows.append(
            {
                "start": seg.start,
                "end": seg.end,
                "length": seg.length,
                "mean_motion": seg.mean_motion,
                "kept": bool(keep),
            }
        )

    filtered: list[np.ndarray] = []
    for idx, m in enumerate(masks_u8):
        keep = any(lo <= idx <= hi for lo, hi in keep_ranges)
        if keep:
            filtered.append(((np.asarray(m) > 0).astype(np.uint8) * 255))
        else:
            filtered.append(np.zeros_like(np.asarray(m), dtype=np.uint8))

    input_active_ratio = float(np.mean(np.array([1.0 if x else 0.0 for x in active], dtype=np.float32)))
    filtered_active = [bool(int((np.asarray(m) > 0).sum()) > 0) for m in filtered]
    filtered_active_ratio = float(np.mean(np.array([1.0 if x else 0.0 for x in filtered_active], dtype=np.float32)))
    filtered_active_ratio_before_fallback = filtered_active_ratio

    fallback_preserve_original = False
    fallback_reason = ""
    has_input_activity = any(active)
    if has_input_activity and not any(filtered_active):
        fallback_preserve_original = True
        fallback_reason = "all_zero_after_filter"
    elif has_input_activity and filtered_active_ratio < (input_active_ratio * min_kept_active_ratio):
        fallback_preserve_original = True
        fallback_reason = "kept_active_ratio_too_low"
    elif has_input_activity and filtered_active_ratio < min_output_active_ratio:
        fallback_preserve_original = True
        fallback_reason = "output_active_ratio_too_low"

    if fallback_preserve_original:
        # Safety fallback: do not over-prune active masks.
        filtered = [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in masks_u8]
        for row in segment_rows:
            row["kept"] = True
        filtered_active_ratio = input_active_ratio

    kept_count = sum(1 for row in segment_rows if row["kept"])
    meta = {
        "min_track_length": min_track_length,
        "track_motion_threshold": track_motion_threshold,
        "motion_smooth_window": smooth_window,
        "iou_link_threshold": iou_link_threshold,
        "min_kept_active_ratio": min_kept_active_ratio,
        "min_output_active_ratio": min_output_active_ratio,
        "segments": segment_rows,
        "kept_segment_count": int(kept_count),
        "dropped_segment_count": int(len(segment_rows) - kept_count),
        "input_active_frame_ratio": input_active_ratio,
        "filtered_active_frame_ratio_before_fallback": filtered_active_ratio_before_fallback,
        "filtered_active_frame_ratio": filtered_active_ratio,
        "kept_frame_ratio": filtered_active_ratio,
        "fallback_preserve_original": bool(fallback_preserve_original),
        "fallback_reason": fallback_reason,
    }
    return filtered, meta
