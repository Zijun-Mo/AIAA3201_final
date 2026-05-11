#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_DYNAMIC_CLASSES = {
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


@dataclass
class DetectionBackendMeta:
    backend_requested: str
    backend_used: str
    fallback_used: bool


class DynamicObjectDetector:
    def __init__(
        self,
        backend_priority: list[str],
        dynamic_classes: set[str],
        yolo_model: str,
        yolo_conf: float,
        yolo_imgsz: int,
        maskrcnn_conf: float,
        device: str = "cpu",
    ) -> None:
        self.backend_priority = [str(x).strip().lower() for x in backend_priority if str(x).strip()]
        self.dynamic_classes = set(x.lower().strip() for x in dynamic_classes if str(x).strip())
        self.yolo_model = yolo_model
        self.yolo_conf = float(yolo_conf)
        self.yolo_imgsz = int(yolo_imgsz)
        self.maskrcnn_conf = float(maskrcnn_conf)
        self.device = device if device in {"cpu", "cuda"} else "cpu"

        self._yolo = None
        self._maskrcnn = None
        self._maskrcnn_categories: list[str] = []
        self._torch = None

    def _ensure_yolo(self):
        if self._yolo is not None:
            return self._yolo
        from ultralytics import YOLO

        self._yolo = YOLO(self.yolo_model)
        return self._yolo

    def _ensure_maskrcnn(self):
        if self._maskrcnn is not None:
            return self._maskrcnn, self._maskrcnn_categories, self._torch

        import torch
        from torchvision.models.detection import (
            MaskRCNN_ResNet50_FPN_V2_Weights,
            maskrcnn_resnet50_fpn_v2,
        )

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn_v2(weights=weights)
        dev = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        model = model.to(dev)
        model.eval()

        self._maskrcnn = model
        self._maskrcnn_categories = list(weights.meta.get("categories", []))
        self._torch = torch
        return self._maskrcnn, self._maskrcnn_categories, self._torch

    def _infer_yolo_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        yolo = self._ensure_yolo()
        device_arg: str | int = 0 if self.device == "cuda" else "cpu"
        results = yolo.predict(
            source=frame_bgr,
            conf=self.yolo_conf,
            imgsz=self.yolo_imgsz,
            device=device_arg,
            verbose=False,
        )
        h, w = frame_bgr.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        if not results:
            return out

        res = results[0]
        if res.boxes is None or res.masks is None or len(res.boxes) == 0:
            return out

        names = res.names if isinstance(res.names, dict) else {}
        boxes = res.boxes
        masks = res.masks.data
        count = min(len(boxes), int(masks.shape[0]))
        for i in range(count):
            cls_id = int(boxes.cls[i].item())
            class_name = str(names.get(cls_id, f"class_{cls_id}")).lower()
            if class_name not in self.dynamic_classes:
                continue
            m = (masks[i].detach().cpu().numpy() > 0.5).astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            out = np.maximum(out, m * 255)
        return out

    def _infer_maskrcnn_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        model, categories, torch = self._ensure_maskrcnn()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(next(model.parameters()).device)

        with torch.no_grad():
            output = model([tensor])[0]

        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        masks = output["masks"].detach().cpu().numpy()

        h, w = frame_bgr.shape[:2]
        out = np.zeros((h, w), dtype=np.uint8)
        for label, score, mask_logits in zip(labels, scores, masks):
            if float(score) < self.maskrcnn_conf:
                continue
            class_name = str(categories[int(label)]) if int(label) < len(categories) else f"class_{int(label)}"
            if class_name.lower() not in self.dynamic_classes:
                continue
            m = (mask_logits[0] > 0.5).astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            out = np.maximum(out, m * 255)
        return out

    def detect_mask(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, DetectionBackendMeta]:
        errors: list[str] = []
        first = self.backend_priority[0] if self.backend_priority else "unknown"
        for backend in self.backend_priority:
            try:
                if backend == "yolo":
                    mask = self._infer_yolo_mask(frame_bgr)
                elif backend == "maskrcnn":
                    mask = self._infer_maskrcnn_mask(frame_bgr)
                else:
                    errors.append(f"unsupported_backend:{backend}")
                    continue
                return mask, DetectionBackendMeta(
                    backend_requested=first,
                    backend_used=backend,
                    fallback_used=bool(backend != first),
                )
            except Exception as e:
                errors.append(f"{backend}:{e}")
                continue

        raise RuntimeError(f"dynamic-object detection failed for all backends: {errors}")


def normalize_odd_kernel(k: int) -> int:
    out = max(1, int(k))
    if out % 2 == 0:
        out += 1
    return out


def ensure_masks_aligned(masks_u8: list[np.ndarray], frame_count: int, frame_shape: tuple[int, int]) -> list[np.ndarray]:
    if frame_count <= 0:
        return []
    h, w = frame_shape
    if not masks_u8:
        return [np.zeros((h, w), dtype=np.uint8) for _ in range(frame_count)]

    out = [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in masks_u8]
    aligned: list[np.ndarray] = []
    for m in out:
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        aligned.append(m)

    if len(aligned) < frame_count:
        aligned.extend([aligned[-1].copy() for _ in range(frame_count - len(aligned))])
    elif len(aligned) > frame_count:
        aligned = aligned[:frame_count]
    return aligned


def compute_tcf_per_frame(
    frames_bgr: list[np.ndarray],
    masks_u8: list[np.ndarray],
    dilate_kernel: int,
) -> tuple[list[float], int]:
    n = len(frames_bgr)
    if n == 0:
        return [], 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (normalize_odd_kernel(dilate_kernel), normalize_odd_kernel(dilate_kernel)))
    values = [0.0]
    empty_count = 0

    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    for idx in range(n - 1):
        g0 = gray[idx]
        g1 = gray[idx + 1]

        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = g0.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        warped = cv2.remap(frames_bgr[idx], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        m0 = cv2.dilate((masks_u8[idx] > 0).astype(np.uint8) * 255, k)
        m1 = cv2.dilate((masks_u8[idx + 1] > 0).astype(np.uint8) * 255, k)
        roi = ((m0 > 0) | (m1 > 0))
        if not np.any(roi):
            values.append(0.0)
            empty_count += 1
            continue

        diff = np.abs(warped.astype(np.float32) - frames_bgr[idx + 1].astype(np.float32)) / 255.0
        values.append(float(diff[roi].mean()))

    return values, int(empty_count)


def compute_remove_quality(
    frames_bgr: list[np.ndarray],
    masks_u8: list[np.ndarray],
    tcf_dilate_kernel: int = 5,
) -> tuple[dict[str, Any], list[dict[str, float]], dict[str, Any]]:
    if not frames_bgr:
        return (
            {"TCF": 0.0, "video_frame_count": 0},
            [],
            {
                "tcf_empty_region_frame_count": 0,
                "tcf_color_space": "bgr",
            },
        )

    h, w = frames_bgr[0].shape[:2]
    aligned_masks = ensure_masks_aligned(masks_u8, frame_count=len(frames_bgr), frame_shape=(h, w))

    tcf_vals, tcf_empty = compute_tcf_per_frame(
        frames_bgr=frames_bgr,
        masks_u8=aligned_masks,
        dilate_kernel=tcf_dilate_kernel,
    )

    frame_metrics: list[dict[str, float]] = []
    for tcf in tcf_vals:
        frame_metrics.append(
            {
                "TCF": float(tcf),
            }
        )

    tcf_m = float(np.mean(np.array(tcf_vals, dtype=np.float32))) if tcf_vals else 0.0

    metrics = {
        "TCF": tcf_m,
        "video_frame_count": int(len(frames_bgr)),
    }
    notes = {
        "tcf_empty_region_frame_count": int(tcf_empty),
        "tcf_color_space": "bgr",
    }
    return metrics, frame_metrics, notes
