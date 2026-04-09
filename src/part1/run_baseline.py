#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.remove_quality import DynamicObjectDetector, compute_remove_quality
from src.common.video_io import (
    cleanup_video_only_outputs,
    encode_dataset_h264_videos,
    resolve_output_policy,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SUPPORTED_SEG_MODELS = {"yolo", "maskrcnn"}


@dataclass
class CandidateSpec:
    stage: str
    name: str
    seg_model: str
    flow_threshold: float
    dilation_kernel: int
    inpaint_method: str
    temporal_window: int

    @property
    def candidate_id(self) -> str:
        raw = f"{self.stage}_{self.name}"
        return sanitize_name(raw)


@dataclass
class DatasetPayload:
    frame_names: list[str]
    frames: list[np.ndarray]


@dataclass
class CandidateResult:
    spec: CandidateSpec
    candidate_root: Path
    eval_exp_id: str
    summary_path: Path
    aggregate: dict[str, Any]
    per_dataset: dict[str, Any]
    mask_stats: dict[str, dict[str, Any]]


def setup_logger(exp_id: str) -> tuple[logging.Logger, Path]:
    log_dir = REPO_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"phase1_{exp_id}.log"

    logger = logging.getLogger("phase1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger, log_path


def str2bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid bool value: {value}")


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_repo_path(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def set_global_seed(seed: int, logger: logging.Logger) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)

    torch_seeded = False
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        torch_seeded = True
    except Exception:
        torch_seeded = False

    logger.info("Global seed set: %d (torch_seeded=%s)", seed, str(torch_seeded).lower())
    return {"seed": seed, "torch_seeded": torch_seeded}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_dataset_cfg(config: dict) -> tuple[dict[str, dict], list[str], list[str]]:
    datasets_cfg = config.get("datasets", {}) or {}
    mandatory = datasets_cfg.get("mandatory", {}) or {}
    optional = datasets_cfg.get("optional", {}) or {}

    if not isinstance(mandatory, dict) or not isinstance(optional, dict):
        raise ValueError("datasets.mandatory and datasets.optional must be mappings")

    merged = {**mandatory, **optional}
    all_names = list(merged.keys())
    mandatory_names = list(mandatory.keys())
    return merged, all_names, mandatory_names


def resolve_dataset_names(spec: str, all_names: list[str], mandatory_names: list[str]) -> list[str]:
    spec = spec.strip().lower()
    if spec == "all":
        return all_names
    if spec == "mandatory":
        return mandatory_names

    requested = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [x for x in requested if x not in all_names]
    if unknown:
        raise ValueError(f"Unknown datasets in --datasets: {unknown}. Valid: {all_names}")
    return requested


def parse_seg_models(spec: str) -> list[str]:
    models: list[str] = []
    for token in [x.strip().lower() for x in spec.split(",") if x.strip()]:
        if token not in SUPPORTED_SEG_MODELS:
            raise ValueError(
                f"Unsupported seg model '{token}'. Supported: {sorted(SUPPORTED_SEG_MODELS)}"
            )
        if token not in models:
            models.append(token)
    if not models:
        raise ValueError("At least one segmentation model is required.")
    return models


def unique_keep_order(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def try_tqdm(iterable, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc, leave=False)
    except Exception:
        return iterable


def maybe_install_ultralytics(auto_install: bool, logger: logging.Logger) -> bool:
    if importlib.util.find_spec("ultralytics") is not None:
        return True
    if not auto_install:
        return False

    logger.warning("ultralytics not found. Attempting auto install via pip ...")
    cmd = [sys.executable, "-m", "pip", "install", "ultralytics>=8.2,<9"]
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install ultralytics automatically: %s", e)
        return False
    return importlib.util.find_spec("ultralytics") is not None


def resolve_device(runtime_cfg: dict, logger: logging.Logger) -> str:
    preferred = str(runtime_cfg.get("device", "auto")).strip().lower()
    try:
        import torch
    except ImportError:
        logger.warning("torch is unavailable, force CPU mode.")
        return "cpu"

    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("runtime.device=cuda but CUDA unavailable, fallback to CPU.")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_dataset_payload(dataset_name: str, ds_cfg: dict) -> DatasetPayload:
    frame_dir = resolve_repo_path(Path(ds_cfg.get("processed_frames_dir", "")))
    if not frame_dir:
        raise ValueError(f"Dataset '{dataset_name}' missing processed_frames_dir in config.")

    frame_paths = list_images(frame_dir)
    if not frame_paths:
        raise RuntimeError(
            f"Dataset '{dataset_name}' has no processed frames in {frame_dir}. "
            "Run preprocess first."
        )

    names: list[str] = []
    frames: list[np.ndarray] = []
    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {p}")
        names.append(p.name)
        frames.append(img)
    return DatasetPayload(frame_names=names, frames=frames)


def build_maskrcnn_segmenter(conf_thr: float, device: str):
    import torch
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model = model.to(device)
    model.eval()
    categories = weights.meta.get("categories", [])
    return model, categories, conf_thr, torch


def infer_maskrcnn_frame(
    frame_bgr: np.ndarray,
    segmenter,
    dynamic_classes: set[str],
) -> list[dict[str, Any]]:
    model, categories, conf_thr, torch = segmenter

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        output = model([tensor])[0]

    labels = output["labels"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    masks = output["masks"].detach().cpu().numpy()

    instances: list[dict[str, Any]] = []
    h, w = frame_bgr.shape[:2]
    for label, score, mask_logits in zip(labels, scores, masks):
        score_f = float(score)
        if score_f < conf_thr:
            continue
        class_name = str(categories[int(label)]) if int(label) < len(categories) else f"class_{int(label)}"
        if class_name.lower() not in dynamic_classes:
            continue
        mask = (mask_logits[0] > 0.5).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        instances.append(
            {
                "class_name": class_name,
                "score": score_f,
                "mask": mask,
                "motion": None,
            }
        )
    return instances


def build_yolo_segmenter(yolo_model_name: str):
    from ultralytics import YOLO

    return YOLO(yolo_model_name)


def infer_yolo_frame(
    frame_bgr: np.ndarray,
    segmenter,
    conf_thr: float,
    imgsz: int,
    dynamic_classes: set[str],
    device: str,
) -> list[dict[str, Any]]:
    device_arg: str | int = 0 if device == "cuda" else "cpu"
    results = segmenter.predict(
        source=frame_bgr,
        conf=conf_thr,
        imgsz=imgsz,
        device=device_arg,
        verbose=False,
    )
    if not results:
        return []

    res = results[0]
    if res.boxes is None or res.masks is None or len(res.boxes) == 0:
        return []

    names = res.names if isinstance(res.names, dict) else {}
    h, w = frame_bgr.shape[:2]

    instances: list[dict[str, Any]] = []
    boxes = res.boxes
    masks = res.masks.data
    count = min(len(boxes), int(masks.shape[0]))
    for i in range(count):
        cls_id = int(boxes.cls[i].item())
        score = float(boxes.conf[i].item())
        class_name = str(names.get(cls_id, f"class_{cls_id}"))
        if class_name.lower() not in dynamic_classes:
            continue

        mask = (masks[i].detach().cpu().numpy() > 0.5).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        instances.append(
            {
                "class_name": class_name,
                "score": score,
                "mask": mask,
                "motion": None,
            }
        )
    return instances


def run_segmentation(
    model_name: str,
    payload: DatasetPayload,
    part1_cfg: dict,
    dynamic_classes: set[str],
    device: str,
    auto_install_missing: bool,
    logger: logging.Logger,
) -> list[list[dict[str, Any]]]:
    seg_cfg = part1_cfg.get("segmentation", {}) or {}
    if model_name == "maskrcnn":
        conf_thr = float(seg_cfg.get("maskrcnn_conf_threshold", 0.5))
        segmenter = build_maskrcnn_segmenter(conf_thr=conf_thr, device=device)
    elif model_name == "yolo":
        if not maybe_install_ultralytics(auto_install=auto_install_missing, logger=logger):
            raise RuntimeError(
                "ultralytics is required for YOLO but unavailable and auto-install failed/disabled."
            )
        segmenter = build_yolo_segmenter(str(seg_cfg.get("yolo_model", "yolov8n-seg.pt")))
        conf_thr = float(seg_cfg.get("yolo_conf_threshold", 0.25))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    frame_instances: list[list[dict[str, Any]]] = []
    iterator = try_tqdm(range(len(payload.frames)), desc=f"segment:{model_name}")
    for idx in iterator:
        frame = payload.frames[idx]
        if model_name == "maskrcnn":
            instances = infer_maskrcnn_frame(
                frame_bgr=frame,
                segmenter=segmenter,
                dynamic_classes=dynamic_classes,
            )
        else:
            instances = infer_yolo_frame(
                frame_bgr=frame,
                segmenter=segmenter,
                conf_thr=conf_thr,
                imgsz=int(seg_cfg.get("yolo_imgsz", 960)),
                dynamic_classes=dynamic_classes,
                device=device,
            )
        frame_instances.append(instances)
    return frame_instances


def estimate_sparse_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    mask_u8: np.ndarray,
    flow_cfg: dict,
) -> float | None:
    mask_px = (mask_u8 > 0).astype(np.uint8) * 255
    if int(mask_px.sum()) == 0:
        return None

    points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=int(flow_cfg.get("max_corners", 200)),
        qualityLevel=float(flow_cfg.get("quality_level", 0.01)),
        minDistance=float(flow_cfg.get("min_distance", 5)),
        mask=mask_px,
        blockSize=int(flow_cfg.get("block_size", 7)),
    )
    if points is None or len(points) == 0:
        return None

    win_size_cfg = flow_cfg.get("win_size", [21, 21])
    win_size = tuple(int(x) for x in win_size_cfg)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        points,
        None,
        winSize=win_size,  # type: ignore[arg-type]
        maxLevel=int(flow_cfg.get("max_level", 3)),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(flow_cfg.get("criteria_count", 30)),
            float(flow_cfg.get("criteria_eps", 0.01)),
        ),
    )
    if p1 is None or st is None:
        return None

    valid = st.reshape(-1) == 1
    if not np.any(valid):
        return None

    delta = p1[valid] - points[valid]
    mags = np.linalg.norm(delta, axis=1)
    if mags.size == 0:
        return None

    mode = str(flow_cfg.get("aggregation", "median")).lower()
    if mode == "mean":
        return float(np.mean(mags))
    return float(np.median(mags))


def annotate_motion(instances_per_frame: list[list[dict[str, Any]]], frames: list[np.ndarray], flow_cfg: dict) -> None:
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    for idx, instances in enumerate(instances_per_frame):
        if idx == 0:
            for inst in instances:
                inst["motion"] = None
            continue
        prev_gray = gray_frames[idx - 1]
        curr_gray = gray_frames[idx]
        for inst in instances:
            inst["motion"] = estimate_sparse_motion(
                prev_gray=prev_gray,
                curr_gray=curr_gray,
                mask_u8=inst["mask"],
                flow_cfg=flow_cfg,
            )


def build_dynamic_masks(
    instances_per_frame: list[list[dict[str, Any]]],
    frame_shape_hw: tuple[int, int],
    flow_threshold: float,
    dilation_kernel: int,
    keep_unknown_motion: bool,
) -> tuple[list[np.ndarray], dict[str, float]]:
    h, w = frame_shape_hw
    masks: list[np.ndarray] = []
    coverage = []
    kept_instances = 0
    total_instances = 0

    if dilation_kernel > 1:
        k = int(dilation_kernel)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    else:
        kernel = None

    for instances in instances_per_frame:
        frame_mask = np.zeros((h, w), dtype=np.uint8)
        total_instances += len(instances)
        for inst in instances:
            motion = inst.get("motion", None)
            if motion is None:
                if not keep_unknown_motion:
                    continue
            elif float(motion) < flow_threshold:
                continue

            frame_mask = np.maximum(frame_mask, inst["mask"])
            kept_instances += 1

        if kernel is not None:
            frame_mask = cv2.dilate(frame_mask, kernel, iterations=1)

        frame_mask = (frame_mask > 0).astype(np.uint8) * 255
        masks.append(frame_mask)
        coverage.append(float((frame_mask > 0).mean()))

    stats = {
        "mean_mask_ratio": float(np.mean(np.array(coverage, dtype=np.float32))) if coverage else 0.0,
        "active_frame_ratio": float(
            np.mean(np.array([1.0 if x > 0.0 else 0.0 for x in coverage], dtype=np.float32))
        )
        if coverage
        else 0.0,
        "kept_instance_count": float(kept_instances),
        "total_instance_count": float(total_instances),
    }
    return masks, stats


def compute_mean_mask_ratio(masks_u8: list[np.ndarray]) -> float:
    if not masks_u8:
        return 0.0
    ratios = [float((m > 0).mean()) for m in masks_u8]
    return float(np.mean(np.array(ratios, dtype=np.float32)))


def compute_active_frame_ratio(masks_u8: list[np.ndarray]) -> float:
    if not masks_u8:
        return 0.0
    active = [1.0 if int((np.asarray(m) > 0).sum()) > 0 else 0.0 for m in masks_u8]
    return float(np.mean(np.array(active, dtype=np.float32)))


def normalize_odd_kernel(kernel_size: int) -> int:
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    return k


def postprocess_motion_mask(mask_u8: np.ndarray, morph_kernel: int) -> np.ndarray:
    k = normalize_odd_kernel(morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)
    return cleaned


def build_wild_fallback_masks(
    frames: list[np.ndarray],
    diff_threshold_percentile: float,
    morph_kernel: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if not frames:
        return [], {"fallback_mean_mask_ratio": 0.0, "fallback_source": "empty_frames"}

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    p = float(np.clip(diff_threshold_percentile, 50.0, 99.9))

    def build_from_diff_maps(diff_maps: list[np.ndarray], source_name: str) -> tuple[list[np.ndarray], dict[str, Any]]:
        masks: list[np.ndarray] = []
        for diff in diff_maps:
            diff_f = diff.astype(np.float32)
            thr = float(np.percentile(diff_f, p))
            if thr <= 0:
                thr = float(diff_f.mean() + 0.5 * diff_f.std())
            bin_mask = (diff_f >= thr).astype(np.uint8) * 255
            masks.append(postprocess_motion_mask(bin_mask, morph_kernel))
        return masks, {
            "fallback_source": source_name,
            "fallback_mean_mask_ratio": compute_mean_mask_ratio(masks),
            "fallback_active_frame_ratio": compute_active_frame_ratio(masks),
        }

    neighbor_diffs: list[np.ndarray] = []
    n = len(gray_frames)
    for idx in range(n):
        if n == 1:
            neighbor = gray_frames[idx]
        elif idx == 0:
            neighbor = gray_frames[1]
        else:
            neighbor = gray_frames[idx - 1]
        neighbor_diffs.append(cv2.absdiff(gray_frames[idx], neighbor))

    masks, stats = build_from_diff_maps(neighbor_diffs, "adjacent_frame_diff")
    if stats["fallback_mean_mask_ratio"] > 0.0:
        return masks, stats

    stack = np.stack(gray_frames, axis=0).astype(np.float32)
    bg = np.median(stack, axis=0).astype(np.uint8)
    bg_diffs = [cv2.absdiff(g, bg) for g in gray_frames]
    masks, stats = build_from_diff_maps(bg_diffs, "median_background_diff")
    if stats["fallback_mean_mask_ratio"] > 0.0:
        return masks, stats

    masks = [np.zeros_like(gray_frames[0], dtype=np.uint8) for _ in gray_frames]
    for idx, diff in enumerate(neighbor_diffs):
        y, x = np.unravel_index(int(np.argmax(diff)), diff.shape)
        cv2.circle(masks[idx], (int(x), int(y)), radius=3, color=255, thickness=-1)
        masks[idx] = postprocess_motion_mask(masks[idx], morph_kernel)

    stats = {
        "fallback_source": "max_diff_anchor",
        "fallback_mean_mask_ratio": compute_mean_mask_ratio(masks),
        "fallback_active_frame_ratio": compute_active_frame_ratio(masks),
    }
    return masks, stats


def classify_failure_case(
    dataset: str,
    ros: float,
    tcf: float,
    bes: float,
    wild_fallback_applied: bool,
) -> str:
    if dataset == "wild" and wild_fallback_applied:
        return "residual_object_fallback_recovery"
    if ros > 0.05:
        return "residual_object"
    if tcf > 0.12:
        return "temporal_flicker"
    if bes > 0.15:
        return "boundary_artifact"
    return "minor_artifact"


def temporal_borrow_fill(
    frame_idx: int,
    frames: list[np.ndarray],
    masks_bool: list[np.ndarray],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    current = frames[frame_idx]
    remaining = masks_bool[frame_idx].copy()
    filled = current.copy()

    if window <= 0:
        return filled, remaining

    frame_count = len(frames)
    for step in range(1, window + 1):
        for neighbor in (frame_idx - step, frame_idx + step):
            if neighbor < 0 or neighbor >= frame_count:
                continue
            donor_valid = remaining & (~masks_bool[neighbor])
            if np.any(donor_valid):
                filled[donor_valid] = frames[neighbor][donor_valid]
                remaining[donor_valid] = False
        if not np.any(remaining):
            break
    return filled, remaining


def restore_frames(
    frames: list[np.ndarray],
    masks_u8: list[np.ndarray],
    inpaint_method: str,
    inpaint_radius: float,
    temporal_window: int,
) -> list[np.ndarray]:
    restored: list[np.ndarray] = []
    masks_bool = [m > 0 for m in masks_u8]
    method = cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS

    for idx in range(len(frames)):
        filled, remaining = temporal_borrow_fill(
            frame_idx=idx,
            frames=frames,
            masks_bool=masks_bool,
            window=temporal_window,
        )
        if np.any(remaining):
            inpaint_mask = remaining.astype(np.uint8) * 255
            restored_frame = cv2.inpaint(filled, inpaint_mask, float(inpaint_radius), method)
        else:
            restored_frame = filled
        restored.append(restored_frame)
    return restored


def write_dataset_outputs(
    out_root: Path,
    dataset_name: str,
    frame_names: list[str],
    restored_frames: list[np.ndarray],
    masks_u8: list[np.ndarray],
    target_fps: float,
    save_mp4: bool,
    output_policy: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    policy = resolve_output_policy(output_policy)
    ds_root = out_root / dataset_name
    frame_dir = ds_root / "frames"
    mask_dir = ds_root / "masks"
    frame_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for name, frame, mask in zip(frame_names, restored_frames, masks_u8):
        cv2.imwrite(str(frame_dir / name), frame)
        cv2.imwrite(str(mask_dir / name), mask)

    if save_mp4 and restored_frames:
        h, w = restored_frames[0].shape[:2]
        mp4_path = ds_root / "video.mp4"
        writer = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(target_fps),
            (w, h),
        )
        for frame in restored_frames:
            writer.write(frame)
        writer.release()

    restored_video_path, mask_video_path = encode_dataset_h264_videos(
        dataset_root=ds_root,
        restored_frames_bgr=restored_frames,
        masks_u8=masks_u8,
        fps=float(target_fps) if float(target_fps) > 0 else 24.0,
        output_policy=policy,
    )
    return {
        "frame_dir": str(frame_dir),
        "mask_dir": str(mask_dir),
        "restored_video": str(restored_video_path) if restored_video_path is not None else None,
        "mask_video": str(mask_video_path) if mask_video_path is not None else None,
    }


def run_evaluation(
    config_path: Path,
    datasets: list[str],
    pred_root: Path,
    gt_root: Path,
    eval_exp_id: str,
    allow_missing_gt: bool,
    save_visualization: bool,
    logger: logging.Logger,
) -> tuple[Path, dict[str, Any]]:
    cmd = [
        sys.executable,
        "src/common/evaluate_experiment.py",
        "--config",
        str(config_path),
        "--exp-id",
        eval_exp_id,
        "--datasets",
        ",".join(datasets),
        "--pred-root",
        str(pred_root),
        "--gt-root",
        str(gt_root),
        "--allow-missing-gt",
        "true" if allow_missing_gt else "false",
        "--save-visualization",
        "true" if save_visualization else "false",
    ]
    logger.info("Evaluate: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    summary_path = REPO_ROOT / "outputs" / "metrics" / eval_exp_id / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Evaluation summary missing: {summary_path}")
    return summary_path, read_json(summary_path)


def metric_or_neg_inf(agg: dict[str, Any], key: str) -> float:
    value = agg.get(key, None)
    if value is None:
        return float("-inf")
    return float(value)


def stage_score(stage: str, agg: dict[str, Any], mean_mask_ratio: float) -> tuple[float, float, float, float]:
    jm = metric_or_neg_inf(agg, "JM")
    jr = metric_or_neg_inf(agg, "JR")
    q_remove = metric_or_neg_inf(agg, "Q_REMOVE")

    # A1-A3 prioritize mask quality first; A4-A5 prioritize reconstruction quality.
    if stage in {"A1", "A2", "A3"}:
        return (jm, jr, q_remove, -abs(mean_mask_ratio - 0.1))
    return (q_remove, jm, jr, -abs(mean_mask_ratio - 0.1))


def parse_selection_coverage_constraints(selection_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    mean_cfg = selection_cfg.get("min_mean_mask_ratio_by_dataset", {}) or {}
    active_cfg = selection_cfg.get("min_active_frame_ratio_by_dataset", {}) or {}
    constraints: dict[str, dict[str, float]] = {}

    if not isinstance(mean_cfg, dict):
        mean_cfg = {}
    if not isinstance(active_cfg, dict):
        active_cfg = {}

    for ds in sorted(set([str(k).strip() for k in mean_cfg.keys()] + [str(k).strip() for k in active_cfg.keys()])):
        if not ds:
            continue
        req: dict[str, float] = {}
        if ds in mean_cfg:
            req["min_mean_mask_ratio"] = float(mean_cfg.get(ds, 0.0))
        if ds in active_cfg:
            req["min_active_frame_ratio"] = float(active_cfg.get(ds, 0.0))
        constraints[ds] = req
    return constraints


def candidate_meets_coverage_constraints(
    entry: CandidateResult,
    coverage_constraints: dict[str, dict[str, float]],
) -> bool:
    if not coverage_constraints:
        return True
    for ds, req in coverage_constraints.items():
        ds_stats = entry.mask_stats.get(ds, {}) or {}
        if not ds_stats:
            continue
        mean_ratio = float(ds_stats.get("mean_mask_ratio", 0.0))
        active_ratio = float(
            ds_stats.get(
                "active_frame_ratio",
                1.0 if mean_ratio > 0.0 else 0.0,
            )
        )
        if "min_mean_mask_ratio" in req and mean_ratio < float(req["min_mean_mask_ratio"]):
            return False
        if "min_active_frame_ratio" in req and active_ratio < float(req["min_active_frame_ratio"]):
            return False
    return True


def select_best(
    stage: str,
    entries: list[CandidateResult],
    coverage_constraints: dict[str, dict[str, float]] | None = None,
    enforce_if_candidate_available: bool = True,
    logger: logging.Logger | None = None,
) -> CandidateResult:
    if not entries:
        raise ValueError(f"No candidate results to select from in stage {stage}")

    pool = entries
    constraints = coverage_constraints or {}
    if constraints:
        eligible = [e for e in entries if candidate_meets_coverage_constraints(e, constraints)]
        if eligible:
            pool = eligible
        elif enforce_if_candidate_available:
            if logger is not None:
                logger.warning(
                    "[%s] No candidate met coverage constraints=%s, fallback to all candidates.",
                    stage,
                    constraints,
                )
        else:
            raise ValueError(f"No candidate meets coverage constraints in stage {stage}: {constraints}")

    def score(entry: CandidateResult) -> tuple[float, float, float, float]:
        ratios = [v.get("mean_mask_ratio", 0.0) for v in entry.mask_stats.values()]
        mean_ratio = float(np.mean(np.array(ratios, dtype=np.float32))) if ratios else 0.0
        return stage_score(stage, entry.aggregate, mean_ratio)

    return max(pool, key=score)


def copy_a_best(best_candidate_root: Path, final_root: Path, datasets: list[str]) -> None:
    for ds in datasets:
        src = best_candidate_root / ds
        if not src.exists():
            raise RuntimeError(f"A-best candidate missing dataset output: {src}")
        dst = final_root / ds
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def build_failure_case_index(
    pred_root: Path,
    gt_root: Path,
    datasets: list[str],
    out_dir: Path,
    detector: DynamicObjectDetector,
    quality_weights: dict[str, float],
    tcf_dilate_kernel: int,
    bes_dilate_kernel: int,
    bes_erode_kernel: int,
    bes_sobel_ksize: int,
    fallback_applied_map: dict[str, bool] | None = None,
    top_k: int = 3,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows_explained: list[dict[str, Any]] = []
    fallback_applied_map = fallback_applied_map or {}

    for ds in datasets:
        pred_dir = pred_root / ds / "frames"
        mask_dir = pred_root / ds / "masks"
        pred_paths = list_images(pred_dir)
        mask_map = {p.name: p for p in list_images(mask_dir)}
        if not pred_paths:
            continue

        frames: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        frame_names: list[str] = []
        for pp in pred_paths:
            pi = cv2.imread(str(pp), cv2.IMREAD_COLOR)
            if pi is None:
                continue
            mp = mask_map.get(pp.name)
            if mp is None:
                pm = np.zeros(pi.shape[:2], dtype=np.uint8)
            else:
                pm = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if pm is None:
                    pm = np.zeros(pi.shape[:2], dtype=np.uint8)
                elif pm.shape != pi.shape[:2]:
                    pm = cv2.resize(pm, (pi.shape[1], pi.shape[0]), interpolation=cv2.INTER_NEAREST)
            frames.append(pi)
            masks.append(pm)
            frame_names.append(pp.name)

        if not frames:
            continue

        _, per_frame_metrics, _ = compute_remove_quality(
            frames_bgr=frames,
            masks_u8=masks,
            detector=detector,
            quality_weights=quality_weights,
            tcf_dilate_kernel=tcf_dilate_kernel,
            bes_dilate_kernel=bes_dilate_kernel,
            bes_erode_kernel=bes_erode_kernel,
            bes_sobel_ksize=bes_sobel_ksize,
        )
        scored = sorted(
            [(float(m.get("Q_REMOVE", 1.0)), idx) for idx, m in enumerate(per_frame_metrics)],
            key=lambda x: x[0],
        )

        for rank, (q_remove, idx) in enumerate(scored[:top_k], start=1):
            frame = frames[idx]
            name = frame_names[idx]
            out_img = out_dir / f"{ds}_{Path(name).stem}_rank{rank}_qremove{q_remove:.4f}.png"
            cv2.imwrite(str(out_img), frame)
            ros = float(per_frame_metrics[idx].get("ROS", 0.0))
            tcf = float(per_frame_metrics[idx].get("TCF", 0.0))
            bes = float(per_frame_metrics[idx].get("BES", 0.0))
            explanation = classify_failure_case(
                dataset=ds,
                ros=ros,
                tcf=tcf,
                bes=bes,
                wild_fallback_applied=bool(fallback_applied_map.get(ds, False)),
            )

            rows.append(
                {
                    "dataset": ds,
                    "frame": name,
                    "rank": rank,
                    "q_remove": q_remove,
                    "compare_image": str(out_img),
                }
            )
            rows_explained.append(
                {
                    "dataset": ds,
                    "frame": name,
                    "rank": rank,
                    "q_remove": q_remove,
                    "ros": ros,
                    "tcf": tcf,
                    "bes": bes,
                    "explanation": explanation,
                    "compare_image": str(out_img),
                }
            )

    csv_path = out_dir / "failure_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "frame", "rank", "q_remove", "compare_image"])
        writer.writeheader()
        writer.writerows(rows)

    explained_csv_path = out_dir / "failure_cases_explained.csv"
    with explained_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "frame",
                "rank",
                "q_remove",
                "ros",
                "tcf",
                "bes",
                "explanation",
                "compare_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_explained)
    return csv_path, explained_csv_path


def run_candidate(
    spec: CandidateSpec,
    datasets: list[str],
    dataset_payloads: dict[str, DatasetPayload],
    seg_cache: dict[tuple[str, str], list[list[dict[str, Any]]]],
    mask_cache: dict[tuple[str, str, float, int], tuple[list[np.ndarray], dict[str, float]]],
    out_root: Path,
    part1_cfg: dict,
    target_fps: float,
    wild_fallback_mask: bool,
    output_policy: dict[str, Any],
    logger: logging.Logger,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, bool]]:
    candidate_root = out_root / spec.stage / spec.candidate_id
    candidate_root.mkdir(parents=True, exist_ok=True)
    inpaint_cfg = part1_cfg.get("inpaint", {}) or {}
    fallback_cfg = part1_cfg.get("fallback", {}) or {}
    keep_unknown_motion = bool(part1_cfg.get("keep_unknown_motion_as_dynamic", True))
    inpaint_radius = float(inpaint_cfg.get("radius", 3.0))
    save_mp4 = bool(part1_cfg.get("save_mp4", False))
    fallback_trigger_ratio = float(fallback_cfg.get("min_area_ratio", 0.0))
    fallback_diff_percentile = float(fallback_cfg.get("diff_threshold_percentile", 97.5))
    fallback_morph_kernel = int(fallback_cfg.get("morph_kernel", 5))

    mask_stats_map: dict[str, dict[str, Any]] = {}
    fallback_applied_map: dict[str, bool] = {}
    for ds in datasets:
        payload = dataset_payloads[ds]
        seg_key = (spec.seg_model, ds)
        if seg_key not in seg_cache:
            raise RuntimeError(f"Segmentation cache missing for key={seg_key}")

        mask_key = (ds, spec.seg_model, float(spec.flow_threshold), int(spec.dilation_kernel))
        if mask_key in mask_cache:
            masks_cached, stats_cached = mask_cache[mask_key]
            masks_u8 = [m.copy() for m in masks_cached]
            mask_stats = dict(stats_cached)
        else:
            masks_u8, mask_stats = build_dynamic_masks(
                instances_per_frame=seg_cache[seg_key],
                frame_shape_hw=payload.frames[0].shape[:2],
                flow_threshold=spec.flow_threshold,
                dilation_kernel=spec.dilation_kernel,
                keep_unknown_motion=keep_unknown_motion,
            )
            mask_cache[mask_key] = ([m.copy() for m in masks_u8], dict(mask_stats))

        fallback_applied = False
        if wild_fallback_mask and ds == "wild":
            current_ratio = float(mask_stats.get("mean_mask_ratio", 0.0))
            if current_ratio <= fallback_trigger_ratio:
                fb_masks, fb_stats = build_wild_fallback_masks(
                    frames=payload.frames,
                    diff_threshold_percentile=fallback_diff_percentile,
                    morph_kernel=fallback_morph_kernel,
                )
                fallback_ratio = float(fb_stats.get("fallback_mean_mask_ratio", 0.0))
                if fallback_ratio > current_ratio:
                    masks_u8 = fb_masks
                    fallback_applied = True
                    mask_stats["original_mean_mask_ratio"] = current_ratio
                    mask_stats["mean_mask_ratio"] = fallback_ratio
                    mask_stats["active_frame_ratio"] = float(
                        fb_stats.get("fallback_active_frame_ratio", compute_active_frame_ratio(fb_masks))
                    )
                    mask_stats["fallback_applied"] = True
                    mask_stats["fallback_source"] = str(fb_stats.get("fallback_source", "unknown"))
                    mask_stats["fallback_trigger_ratio"] = fallback_trigger_ratio
                    logger.warning(
                        "Applied wild fallback masks | stage=%s candidate=%s ratio %.6f -> %.6f",
                        spec.stage,
                        spec.name,
                        current_ratio,
                        fallback_ratio,
                    )

        if not fallback_applied:
            mask_stats["fallback_applied"] = False
        if "active_frame_ratio" not in mask_stats:
            mask_stats["active_frame_ratio"] = compute_active_frame_ratio(masks_u8)

        restored_frames = restore_frames(
            frames=payload.frames,
            masks_u8=masks_u8,
            inpaint_method=spec.inpaint_method,
            inpaint_radius=inpaint_radius,
            temporal_window=spec.temporal_window,
        )
        write_dataset_outputs(
            out_root=candidate_root,
            dataset_name=ds,
            frame_names=payload.frame_names,
            restored_frames=restored_frames,
            masks_u8=masks_u8,
            target_fps=target_fps,
            save_mp4=save_mp4,
            output_policy=output_policy,
        )
        mask_stats_map[ds] = mask_stats
        fallback_applied_map[ds] = bool(mask_stats.get("fallback_applied", False))

    write_json(
        candidate_root / "candidate_config.json",
        {
            "candidate": asdict(spec),
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "mask_stats": mask_stats_map,
            "fallback_applied": fallback_applied_map,
        },
    )
    logger.info(
        "Candidate complete: stage=%s name=%s root=%s",
        spec.stage,
        spec.name,
        candidate_root,
    )
    return candidate_root, mask_stats_map, fallback_applied_map


def write_ablation_outputs(
    exp_metrics_dir: Path,
    all_results: list[CandidateResult],
    stage_best_map: dict[str, CandidateResult],
    final_best: CandidateResult,
) -> None:
    exp_metrics_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for r in all_results:
        ratios = [v.get("mean_mask_ratio", 0.0) for v in r.mask_stats.values()]
        mean_ratio = float(np.mean(np.array(ratios, dtype=np.float32))) if ratios else 0.0
        active_ratios = [
            v.get("active_frame_ratio", 1.0 if float(v.get("mean_mask_ratio", 0.0)) > 0 else 0.0)
            for v in r.mask_stats.values()
        ]
        active_ratio = float(np.mean(np.array(active_ratios, dtype=np.float32))) if active_ratios else 0.0
        rows.append(
            {
                "stage": r.spec.stage,
                "candidate": r.spec.name,
                "seg_model": r.spec.seg_model,
                "flow_threshold": r.spec.flow_threshold,
                "dilation_kernel": r.spec.dilation_kernel,
                "inpaint_method": r.spec.inpaint_method,
                "temporal_window": r.spec.temporal_window,
                "JM": r.aggregate.get("JM"),
                "JR": r.aggregate.get("JR"),
                "ROS": r.aggregate.get("ROS"),
                "TCF": r.aggregate.get("TCF"),
                "BES": r.aggregate.get("BES"),
                "Q_REMOVE": r.aggregate.get("Q_REMOVE"),
                "mean_mask_ratio": mean_ratio,
                "active_frame_ratio": active_ratio,
                "pred_root": str(r.candidate_root),
                "eval_exp_id": r.eval_exp_id,
                "is_stage_best": int(stage_best_map.get(r.spec.stage) is r),
                "is_final_best": int(final_best is r),
            }
        )

    csv_path = exp_metrics_dir / "phase1_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "candidate",
                "seg_model",
                "flow_threshold",
                "dilation_kernel",
                "inpaint_method",
                "temporal_window",
                "JM",
                "JR",
                "ROS",
                "TCF",
                "BES",
                "Q_REMOVE",
                "mean_mask_ratio",
                "active_frame_ratio",
                "pred_root",
                "eval_exp_id",
                "is_stage_best",
                "is_final_best",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        exp_metrics_dir / "phase1_selection.json",
        {
            "stage_best": {k: asdict(v.spec) for k, v in stage_best_map.items()},
            "final_best": asdict(final_best.spec),
            "candidate_count": len(all_results),
        },
    )


def write_acceptance_report(
    report_path: Path,
    exp_id: str,
    summary: dict[str, Any],
    stage_best: dict[str, CandidateResult],
    failure_explained_csv: Path,
    fallback_applied_final: dict[str, bool],
    seed: int,
) -> None:
    aggregate = summary.get("aggregate", {}) or {}
    datasets = summary.get("datasets", {}) or {}

    lines: list[str] = []
    lines.append(f"# Phase 1 Acceptance Report: `{exp_id}`")
    lines.append("")
    lines.append("## Final Aggregate")
    lines.append("")
    lines.append("| JM | JR | ROS | TCF | BES | Q_REMOVE |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| {aggregate.get('JM')} | {aggregate.get('JR')} | {aggregate.get('ROS')} | {aggregate.get('TCF')} | {aggregate.get('BES')} | {aggregate.get('Q_REMOVE')} |"
    )
    lines.append("")
    lines.append("## Stage Best")
    lines.append("")
    lines.append("| Stage | Candidate | Seg | Flow | Dilation | Inpaint | Temporal |")
    lines.append("| --- | --- | --- | ---: | ---: | --- | ---: |")
    for stage in ["A1", "A2", "A3", "A4", "A5"]:
        result = stage_best.get(stage)
        if result is None:
            continue
        spec = result.spec
        lines.append(
            f"| {stage} | {spec.name} | {spec.seg_model} | {spec.flow_threshold} | {spec.dilation_kernel} | {spec.inpaint_method} | {spec.temporal_window} |"
        )
    lines.append("")
    lines.append("## Per-Dataset Metrics")
    lines.append("")
    lines.append("| Dataset | JM | JR | ROS | TCF | BES | Q_REMOVE | Fallback Applied |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for ds_name, ds_payload in datasets.items():
        metrics = ds_payload.get("metrics", {}) or {}
        lines.append(
            f"| {ds_name} | {metrics.get('JM')} | {metrics.get('JR')} | {metrics.get('ROS')} | {metrics.get('TCF')} | {metrics.get('BES')} | {metrics.get('Q_REMOVE')} | {bool(fallback_applied_final.get(ds_name, False))} |"
        )
    lines.append("")
    lines.append("## Acceptance Checks")
    lines.append("")
    lines.append(f"- Seed set and logged: `{seed}`")
    lines.append(f"- Failure case explanations: `{failure_explained_csv}`")
    lines.append("- Wild note: current wild source is auto-generated content and should be interpreted cautiously.")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 baseline (A1-A5): YOLO/MaskRCNN -> flow filter -> inpaint."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--datasets", type=str, default="mandatory", help="mandatory | all | csv")
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/videos"))

    parser.add_argument("--seg-models", type=str, default="yolo,maskrcnn")
    parser.add_argument("--flow-threshold", type=float, default=None)
    parser.add_argument("--dilation-kernel", type=int, default=None)
    parser.add_argument("--inpaint-method", type=str, default=None, choices=["telea", "ns"])
    parser.add_argument("--temporal-window", type=int, default=None)
    parser.add_argument("--auto-install-missing", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Global seed. Default from config project.seed")
    parser.add_argument(
        "--wild-fallback-mask",
        type=str,
        default=None,
        help="Enable fallback motion masks for wild when predicted mask is near-zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_id = args.exp_id or f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger, log_path = setup_logger(exp_id)
    logger.info("Phase 1 baseline start | exp_id=%s", exp_id)

    config_path = resolve_repo_path(Path(args.config))
    config = read_yaml(config_path)
    part1_cfg = config.get("part1", {}) or {}
    output_policy = resolve_output_policy(config)

    ds_map, all_names, mandatory_names = collect_dataset_cfg(config)
    selected_datasets = resolve_dataset_names(args.datasets, all_names, mandatory_names)
    logger.info("Datasets: %s", selected_datasets)

    runtime_cfg = part1_cfg.get("runtime", {}) or {}
    fallback_cfg = part1_cfg.get("fallback", {}) or {}
    auto_install_missing = str2bool(
        args.auto_install_missing,
        default=bool(runtime_cfg.get("auto_install_missing", True)),
    )
    wild_fallback_mask = str2bool(
        args.wild_fallback_mask,
        default=bool(fallback_cfg.get("enable_wild_zero_mask_fallback", True)),
    )
    seed_value = int(args.seed) if args.seed is not None else int(config.get("project", {}).get("seed", 42))
    seed_meta = set_global_seed(seed_value, logger)
    device = resolve_device(runtime_cfg=runtime_cfg, logger=logger)

    selected_models = parse_seg_models(args.seg_models)
    if "yolo" in selected_models and not maybe_install_ultralytics(auto_install_missing, logger):
        logger.warning("YOLO unavailable; fallback to remaining segmentation models.")
        selected_models = [m for m in selected_models if m != "yolo"]
    if not selected_models:
        raise RuntimeError("No segmentation model available after dependency checks.")
    logger.info("Segmentation models: %s | device=%s", selected_models, device)
    logger.info("Wild fallback mask enabled=%s", str(wild_fallback_mask).lower())
    logger.info(
        "Output policy: video_only=%s write_h264_videos=%s auto_cleanup_intermediates=%s",
        bool(output_policy.get("video_only", True)),
        bool(output_policy.get("write_h264_videos", True)),
        bool(output_policy.get("auto_cleanup_intermediates", True)),
    )

    dynamic_classes = set(
        x.strip().lower()
        for x in part1_cfg.get(
            "dynamic_classes",
            [
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
            ],
        )
        if str(x).strip()
    )
    flow_cfg = part1_cfg.get("flow", {}) or {}

    target_fps = float(config.get("preprocess", {}).get("target_fps", 24.0))
    gt_root = resolve_repo_path(Path(config.get("paths", {}).get("gt_data_dir", "data/gt")))
    eval_cfg = config.get("evaluation", {}) or {}
    eval_selection_cfg = eval_cfg.get("selection", {}) or {}
    selection_coverage_constraints = parse_selection_coverage_constraints(eval_selection_cfg)
    enforce_selection_coverage = bool(eval_selection_cfg.get("enforce_if_candidate_available", True))
    allow_missing_gt = bool(eval_cfg.get("allow_missing_gt", True))
    save_visualization = bool(eval_cfg.get("save_visualization", True))
    quality_weights_cfg = eval_cfg.get("quality_weights", {}) or {}
    quality_weights = {
        "ros": float(quality_weights_cfg.get("ros", 0.5)),
        "tcf": float(quality_weights_cfg.get("tcf", 0.3)),
        "bes": float(quality_weights_cfg.get("bes", 0.2)),
    }
    ros_cfg = eval_cfg.get("ros", {}) or {}
    tcf_cfg = eval_cfg.get("tcf", {}) or {}
    bes_cfg = eval_cfg.get("bes", {}) or {}
    if selection_coverage_constraints:
        logger.info(
            "Selection coverage constraints=%s enforce_if_candidate_available=%s",
            selection_coverage_constraints,
            str(enforce_selection_coverage).lower(),
        )
    seg_cfg = part1_cfg.get("segmentation", {}) or {}
    detector = DynamicObjectDetector(
        backend_priority=[str(x).strip().lower() for x in (ros_cfg.get("backend_priority", ["yolo", "maskrcnn"]) or []) if str(x).strip()],
        dynamic_classes=dynamic_classes,
        yolo_model=str(seg_cfg.get("yolo_model", "yolov8n-seg.pt")),
        yolo_conf=float(seg_cfg.get("yolo_conf_threshold", 0.25)),
        yolo_imgsz=int(seg_cfg.get("yolo_imgsz", 960)),
        maskrcnn_conf=float(seg_cfg.get("maskrcnn_conf_threshold", 0.5)),
        device=device,
    )

    pred_root_base = resolve_repo_path(Path(args.pred_root))
    exp_pred_root = pred_root_base / exp_id
    candidate_root = exp_pred_root / "_candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)

    dataset_payloads: dict[str, DatasetPayload] = {}
    for ds in selected_datasets:
        dataset_payloads[ds] = load_dataset_payload(ds, ds_map[ds])
        logger.info("Loaded %s frames for %s", len(dataset_payloads[ds].frames), ds)

    seg_cache: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    for model_name in selected_models:
        for ds in selected_datasets:
            logger.info("Segmentation start: model=%s dataset=%s", model_name, ds)
            inst = run_segmentation(
                model_name=model_name,
                payload=dataset_payloads[ds],
                part1_cfg=part1_cfg,
                dynamic_classes=dynamic_classes,
                device=device,
                auto_install_missing=auto_install_missing,
                logger=logger,
            )
            annotate_motion(instances_per_frame=inst, frames=dataset_payloads[ds].frames, flow_cfg=flow_cfg)
            seg_cache[(model_name, ds)] = inst
            logger.info("Segmentation cached: model=%s dataset=%s", model_name, ds)

    defaults_cfg = part1_cfg.get("defaults", {}) or {}
    grids_cfg = part1_cfg.get("grids", {}) or {}

    base_flow = float(args.flow_threshold) if args.flow_threshold is not None else float(
        defaults_cfg.get("flow_threshold", 1.2)
    )
    base_dilation = int(args.dilation_kernel) if args.dilation_kernel is not None else int(
        defaults_cfg.get("dilation_kernel", 7)
    )
    base_inpaint = args.inpaint_method or str(defaults_cfg.get("inpaint_method", "telea")).lower()
    base_temporal = int(args.temporal_window) if args.temporal_window is not None else int(
        defaults_cfg.get("temporal_window", 1)
    )

    flow_grid = [base_flow] if args.flow_threshold is not None else [
        float(x) for x in grids_cfg.get("flow_thresholds", [0.8, 1.2, 1.6])
    ]
    dilation_grid = [base_dilation] if args.dilation_kernel is not None else [
        int(x) for x in grids_cfg.get("dilation_kernels", [3, 7, 11])
    ]
    inpaint_grid = [base_inpaint] if args.inpaint_method is not None else [
        str(x).lower() for x in grids_cfg.get("inpaint_methods", ["telea", "ns"])
    ]
    temporal_grid = [base_temporal] if args.temporal_window is not None else [
        int(x) for x in grids_cfg.get("temporal_windows", [0, 1, 2])
    ]

    flow_grid = unique_keep_order(flow_grid)
    dilation_grid = unique_keep_order(dilation_grid)
    inpaint_grid = unique_keep_order(inpaint_grid)
    temporal_grid = unique_keep_order(temporal_grid)

    # Stage candidate definitions (A1 -> A5)
    a1_candidates = [
        CandidateSpec(
            stage="A1",
            name=model,
            seg_model=model,
            flow_threshold=base_flow,
            dilation_kernel=base_dilation,
            inpaint_method=base_inpaint,
            temporal_window=base_temporal,
        )
        for model in selected_models
    ]

    # Run stages sequentially and lock best params stage-by-stage.
    all_results: list[CandidateResult] = []
    stage_best: dict[str, CandidateResult] = {}
    mask_cache: dict[tuple[str, str, float, int], tuple[list[np.ndarray], dict[str, float]]] = {}

    def execute_stage(stage_name: str, specs: list[CandidateSpec]) -> CandidateResult:
        logger.info("=== %s start (%d candidates) ===", stage_name, len(specs))
        stage_results: list[CandidateResult] = []
        for spec in specs:
            pred_root, mask_stats, _fallback_map = run_candidate(
                spec=spec,
                datasets=selected_datasets,
                dataset_payloads=dataset_payloads,
                seg_cache=seg_cache,
                mask_cache=mask_cache,
                out_root=candidate_root,
                part1_cfg=part1_cfg,
                target_fps=target_fps,
                wild_fallback_mask=wild_fallback_mask,
                output_policy=output_policy,
                logger=logger,
            )
            eval_exp_id = f"{exp_id}__{spec.stage}__{spec.candidate_id}"
            summary_path, summary = run_evaluation(
                config_path=config_path,
                datasets=selected_datasets,
                pred_root=pred_root,
                gt_root=gt_root,
                eval_exp_id=eval_exp_id,
                allow_missing_gt=allow_missing_gt,
                save_visualization=save_visualization,
                logger=logger,
            )
            result = CandidateResult(
                spec=spec,
                candidate_root=pred_root,
                eval_exp_id=eval_exp_id,
                summary_path=summary_path,
                aggregate=summary.get("aggregate", {}),
                per_dataset=summary.get("datasets", {}),
                mask_stats=mask_stats,
            )
            stage_results.append(result)
            all_results.append(result)
            logger.info(
                "[%s] candidate=%s -> JM=%.4f JR=%.4f ROS=%.4f TCF=%.4f BES=%.4f Q_REMOVE=%.4f",
                stage_name,
                spec.name,
                metric_or_neg_inf(result.aggregate, "JM"),
                metric_or_neg_inf(result.aggregate, "JR"),
                metric_or_neg_inf(result.aggregate, "ROS"),
                metric_or_neg_inf(result.aggregate, "TCF"),
                metric_or_neg_inf(result.aggregate, "BES"),
                metric_or_neg_inf(result.aggregate, "Q_REMOVE"),
            )

        best = select_best(
            stage_name,
            stage_results,
            coverage_constraints=selection_coverage_constraints,
            enforce_if_candidate_available=enforce_selection_coverage,
            logger=logger,
        )
        stage_best[stage_name] = best
        logger.info(
            "=== %s best: %s | seg=%s flow=%.3f dil=%d inpaint=%s temporal=%d ===",
            stage_name,
            best.spec.name,
            best.spec.seg_model,
            best.spec.flow_threshold,
            best.spec.dilation_kernel,
            best.spec.inpaint_method,
            best.spec.temporal_window,
        )
        return best

    best_a1 = execute_stage("A1", a1_candidates)

    a2_candidates = [
        CandidateSpec(
            stage="A2",
            name=f"flow_{thr:g}",
            seg_model=best_a1.spec.seg_model,
            flow_threshold=float(thr),
            dilation_kernel=best_a1.spec.dilation_kernel,
            inpaint_method=best_a1.spec.inpaint_method,
            temporal_window=best_a1.spec.temporal_window,
        )
        for thr in flow_grid
    ]
    best_a2 = execute_stage("A2", a2_candidates)

    a3_candidates = [
        CandidateSpec(
            stage="A3",
            name=f"dilate_{k}",
            seg_model=best_a2.spec.seg_model,
            flow_threshold=best_a2.spec.flow_threshold,
            dilation_kernel=int(k),
            inpaint_method=best_a2.spec.inpaint_method,
            temporal_window=best_a2.spec.temporal_window,
        )
        for k in dilation_grid
    ]
    best_a3 = execute_stage("A3", a3_candidates)

    a4_candidates = [
        CandidateSpec(
            stage="A4",
            name=f"inpaint_{method}",
            seg_model=best_a3.spec.seg_model,
            flow_threshold=best_a3.spec.flow_threshold,
            dilation_kernel=best_a3.spec.dilation_kernel,
            inpaint_method=method,
            temporal_window=best_a3.spec.temporal_window,
        )
        for method in inpaint_grid
    ]
    best_a4 = execute_stage("A4", a4_candidates)

    a5_candidates = [
        CandidateSpec(
            stage="A5",
            name=f"temporal_{w}",
            seg_model=best_a4.spec.seg_model,
            flow_threshold=best_a4.spec.flow_threshold,
            dilation_kernel=best_a4.spec.dilation_kernel,
            inpaint_method=best_a4.spec.inpaint_method,
            temporal_window=int(w),
        )
        for w in temporal_grid
    ]
    best_a5 = execute_stage("A5", a5_candidates)

    # Materialize A-best to outputs/videos/<exp_id>/<dataset> for downstream compatibility.
    copy_a_best(best_candidate_root=best_a5.candidate_root, final_root=exp_pred_root, datasets=selected_datasets)

    # Final evaluation under exp_id (requested canonical output for A-best).
    final_summary_path, final_summary = run_evaluation(
        config_path=config_path,
        datasets=selected_datasets,
        pred_root=exp_pred_root,
        gt_root=gt_root,
        eval_exp_id=exp_id,
        allow_missing_gt=allow_missing_gt,
        save_visualization=save_visualization,
        logger=logger,
    )

    final_result = CandidateResult(
        spec=best_a5.spec,
        candidate_root=exp_pred_root,
        eval_exp_id=exp_id,
        summary_path=final_summary_path,
        aggregate=final_summary.get("aggregate", {}),
        per_dataset=final_summary.get("datasets", {}),
        mask_stats=best_a5.mask_stats,
    )

    exp_metrics_dir = REPO_ROOT / "outputs" / "metrics" / exp_id
    write_ablation_outputs(
        exp_metrics_dir=exp_metrics_dir,
        all_results=all_results,
        stage_best_map=stage_best,
        final_best=best_a5,
    )

    failure_csv, failure_explained_csv = build_failure_case_index(
        pred_root=exp_pred_root,
        gt_root=gt_root,
        datasets=selected_datasets,
        out_dir=REPO_ROOT / "outputs" / "figures" / exp_id / "failure_cases",
        detector=detector,
        quality_weights=quality_weights,
        tcf_dilate_kernel=int(tcf_cfg.get("dilate_kernel", 5)),
        bes_dilate_kernel=int(bes_cfg.get("dilate_kernel", 5)),
        bes_erode_kernel=int(bes_cfg.get("erode_kernel", 3)),
        bes_sobel_ksize=int(bes_cfg.get("sobel_ksize", 3)),
        fallback_applied_map={ds: bool(v.get("fallback_applied", False)) for ds, v in best_a5.mask_stats.items()},
        top_k=3,
    )

    report_path = exp_metrics_dir / "phase1_acceptance_report.md"
    write_acceptance_report(
        report_path=report_path,
        exp_id=exp_id,
        summary=final_summary,
        stage_best=stage_best,
        failure_explained_csv=failure_explained_csv,
        fallback_applied_final={ds: bool(v.get("fallback_applied", False)) for ds, v in best_a5.mask_stats.items()},
        seed=seed_value,
    )

    write_json(
        exp_metrics_dir / "phase1_run_meta.json",
        {
            "exp_id": exp_id,
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "config": str(args.config),
            "datasets": selected_datasets,
            "seed": seed_value,
            "seed_meta": seed_meta,
            "device": device,
            "selected_models": selected_models,
            "wild_fallback_mask_enabled": wild_fallback_mask,
            "output_policy": output_policy,
            "stage_best": {k: asdict(v.spec) for k, v in stage_best.items()},
            "a_best_candidate_root": str(best_a5.candidate_root),
            "a_best_pred_root": str(exp_pred_root),
            "a_best_eval_summary": str(final_summary_path),
            "a_best_aggregate": final_result.aggregate,
            "ablation_csv": str(exp_metrics_dir / "phase1_ablation.csv"),
            "selection_json": str(exp_metrics_dir / "phase1_selection.json"),
            "failure_case_csv": str(failure_csv),
            "failure_case_explained_csv": str(failure_explained_csv),
            "acceptance_report": str(report_path),
            "log_path": str(log_path),
        },
    )

    cleanup_stats = cleanup_video_only_outputs(
        exp_pred_root=exp_pred_root,
        datasets=selected_datasets,
        output_policy=output_policy,
    )
    logger.info("Video-only cleanup stats: %s", cleanup_stats)

    logger.info("Phase 1 complete.")
    logger.info("A-best summary: %s", final_summary_path)
    logger.info("Ablation table: %s", exp_metrics_dir / "phase1_ablation.csv")
    logger.info("Failure cases: %s", failure_csv)
    logger.info("Failure cases (explained): %s", failure_explained_csv)
    logger.info("Acceptance report: %s", report_path)


if __name__ == "__main__":
    main()
