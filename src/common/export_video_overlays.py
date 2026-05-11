#!/usr/bin/env python3
"""Export mask-on-input overlay videos for predictions and GT masks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.evaluate_experiment import build_gt_mask_index, normalize_mask_frame_key, read_union_mask
from src.common.video_io import (
    _encode_h264_raw_frames,
    dataset_video_paths,
    decode_video_frames,
    load_masks_by_names_with_video_fallback,
    resolve_output_policy,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DEFAULT_PRED_OVERLAY_NAME = "mask_overlay_h264.mp4"
DEFAULT_GT_ROOT_NAME = "gt"
DEFAULT_GT_OVERLAY_NAME = "mask_overlay_h264.mp4"


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def collect_dataset_cfg(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    datasets_cfg = config.get("datasets", {}) or {}
    mandatory = datasets_cfg.get("mandatory", {}) or {}
    optional = datasets_cfg.get("optional", {}) or {}
    all_map: dict[str, dict[str, Any]] = {}
    for name, cfg in {**mandatory, **optional}.items():
        all_map[str(name)] = cfg or {}
    mandatory_names = [str(x) for x in mandatory.keys()]
    all_names = list(all_map.keys())
    return all_map, all_names, mandatory_names


def resolve_dataset_names(spec: str, all_names: list[str], mandatory_names: list[str]) -> list[str]:
    token = str(spec or "mandatory").strip()
    low = token.lower()
    if low == "mandatory":
        return list(mandatory_names)
    if low == "all":
        return list(all_names)
    requested = [x.strip() for x in token.split(",") if x.strip()]
    unknown = [x for x in requested if x not in all_names]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Valid: {all_names}")
    return requested


def resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def load_original_frames(dataset_cfg: dict[str, Any], dataset_name: str) -> tuple[list[np.ndarray], list[str]]:
    frame_dir = resolve_repo_path(dataset_cfg.get("processed_frames_dir", f"data/processed/{dataset_name}/frames"))
    frame_paths = list_images(frame_dir)
    frames: list[np.ndarray] = []
    names: list[str] = []
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frames.append(frame)
        names.append(path.name)
    if frames:
        return frames, names

    raw_video = resolve_repo_path(dataset_cfg.get("raw_video", f"data/raw/{dataset_name}.mp4"))
    frames = decode_video_frames(raw_video, as_gray=False)
    names = [f"frame_{idx:06d}.png" for idx in range(len(frames))]
    return frames, names


def overlay_masks(
    frames_bgr: list[np.ndarray],
    masks_u8: list[np.ndarray],
    *,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    alpha_f = float(np.clip(alpha, 0.0, 1.0))
    for frame, mask in zip(frames_bgr, masks_u8):
        base = np.asarray(frame).copy()
        mask_arr = np.asarray(mask)
        if mask_arr.shape[:2] != base.shape[:2]:
            mask_arr = cv2.resize(mask_arr, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
        idx = mask_arr > 0
        if np.any(idx):
            blended = base.astype(np.float32)
            blended[idx] = blended[idx] * (1.0 - alpha_f) + color * alpha_f
            base = np.clip(blended, 0, 255).astype(np.uint8)
        out.append(base)
    return out


def encode_color(frames: list[np.ndarray], out_path: Path, fps: float, *, crf: int, preset: str) -> None:
    _encode_h264_raw_frames(
        frames=frames,
        out_path=out_path,
        fps=fps,
        crf=crf,
        preset=preset,
        output_pix_fmt="yuv420p",
        input_pix_fmt="bgr24",
    )


def encode_gray(frames: list[np.ndarray], out_path: Path, fps: float, *, crf: int, preset: str) -> None:
    _encode_h264_raw_frames(
        frames=frames,
        out_path=out_path,
        fps=fps,
        crf=crf,
        preset=preset,
        output_pix_fmt="gray",
        input_pix_fmt="gray",
    )


def load_final_exp_ids(videos_root: Path) -> list[str]:
    manifest = REPO_ROOT / "outputs" / "metrics" / "final_results" / "final_results_manifest.json"
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    source = payload.get("source_experiments", {}) or {}
    exp_ids: list[str] = []
    for exp_id in source.values():
        token = str(exp_id).strip()
        if token and token not in exp_ids and (videos_root / token).exists():
            exp_ids.append(token)
    return exp_ids


def resolve_exp_ids(spec: str, videos_root: Path) -> list[str]:
    token = str(spec or "final").strip()
    if token.lower() == "final":
        exp_ids = load_final_exp_ids(videos_root)
        if not exp_ids:
            raise RuntimeError("No final experiment ids found in outputs/metrics/final_results/final_results_manifest.json")
        return exp_ids
    if token.lower() == "all":
        return sorted([p.name for p in videos_root.iterdir() if p.is_dir() and p.name != DEFAULT_GT_ROOT_NAME])
    return [x.strip() for x in token.split(",") if x.strip()]


def export_prediction_overlay(
    *,
    exp_id: str,
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    videos_root: Path,
    output_policy: dict[str, Any],
    fps: float,
    alpha: float,
    color_bgr: tuple[int, int, int],
    overlay_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    dataset_root = videos_root / exp_id / dataset_name
    restored_path, mask_video_path = dataset_video_paths(dataset_root, output_policy)
    out_path = dataset_root / overlay_name
    meta: dict[str, Any] = {
        "exp_id": exp_id,
        "dataset": dataset_name,
        "output": str(out_path),
        "restored_video": str(restored_path),
        "mask_video": str(mask_video_path),
    }
    if not dataset_root.exists():
        meta["status"] = "skipped"
        meta["reason"] = "dataset_output_missing"
        return meta
    if out_path.exists() and not overwrite:
        meta["status"] = "cached"
        return meta
    if not mask_video_path.exists() and not (dataset_root / "masks").exists():
        meta["status"] = "skipped"
        meta["reason"] = "pred_mask_missing"
        return meta

    frames, frame_names = load_original_frames(dataset_cfg, dataset_name)
    if not frames:
        meta["status"] = "skipped"
        meta["reason"] = "original_frames_missing"
        return meta
    h, w = frames[0].shape[:2]
    masks, mask_meta = load_masks_by_names_with_video_fallback(
        mask_dir=dataset_root / "masks",
        frame_names=frame_names,
        frame_shape=(h, w),
        mask_video_path=mask_video_path,
        threshold=int((output_policy.get("mask_h264", {}) or {}).get("threshold", 127)),
    )
    meta["mask_load"] = mask_meta
    if not masks:
        meta["status"] = "skipped"
        meta["reason"] = "pred_mask_load_failed"
        return meta

    n = min(len(frames), len(masks))
    overlays = overlay_masks(frames[:n], masks[:n], color_bgr=color_bgr, alpha=alpha)
    encode_color(
        overlays,
        out_path,
        fps=fps,
        crf=int((output_policy.get("restored_h264", {}) or {}).get("crf", 23)),
        preset=str((output_policy.get("restored_h264", {}) or {}).get("preset", "medium")),
    )
    meta["status"] = "ok"
    meta["frame_count"] = int(n)
    return meta


def build_gt_masks(
    *,
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    frames: list[np.ndarray],
    frame_names: list[str],
    gt_root: Path,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    gt_mask_dir = gt_root / dataset_name / "masks"
    meta: dict[str, Any] = {
        "dataset": dataset_name,
        "gt_mask_dir": str(gt_mask_dir),
    }
    if not gt_mask_dir.exists():
        meta["status"] = "skipped"
        meta["reason"] = "gt_mask_dir_missing"
        return [], meta
    gt_index = build_gt_mask_index(gt_mask_dir)
    if not gt_index:
        meta["status"] = "skipped"
        meta["reason"] = "gt_mask_index_empty"
        return [], meta

    h, w = frames[0].shape[:2]
    masks: list[np.ndarray] = []
    matched = 0
    parts = 0
    for name in frame_names:
        paths = gt_index.get(normalize_mask_frame_key(name))
        if paths:
            mask = read_union_mask(paths, target_shape=(h, w))
            matched += 1
            parts += len(paths)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        masks.append(mask)
    meta.update(
        {
            "status": "ok",
            "frame_count": int(len(masks)),
            "matched_frames": int(matched),
            "gt_mask_parts_merged": int(parts),
        }
    )
    return masks, meta


def export_gt_videos(
    *,
    dataset_name: str,
    dataset_cfg: dict[str, Any],
    videos_root: Path,
    gt_root: Path,
    fps: float,
    alpha: float,
    color_bgr: tuple[int, int, int],
    output_policy: dict[str, Any],
    gt_root_name: str,
    overlay_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    out_root = videos_root / gt_root_name / dataset_name
    mask_out = out_root / str(output_policy.get("mask_video_name", "mask_h264.mp4"))
    overlay_out = out_root / overlay_name
    meta: dict[str, Any] = {
        "dataset": dataset_name,
        "mask_output": str(mask_out),
        "overlay_output": str(overlay_out),
    }
    if mask_out.exists() and overlay_out.exists() and not overwrite:
        meta["status"] = "cached"
        return meta

    frames, frame_names = load_original_frames(dataset_cfg, dataset_name)
    if not frames:
        meta["status"] = "skipped"
        meta["reason"] = "original_frames_missing"
        return meta
    gt_masks, gt_meta = build_gt_masks(
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        frames=frames,
        frame_names=frame_names,
        gt_root=gt_root,
    )
    meta["gt_mask_load"] = gt_meta
    if not gt_masks:
        meta["status"] = "skipped"
        meta["reason"] = gt_meta.get("reason", "gt_mask_unavailable")
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "gt_missing.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    n = min(len(frames), len(gt_masks))
    encode_gray(
        [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in gt_masks[:n]],
        mask_out,
        fps=fps,
        crf=int((output_policy.get("mask_h264", {}) or {}).get("crf", 0)),
        preset=str((output_policy.get("mask_h264", {}) or {}).get("preset", "medium")),
    )
    overlays = overlay_masks(frames[:n], gt_masks[:n], color_bgr=color_bgr, alpha=alpha)
    encode_color(
        overlays,
        overlay_out,
        fps=fps,
        crf=int((output_policy.get("restored_h264", {}) or {}).get("crf", 23)),
        preset=str((output_policy.get("restored_h264", {}) or {}).get("preset", "medium")),
    )
    meta["status"] = "ok"
    meta["frame_count"] = int(n)
    return meta


def parse_color(token: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    parts = [x.strip() for x in str(token or "").split(",") if x.strip()]
    if len(parts) != 3:
        return default
    vals = tuple(int(np.clip(int(x), 0, 255)) for x in parts)
    return vals  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export prediction/GT mask overlay videos under outputs/videos.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--datasets", default="mandatory", help="mandatory | all | comma-separated dataset names")
    parser.add_argument("--exp-ids", default="final", help="final | all | comma-separated experiment ids")
    parser.add_argument("--videos-root", default="outputs/videos")
    parser.add_argument("--gt-root", default="data/gt")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--overwrite", default="false")
    parser.add_argument("--pred-overlay-name", default=DEFAULT_PRED_OVERLAY_NAME)
    parser.add_argument("--gt-root-name", default=DEFAULT_GT_ROOT_NAME)
    parser.add_argument("--gt-overlay-name", default=DEFAULT_GT_OVERLAY_NAME)
    parser.add_argument("--pred-color-bgr", default="0,0,255", help="B,G,R for prediction overlay. Default red.")
    parser.add_argument("--gt-color-bgr", default="0,180,0", help="B,G,R for GT overlay. Default green.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(resolve_repo_path(args.config))
    output_policy = resolve_output_policy(config)
    datasets_map, all_names, mandatory_names = collect_dataset_cfg(config)
    datasets = resolve_dataset_names(args.datasets, all_names, mandatory_names)
    videos_root = resolve_repo_path(args.videos_root)
    gt_root = resolve_repo_path(args.gt_root)
    exp_ids = resolve_exp_ids(args.exp_ids, videos_root)
    fps = float(args.fps) if args.fps is not None else float((config.get("preprocess", {}) or {}).get("target_fps", 24))
    overwrite = str2bool(args.overwrite)
    pred_color = parse_color(args.pred_color_bgr, (0, 0, 255))
    gt_color = parse_color(args.gt_color_bgr, (0, 180, 0))

    results: dict[str, Any] = {
        "exp_ids": exp_ids,
        "datasets": datasets,
        "prediction_overlays": [],
        "gt_videos": [],
    }

    for exp_id in exp_ids:
        for ds in datasets:
            meta = export_prediction_overlay(
                exp_id=exp_id,
                dataset_name=ds,
                dataset_cfg=datasets_map[ds],
                videos_root=videos_root,
                output_policy=output_policy,
                fps=fps,
                alpha=float(args.alpha),
                color_bgr=pred_color,
                overlay_name=str(args.pred_overlay_name),
                overwrite=overwrite,
            )
            results["prediction_overlays"].append(meta)
            print(f"[pred] {exp_id}/{ds}: {meta.get('status')} -> {meta.get('output')}")

    for ds in datasets:
        meta = export_gt_videos(
            dataset_name=ds,
            dataset_cfg=datasets_map[ds],
            videos_root=videos_root,
            gt_root=gt_root,
            fps=fps,
            alpha=float(args.alpha),
            color_bgr=gt_color,
            output_policy=output_policy,
            gt_root_name=str(args.gt_root_name),
            overlay_name=str(args.gt_overlay_name),
            overwrite=overwrite,
        )
        results["gt_videos"].append(meta)
        print(f"[gt] {ds}: {meta.get('status')} -> {meta.get('overlay_output')}")

    manifest_path = videos_root / "overlay_export_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
