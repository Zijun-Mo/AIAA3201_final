#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.fast_vqa import compute_fast_vqa_for_video
from src.common.mask_ranking import GT_COVERAGE_KEY, mask_score
from src.common.remove_quality import compute_remove_quality, ensure_masks_aligned
from src.common.video_io import dataset_video_paths, decode_video_frames, resolve_output_policy


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid bool value: {value}")


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_datasets(config: dict) -> tuple[list[str], list[str]]:
    datasets_cfg = config.get("datasets", {})
    mandatory = datasets_cfg.get("mandatory", {}) or {}
    optional = datasets_cfg.get("optional", {}) or {}

    all_names = list(mandatory.keys()) + [k for k in optional.keys() if k not in mandatory]
    mandatory_names = list(mandatory.keys())
    return all_names, mandatory_names


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


def list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def list_images_recursive(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def normalize_mask_frame_key(name: str) -> str:
    stem = Path(name).stem.lower()
    match = re.search(r"(?:frame[_-]?|^)(\d{1,8})(?=$|[_.-])", stem)
    if match:
        return f"frame_{int(match.group(1)):06d}"
    return stem


def build_gt_mask_index(gt_mask_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in list_images_recursive(gt_mask_dir):
        key = normalize_mask_frame_key(path.name)
        index.setdefault(key, []).append(path)
    return {key: sorted(paths) for key, paths in sorted(index.items())}


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Unable to read grayscale image: {path}")
    return img


def read_color(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Unable to read color image: {path}")
    return img


def read_union_mask(paths: list[Path], target_shape: tuple[int, int] | None = None) -> np.ndarray:
    if not paths:
        raise ValueError("read_union_mask requires at least one mask path")

    union: np.ndarray | None = None
    for path in paths:
        mask = read_gray(path)
        if target_shape is not None and mask.shape != target_shape:
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_bin = mask > 0
        union = mask_bin if union is None else np.logical_or(union, mask_bin)

    return (union.astype(np.uint8) * 255)


def compute_mask_metrics(
    pred_masks: list[np.ndarray],
    pred_frame_names: list[str],
    gt_mask_dir: Path,
    threshold: float,
) -> tuple[dict | None, str | None]:
    gt_index = build_gt_mask_index(gt_mask_dir)
    gt_items = list(gt_index.items())
    if not pred_masks:
        return None, "pred masks empty"
    if not gt_items:
        return None, f"gt mask folder empty: {gt_mask_dir}"

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    matched_by_name = 0
    gt_parts_merged = 0

    for name, pm in zip(pred_frame_names, pred_masks):
        gpaths = gt_index.get(normalize_mask_frame_key(name))
        if not gpaths:
            continue
        pm = np.asarray(pm)
        gm = read_union_mask(gpaths, target_shape=pm.shape[:2])
        pairs.append((pm, gm))
        matched_by_name += 1
        gt_parts_merged += len(gpaths)

    match_mode = "filename"
    if not pairs:
        match_mode = "index"
        n = min(len(pred_masks), len(gt_items))
        for idx in range(n):
            pm = np.asarray(pred_masks[idx])
            _, gpaths = gt_items[idx]
            gm = read_union_mask(gpaths, target_shape=pm.shape[:2])
            pairs.append((pm, gm))
            gt_parts_merged += len(gpaths)

    if not pairs:
        return None, "no usable mask pairs for JM/JR"

    scores = []
    gt_coverages = []
    for pm, gm in pairs:
        pm_bin = np.asarray(pm) > 0
        gm_bin = np.asarray(gm) > 0
        inter = np.logical_and(pm_bin, gm_bin).sum()
        union = np.logical_or(pm_bin, gm_bin).sum()
        gt_area = gm_bin.sum()
        pred_area = pm_bin.sum()
        iou = float(inter / union) if union > 0 else 1.0
        gt_coverage = float(inter / gt_area) if gt_area > 0 else (1.0 if pred_area == 0 else 0.0)
        scores.append(iou)
        gt_coverages.append(gt_coverage)

    arr = np.array(scores, dtype=np.float32)
    cov_arr = np.array(gt_coverages, dtype=np.float32)
    metrics = {
        "JM": float(arr.mean()),
        "JR": float((arr >= threshold).mean()),
        "GT_Coverage": float(cov_arr.mean()),
        "mask_frame_count": int(len(arr)),
        "mask_match_mode": match_mode,
        "mask_filename_matches": int(matched_by_name),
        "gt_mask_parts_merged": int(gt_parts_merged),
    }
    return metrics, None


def save_visualizations(
    pred_frames: list[np.ndarray],
    pred_masks: list[np.ndarray],
    pred_frame_names: list[str],
    gt_frame_dir: Path,
    gt_mask_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_frames:
        return

    first_pred = np.asarray(pred_frames[0])
    cv2.imwrite(str(out_dir / "sample_pred.png"), first_pred)

    gt_frames = list_images(gt_frame_dir)
    pred_first_name = pred_frame_names[0] if pred_frame_names else ""
    gt_map = {p.name: p for p in gt_frames}
    gt_first_path = gt_map.get(pred_first_name) or (gt_frames[0] if gt_frames else None)

    if gt_first_path is not None:
        gt_img = read_color(gt_first_path)
        if gt_img.shape != first_pred.shape:
            gt_img = cv2.resize(gt_img, (first_pred.shape[1], first_pred.shape[0]), interpolation=cv2.INTER_LINEAR)
        concat = cv2.hconcat([first_pred, gt_img])
        cv2.imwrite(str(out_dir / "sample_pred_vs_gt.png"), concat)

    gt_mask_index = build_gt_mask_index(gt_mask_dir)
    if pred_masks and gt_mask_index:
        pm = np.asarray(pred_masks[0])
        if pm.ndim == 3:
            pm = cv2.cvtColor(pm, cv2.COLOR_BGR2GRAY)
        gt_key = normalize_mask_frame_key(pred_first_name)
        gt_paths = gt_mask_index.get(gt_key) or next(iter(gt_mask_index.values()))
        gm = read_union_mask(gt_paths, target_shape=pm.shape[:2])

        pm3 = cv2.cvtColor(pm, cv2.COLOR_GRAY2BGR)
        gm3 = cv2.cvtColor(gm, cv2.COLOR_GRAY2BGR)
        mask_concat = cv2.hconcat([pm3, gm3])
        cv2.imwrite(str(out_dir / "sample_mask_vs_gt.png"), mask_concat)


def load_prediction_dataset(
    pred_root: Path,
    dataset: str,
    output_policy: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict[str, Any]]:
    ds_root = pred_root / dataset
    pred_frame_dir = ds_root / "frames"
    pred_mask_dir = ds_root / "masks"
    restored_video_path, mask_video_path = dataset_video_paths(ds_root, output_policy)
    mask_threshold = int(((output_policy.get("mask_h264", {}) or {}).get("threshold", 127)))

    frame_paths = list_images(pred_frame_dir)
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    frame_names: list[str] = []
    source = "none"

    if frame_paths:
        source = "frame_dir"
        frame_names = [p.name for p in frame_paths]
        frames = [read_color(p) for p in frame_paths]
        mask_map = {p.name: p for p in list_images(pred_mask_dir)}
        if mask_map:
            for name, frame in zip(frame_names, frames):
                mp = mask_map.get(name)
                if mp is None:
                    masks.append(np.zeros(frame.shape[:2], dtype=np.uint8))
                    continue
                mask = read_gray(mp)
                if mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                masks.append(((mask > 0).astype(np.uint8) * 255))
        elif mask_video_path.exists():
            source = "frame_dir+mask_video"
            decoded_masks = decode_video_frames(mask_video_path, as_gray=True)
            for idx, frame in enumerate(frames):
                if idx < len(decoded_masks):
                    m = np.asarray(decoded_masks[idx])
                    if m.shape != frame.shape[:2]:
                        m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    masks.append(((m > mask_threshold).astype(np.uint8) * 255))
                else:
                    masks.append(np.zeros(frame.shape[:2], dtype=np.uint8))
        else:
            masks = [np.zeros(frame.shape[:2], dtype=np.uint8) for frame in frames]
    else:
        decoded_frames = decode_video_frames(restored_video_path, as_gray=False)
        if not decoded_frames:
            raise RuntimeError(
                f"Dataset '{dataset}' has no prediction frames in either {pred_frame_dir} "
                f"or {restored_video_path}"
            )
        source = "video"
        frames = [np.asarray(f) for f in decoded_frames]
        frame_names = [f"frame_{idx:06d}.png" for idx in range(len(frames))]
        decoded_masks = decode_video_frames(mask_video_path, as_gray=True) if mask_video_path.exists() else []
        for idx, frame in enumerate(frames):
            if idx < len(decoded_masks):
                m = np.asarray(decoded_masks[idx])
                if m.shape != frame.shape[:2]:
                    m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                masks.append(((m > mask_threshold).astype(np.uint8) * 255))
            else:
                masks.append(np.zeros(frame.shape[:2], dtype=np.uint8))

    if not frames:
        raise RuntimeError(f"Dataset '{dataset}' produced empty frames after loading from {ds_root}")
    masks = ensure_masks_aligned(masks_u8=masks, frame_count=len(frames), frame_shape=frames[0].shape[:2])
    return frames, masks, frame_names, {
        "pred_frames": str(pred_frame_dir),
        "pred_masks": str(pred_mask_dir),
        "pred_restored_video": str(restored_video_path),
        "pred_mask_video": str(mask_video_path),
        "input_source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified evaluation entry for GT_Coverage/JM/JR + color TCF + FAST-VQA. "
            "Inputs support either <dataset>/frames+masks or restored_h264.mp4+mask_h264.mp4."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--exp-id", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="mandatory", help="mandatory | all | comma-separated")
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/gt"))
    parser.add_argument("--allow-missing-gt", type=str, default=None)
    parser.add_argument("--save-visualization", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {}) or {}

    allow_missing_gt = (
        str2bool(args.allow_missing_gt)
        if args.allow_missing_gt is not None
        else bool(eval_cfg.get("allow_missing_gt", True))
    )
    save_viz = (
        str2bool(args.save_visualization)
        if args.save_visualization is not None
        else bool(eval_cfg.get("save_visualization", True))
    )
    jr_threshold = float(eval_cfg.get("jr_iou_threshold", 0.5))
    metric_schema_version = str(eval_cfg.get("metric_schema_version", "v4_maskscore_color_tcf_fastvqa"))
    output_policy = resolve_output_policy(config)

    selection_cfg = eval_cfg.get("selection", {}) or {}
    exclude_for_aggregate = [str(x).strip() for x in (selection_cfg.get("exclude_datasets", []) or []) if str(x).strip()]

    tcf_cfg = eval_cfg.get("tcf", {}) or {}
    fast_vqa_cfg = eval_cfg.get("fast_vqa", {}) or {}

    all_names, mandatory_names = collect_datasets(config)
    selected_names = resolve_dataset_names(args.datasets, all_names, mandatory_names)

    metrics_dir = Path("outputs/metrics") / args.exp_id
    figures_dir = Path("outputs/figures") / args.exp_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "exp_id": args.exp_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "pred_root": str(args.pred_root),
        "gt_root": str(args.gt_root),
        "allow_missing_gt": allow_missing_gt,
        "jr_iou_threshold": jr_threshold,
        "metric_schema_version": metric_schema_version,
        "quality_metrics": ["TCF", "FAST_VQA"],
        "tcf": {"color_space": "bgr", **tcf_cfg},
        "fast_vqa": fast_vqa_cfg,
        "output_policy": output_policy,
        "selection": {"exclude_datasets": exclude_for_aggregate},
        "datasets": {},
        "aggregate": {},
    }

    rows = []
    jm_vals: list[float] = []
    jr_vals: list[float] = []
    gt_coverage_vals: list[float] = []
    tcf_vals: list[float] = []
    fast_vqa_vals: list[float] = []

    exclude_set = set(exclude_for_aggregate)
    for dataset in selected_names:
        pred_frames, pred_masks, pred_frame_names, pred_paths_meta = load_prediction_dataset(
            pred_root=args.pred_root,
            dataset=dataset,
            output_policy=output_policy,
        )
        gt_frame_dir = args.gt_root / dataset / "frames"
        gt_mask_dir = args.gt_root / dataset / "masks"

        ds_result = {
            "status": "ok",
            "notes": [],
            "metrics": {},
            "paths": {
                **pred_paths_meta,
                "gt_frames": str(gt_frame_dir),
                "gt_masks": str(gt_mask_dir),
            },
            "quality_detail": {},
        }

        gt_available = gt_frame_dir.exists() and bool(list_images(gt_frame_dir))

        if not gt_available and not allow_missing_gt:
            raise RuntimeError(
                f"Dataset '{dataset}' GT frames missing at {gt_frame_dir} and allow_missing_gt=false"
            )

        remove_metrics, frame_metrics, remove_notes = compute_remove_quality(
            frames_bgr=pred_frames,
            masks_u8=pred_masks,
            tcf_dilate_kernel=int(tcf_cfg.get("dilate_kernel", 5)),
        )
        ds_result["metrics"].update(remove_metrics)
        fast_vqa_score, fast_vqa_meta = compute_fast_vqa_for_video(
            Path(pred_paths_meta.get("pred_restored_video", "")),
            fast_vqa_cfg,
            repo_root=REPO_ROOT,
        )
        ds_result["metrics"]["FAST_VQA"] = fast_vqa_score
        ds_result["quality_detail"] = {
            "frame_metrics_path": str(metrics_dir / f"{dataset}_frame_metrics.csv"),
            **remove_notes,
            "fast_vqa": fast_vqa_meta,
        }
        if int(remove_notes.get("tcf_empty_region_frame_count", 0)) > 0:
            ds_result["notes"].append(
                f"TCF empty-region frames set to 0: {int(remove_notes.get('tcf_empty_region_frame_count', 0))}"
            )
        if fast_vqa_score is None:
            ds_result["notes"].append(f"FAST_VQA unavailable: {fast_vqa_meta.get('status')}")

        frame_metrics_path = metrics_dir / f"{dataset}_frame_metrics.csv"
        with frame_metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_idx", "TCF"])
            writer.writeheader()
            for idx, item in enumerate(frame_metrics):
                writer.writerow({"frame_idx": idx, **item})

        if not gt_available:
            ds_result["status"] = "gt_missing"
            ds_result["notes"].append("GT missing, JM/JR skipped")
        else:
            mask_metrics, mask_note = compute_mask_metrics(
                pred_masks=pred_masks,
                pred_frame_names=pred_frame_names,
                gt_mask_dir=gt_mask_dir,
                threshold=jr_threshold,
            )
            if mask_metrics is not None:
                ds_result["metrics"].update(mask_metrics)
                ds_result["metrics"]["MaskScore"] = mask_score(ds_result["metrics"])
            else:
                ds_result["notes"].append(f"mask metrics skipped: {mask_note}")

        if save_viz:
            save_visualizations(
                pred_frames=pred_frames,
                pred_masks=pred_masks,
                pred_frame_names=pred_frame_names,
                gt_frame_dir=gt_frame_dir,
                gt_mask_dir=gt_mask_dir,
                out_dir=figures_dir / dataset,
            )

        summary["datasets"][dataset] = ds_result

        ds_metrics = ds_result["metrics"]
        if ds_metrics.get("JM") is not None:
            jm_vals.append(float(ds_metrics["JM"]))
        if ds_metrics.get("JR") is not None:
            jr_vals.append(float(ds_metrics["JR"]))
        if ds_metrics.get("GT_Coverage") is not None:
            gt_coverage_vals.append(float(ds_metrics["GT_Coverage"]))

        if dataset not in exclude_set:
            tcf_vals.append(float(ds_metrics.get("TCF", 0.0)))
            if ds_metrics.get("FAST_VQA") is not None:
                fast_vqa_vals.append(float(ds_metrics["FAST_VQA"]))

        rows.append(
            {
                "dataset": dataset,
                "status": ds_result["status"],
                "JM": ds_metrics.get("JM", ""),
                "JR": ds_metrics.get("JR", ""),
                GT_COVERAGE_KEY: ds_metrics.get(GT_COVERAGE_KEY, ""),
                "MaskScore": ds_metrics.get("MaskScore", ""),
                "TCF": ds_metrics.get("TCF", ""),
                "FAST_VQA": ds_metrics.get("FAST_VQA", ""),
                "mask_frame_count": ds_metrics.get("mask_frame_count", ""),
                "video_frame_count": ds_metrics.get("video_frame_count", ""),
                "fast_vqa_status": ((ds_result.get("quality_detail", {}) or {}).get("fast_vqa", {}) or {}).get("status", ""),
                "notes": " | ".join(ds_result["notes"]),
            }
        )

    summary["aggregate"] = {
        "datasets_evaluated": len(selected_names),
        "datasets_aggregated": int(len(selected_names) - len([x for x in selected_names if x in set(exclude_for_aggregate)])),
        "aggregated_excluding": exclude_for_aggregate,
        "JM": float(np.mean(np.array(jm_vals))) if jm_vals else None,
        "JR": float(np.mean(np.array(jr_vals))) if jr_vals else None,
        GT_COVERAGE_KEY: float(np.mean(np.array(gt_coverage_vals))) if gt_coverage_vals else None,
        "TCF": float(np.mean(np.array(tcf_vals))) if tcf_vals else None,
        "FAST_VQA": float(np.mean(np.array(fast_vqa_vals))) if fast_vqa_vals else None,
    }
    summary["aggregate"]["MaskScore"] = mask_score(summary["aggregate"])

    summary_path = metrics_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = metrics_dir / "per_dataset.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "status",
                "JM",
                "JR",
                GT_COVERAGE_KEY,
                "MaskScore",
                "TCF",
                "FAST_VQA",
                "mask_frame_count",
                "video_frame_count",
                "fast_vqa_status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Summary: {summary_path}")
    print(f"[OK] CSV: {csv_path}")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
