#!/usr/bin/env python3
"""Export Phase 4 vs Phase 6 mask/GT overlay diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.evaluate_experiment import (  # noqa: E402
    build_gt_mask_index,
    list_images,
    load_config,
    normalize_mask_frame_key,
    read_color,
    read_gray,
    read_union_mask,
)
from src.common.video_io import decode_video_frames, resolve_output_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute per-frame IoU for Phase 4 and Phase 6 from masks and GT union masks, "
            "then export frames where Phase 6 is strongest versus Phase 4."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--phase4-exp-id", default="phase4_phase3to5_20260508_155231_pl220")
    parser.add_argument("--phase6-exp-id", default="phase6_20260509_123005_pl220")
    parser.add_argument("--datasets", default="bmx-trees,tennis")
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/gt"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics/phase6_vs_phase4_gt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures/phase6_vs_phase4_gt"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--tile-width", type=int, default=420)
    return parser.parse_args()


def load_binary_masks(
    exp_id: str,
    dataset: str,
    pred_root: Path,
    output_policy: dict[str, Any],
) -> tuple[list[np.ndarray], list[str], str]:
    ds_root = pred_root / exp_id / dataset
    mask_dir = ds_root / "masks"
    mask_paths = list_images(mask_dir)
    threshold = int(((output_policy.get("mask_h264", {}) or {}).get("threshold", 127)))

    if mask_paths:
        masks: list[np.ndarray] = []
        frame_names: list[str] = []
        for path in mask_paths:
            mask = read_gray(path)
            masks.append(((mask > 0).astype(np.uint8) * 255))
            frame_names.append(path.name)
        return masks, frame_names, "mask_dir"

    mask_video_name = str(output_policy.get("mask_video_name", "mask_h264.mp4"))
    mask_video_path = ds_root / mask_video_name
    decoded = decode_video_frames(mask_video_path, as_gray=True)
    if not decoded:
        raise RuntimeError(f"No masks found for {exp_id}/{dataset}: {mask_dir} or {mask_video_path}")

    masks = [((np.asarray(mask) > threshold).astype(np.uint8) * 255) for mask in decoded]
    frame_names = [f"frame_{idx:06d}.png" for idx in range(len(masks))]
    return masks, frame_names, "mask_video"


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, int, int, int, int, int]:
    pred_bin = np.asarray(pred) > 0
    gt_bin = np.asarray(gt) > 0
    inter = int(np.logical_and(pred_bin, gt_bin).sum())
    union = int(np.logical_or(pred_bin, gt_bin).sum())
    pred_area = int(pred_bin.sum())
    gt_area = int(gt_bin.sum())
    fp = int(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = int(np.logical_and(~pred_bin, gt_bin).sum())
    iou = float(inter / union) if union > 0 else 1.0
    return iou, inter, union, pred_area, gt_area, fp + fn


def resize_to_mask(image: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == mask_shape:
        return image
    return cv2.resize(image, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_LINEAR)


def load_input_frame(processed_root: Path, dataset: str, frame_name: str, shape: tuple[int, int]) -> np.ndarray:
    frame_path = processed_root / dataset / "frames" / frame_name
    if frame_path.exists():
        return resize_to_mask(read_color(frame_path), shape)
    return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)


def put_label(image: np.ndarray, text: str) -> np.ndarray:
    h, w = image.shape[:2]
    title_h = 44
    out = np.full((h + title_h, w, 3), 255, dtype=np.uint8)
    out[title_h:, :] = image
    cv2.putText(out, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (25, 25, 25), 2, cv2.LINE_AA)
    return out


def resize_panel(image: np.ndarray, tile_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    tile_height = max(1, int(round(h * tile_width / w)))
    return cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: float = 0.46) -> np.ndarray:
    base = image.copy()
    colored = np.zeros_like(base)
    colored[:, :] = color_bgr
    mask_bin = np.asarray(mask) > 0
    base[mask_bin] = cv2.addWeighted(base[mask_bin], 1.0 - alpha, colored[mask_bin], alpha, 0.0)
    return base


def overlay_two(
    image: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    pred_color: tuple[int, int, int],
) -> np.ndarray:
    out = overlay_mask(image, gt, (0, 190, 0), alpha=0.42)
    out = overlay_mask(out, pred, pred_color, alpha=0.38)
    overlap = np.logical_and(gt > 0, pred > 0)
    out[overlap] = cv2.addWeighted(out[overlap], 0.45, np.full_like(out[overlap], (0, 215, 215)), 0.55, 0)
    return out


def error_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    out = image.copy()
    pred_bin = pred > 0
    gt_bin = gt > 0
    tp = np.logical_and(pred_bin, gt_bin)
    fp = np.logical_and(pred_bin, ~gt_bin)
    fn = np.logical_and(~pred_bin, gt_bin)
    out[tp] = cv2.addWeighted(out[tp], 0.35, np.full_like(out[tp], (0, 190, 0)), 0.65, 0)
    out[fp] = cv2.addWeighted(out[fp], 0.35, np.full_like(out[fp], (0, 0, 230)), 0.65, 0)
    out[fn] = cv2.addWeighted(out[fn], 0.35, np.full_like(out[fn], (230, 0, 0)), 0.65, 0)
    return out


def make_grid(
    frame: np.ndarray,
    gt: np.ndarray,
    phase4: np.ndarray,
    phase6: np.ndarray,
    row: dict[str, Any],
    tile_width: int,
) -> np.ndarray:
    p4_iou = float(row["phase4_iou"])
    p6_iou = float(row["phase6_iou"])
    delta = float(row["delta_iou"])

    panels = [
        put_label(resize_panel(frame, tile_width), f"{row['dataset']} {row['frame_name']}"),
        put_label(resize_panel(overlay_mask(frame, gt, (0, 190, 0)), tile_width), "GT union mask"),
        put_label(resize_panel(overlay_mask(frame, phase4, (0, 0, 230)), tile_width), f"Phase 4 mask IoU={p4_iou:.3f}"),
        put_label(resize_panel(overlay_mask(frame, phase6, (230, 0, 0)), tile_width), f"Phase 6 mask IoU={p6_iou:.3f}"),
        put_label(resize_panel(error_overlay(frame, phase4, gt), tile_width), "Phase 4 error: TP/FP/FN"),
        put_label(resize_panel(error_overlay(frame, phase6, gt), tile_width), "Phase 6 error: TP/FP/FN"),
        put_label(resize_panel(overlay_two(frame, gt, phase4, (0, 0, 230)), tile_width), "GT+Phase 4"),
        put_label(resize_panel(overlay_two(frame, gt, phase6, (230, 0, 0)), tile_width), f"GT+Phase 6 delta={delta:+.3f}"),
    ]
    row1 = cv2.hconcat(panels[:4])
    row2 = cv2.hconcat(panels[4:])
    legend_h = 36
    legend = np.full((legend_h, row1.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        legend,
        "Colors: GT=green, Phase4=red, Phase6=blue, overlap=yellow; error maps: TP=green FP=red FN=blue",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    return cv2.vconcat([row1, row2, legend])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "frame_idx",
        "frame_name",
        "phase4_iou",
        "phase6_iou",
        "delta_iou",
        "phase4_intersection",
        "phase4_union",
        "phase6_intersection",
        "phase6_union",
        "phase4_area",
        "phase6_area",
        "gt_area",
        "phase4_error_pixels",
        "phase6_error_pixels",
        "phase4_mask_source",
        "phase6_mask_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_dataset(rows: list[dict[str, Any]], jr_threshold: float = 0.5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        ds_rows = [row for row in rows if str(row["dataset"]) == dataset]
        p4 = np.array([float(row["phase4_iou"]) for row in ds_rows], dtype=np.float32)
        p6 = np.array([float(row["phase6_iou"]) for row in ds_rows], dtype=np.float32)
        out.append(
            {
                "dataset": dataset,
                "frame_count": int(len(ds_rows)),
                "phase4_jm": float(p4.mean()) if len(p4) else None,
                "phase6_jm": float(p6.mean()) if len(p6) else None,
                "delta_jm": float(p6.mean() - p4.mean()) if len(p4) and len(p6) else None,
                "phase4_jr": float((p4 >= jr_threshold).mean()) if len(p4) else None,
                "phase6_jr": float((p6 >= jr_threshold).mean()) if len(p6) else None,
                "delta_jr": float((p6 >= jr_threshold).mean() - (p4 >= jr_threshold).mean())
                if len(p4) and len(p6)
                else None,
            }
        )
    return out


def mean_optional(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(np.array(vals, dtype=np.float32))) if vals else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_policy = resolve_output_policy(config)
    jr_threshold = float(((config.get("evaluation", {}) or {}).get("jr_iou_threshold", 0.5)))
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        gt_index = build_gt_mask_index(args.gt_root / dataset / "masks")
        if not gt_index:
            raise RuntimeError(f"No GT masks found for {dataset}: {args.gt_root / dataset / 'masks'}")

        p4_masks, p4_names, p4_source = load_binary_masks(args.phase4_exp_id, dataset, args.pred_root, output_policy)
        p6_masks, p6_names, p6_source = load_binary_masks(args.phase6_exp_id, dataset, args.pred_root, output_policy)
        p4_by_key = {normalize_mask_frame_key(name): mask for name, mask in zip(p4_names, p4_masks)}
        p6_by_key = {normalize_mask_frame_key(name): mask for name, mask in zip(p6_names, p6_masks)}

        for frame_idx, (gt_key, gt_paths) in enumerate(gt_index.items()):
            p4 = p4_by_key.get(gt_key)
            p6 = p6_by_key.get(gt_key)
            if p4 is None or p6 is None:
                continue
            if p4.shape != p6.shape:
                p6 = cv2.resize(p6, (p4.shape[1], p4.shape[0]), interpolation=cv2.INTER_NEAREST)
            gt = read_union_mask(gt_paths, target_shape=p4.shape[:2])
            p4_iou, p4_inter, p4_union, p4_area, gt_area, p4_err = compute_iou(p4, gt)
            p6_iou, p6_inter, p6_union, p6_area, _, p6_err = compute_iou(p6, gt)
            row = {
                "dataset": dataset,
                "frame_idx": frame_idx,
                "frame_name": f"{gt_key}.png",
                "phase4_iou": p4_iou,
                "phase6_iou": p6_iou,
                "delta_iou": p6_iou - p4_iou,
                "phase4_intersection": p4_inter,
                "phase4_union": p4_union,
                "phase6_intersection": p6_inter,
                "phase6_union": p6_union,
                "phase4_area": p4_area,
                "phase6_area": p6_area,
                "gt_area": gt_area,
                "phase4_error_pixels": p4_err,
                "phase6_error_pixels": p6_err,
                "phase4_mask_source": p4_source,
                "phase6_mask_source": p6_source,
            }
            all_rows.append(row)

    csv_path = args.metrics_dir / "phase4_phase6_frame_iou.csv"
    write_csv(csv_path, all_rows)

    candidates = sorted(
        [r for r in all_rows if float(r["delta_iou"]) >= args.min_delta],
        key=lambda r: (float(r["delta_iou"]), float(r["phase6_iou"])),
        reverse=True,
    )
    if not candidates:
        candidates = sorted(all_rows, key=lambda r: (float(r["delta_iou"]), float(r["phase6_iou"])), reverse=True)
    selected_rows = candidates[: max(0, args.top_k)]

    for rank, row in enumerate(selected_rows, start=1):
        dataset = str(row["dataset"])
        frame_name = str(row["frame_name"])
        key = normalize_mask_frame_key(frame_name)
        p4_masks, p4_names, _ = load_binary_masks(args.phase4_exp_id, dataset, args.pred_root, output_policy)
        p6_masks, p6_names, _ = load_binary_masks(args.phase6_exp_id, dataset, args.pred_root, output_policy)
        p4 = {normalize_mask_frame_key(name): mask for name, mask in zip(p4_names, p4_masks)}[key]
        p6 = {normalize_mask_frame_key(name): mask for name, mask in zip(p6_names, p6_masks)}[key]
        if p4.shape != p6.shape:
            p6 = cv2.resize(p6, (p4.shape[1], p4.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt = read_union_mask(build_gt_mask_index(args.gt_root / dataset / "masks")[key], target_shape=p4.shape[:2])
        frame = load_input_frame(args.processed_root, dataset, frame_name, p4.shape[:2])
        grid = make_grid(frame, gt, p4, p6, row, args.tile_width)
        out_name = (
            f"rank{rank:02d}_{dataset}_{key}_"
            f"p4_{float(row['phase4_iou']):.3f}_p6_{float(row['phase6_iou']):.3f}_"
            f"delta_{float(row['delta_iou']):+.3f}.png"
        ).replace("+", "plus")
        cv2.imwrite(str(args.out_dir / out_name), grid)

    dataset_summary = summarize_by_dataset(all_rows, jr_threshold=jr_threshold)
    aggregate_dataset_mean = {
        "phase4_jm": mean_optional([row["phase4_jm"] for row in dataset_summary]),
        "phase6_jm": mean_optional([row["phase6_jm"] for row in dataset_summary]),
        "delta_jm": mean_optional([row["delta_jm"] for row in dataset_summary]),
        "phase4_jr": mean_optional([row["phase4_jr"] for row in dataset_summary]),
        "phase6_jr": mean_optional([row["phase6_jr"] for row in dataset_summary]),
        "delta_jr": mean_optional([row["delta_jr"] for row in dataset_summary]),
    }
    summary = {
        "phase4_exp_id": args.phase4_exp_id,
        "phase6_exp_id": args.phase6_exp_id,
        "datasets": datasets,
        "frame_count": len(all_rows),
        "jr_threshold": jr_threshold,
        "selected_count": len(selected_rows),
        "min_delta": args.min_delta,
        "csv": str(csv_path),
        "out_dir": str(args.out_dir),
        "mean_phase4_iou": float(np.mean([float(r["phase4_iou"]) for r in all_rows])) if all_rows else None,
        "mean_phase6_iou": float(np.mean([float(r["phase6_iou"]) for r in all_rows])) if all_rows else None,
        "mean_delta_iou": float(np.mean([float(r["delta_iou"]) for r in all_rows])) if all_rows else None,
        "dataset_summary": dataset_summary,
        "aggregate_dataset_mean": aggregate_dataset_mean,
        "selected": selected_rows,
    }
    summary_path = args.metrics_dir / "selection_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md_lines = [
        "# Phase 4 vs Phase 6 GT Overlay Diagnostics",
        "",
        f"- Phase 4: `{args.phase4_exp_id}`",
        f"- Phase 6: `{args.phase6_exp_id}`",
        f"- CSV: `{csv_path}`",
        f"- Figure directory: `{args.out_dir}`",
        f"- Frame-weighted mean Phase4 IoU: `{summary['mean_phase4_iou']:.4f}`" if summary["mean_phase4_iou"] is not None else "- Frame-weighted mean Phase4 IoU: `N/A`",
        f"- Frame-weighted mean Phase6 IoU: `{summary['mean_phase6_iou']:.4f}`" if summary["mean_phase6_iou"] is not None else "- Frame-weighted mean Phase6 IoU: `N/A`",
        f"- Frame-weighted mean delta IoU: `{summary['mean_delta_iou']:+.4f}`" if summary["mean_delta_iou"] is not None else "- Frame-weighted mean delta IoU: `N/A`",
        f"- JR threshold: `{jr_threshold:.4f}`",
        f"- Dataset-mean Phase4 JM/JR: `{aggregate_dataset_mean['phase4_jm']:.4f}` / `{aggregate_dataset_mean['phase4_jr']:.4f}`",
        f"- Dataset-mean Phase6 JM/JR: `{aggregate_dataset_mean['phase6_jm']:.4f}` / `{aggregate_dataset_mean['phase6_jr']:.4f}`",
        f"- Dataset-mean delta JM/JR: `{aggregate_dataset_mean['delta_jm']:+.4f}` / `{aggregate_dataset_mean['delta_jr']:+.4f}`",
        "",
        "## Per-Dataset",
        "",
        "| Dataset | Frames | Phase4 JM | Phase6 JM | Delta JM | Phase4 JR | Phase6 JR | Delta JR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_summary:
        md_lines.append(
            f"| {row['dataset']} | {row['frame_count']} | "
            f"{float(row['phase4_jm']):.4f} | {float(row['phase6_jm']):.4f} | {float(row['delta_jm']):+.4f} | "
            f"{float(row['phase4_jr']):.4f} | {float(row['phase6_jr']):.4f} | {float(row['delta_jr']):+.4f} |"
        )
    md_lines.extend(
        [
            "",
            "## Selected Phase6-Better Frames",
            "",
            "| Rank | Dataset | Frame | Phase4 IoU | Phase6 IoU | Delta |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(selected_rows, start=1):
        md_lines.append(
            f"| {rank} | {row['dataset']} | {row['frame_name']} | "
            f"{float(row['phase4_iou']):.4f} | {float(row['phase6_iou']):.4f} | {float(row['delta_iou']):+.4f} |"
        )
    md_path = args.out_dir / "selection_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote figures under {args.out_dir}")
    print(
        json.dumps(
            {
                "frame_count": summary["frame_count"],
                "selected_count": summary["selected_count"],
                "frame_weighted": {
                    "mean_phase4_iou": summary["mean_phase4_iou"],
                    "mean_phase6_iou": summary["mean_phase6_iou"],
                    "mean_delta_iou": summary["mean_delta_iou"],
                },
                "dataset_mean": aggregate_dataset_mean,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
