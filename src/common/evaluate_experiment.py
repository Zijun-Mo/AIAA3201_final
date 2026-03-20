#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.metrics import structural_similarity as ssim


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


def compute_mask_metrics(pred_mask_dir: Path, gt_mask_dir: Path, threshold: float) -> tuple[dict | None, str | None]:
    pred = list_images(pred_mask_dir)
    gt_map = {p.name: p for p in list_images(gt_mask_dir)}

    if not pred:
        return None, f"pred mask folder empty: {pred_mask_dir}"
    if not gt_map:
        return None, f"gt mask folder empty: {gt_mask_dir}"

    scores = []
    for p in pred:
        gpath = gt_map.get(p.name)
        if gpath is None:
            continue
        pm = read_gray(p)
        gm = read_gray(gpath)
        if pm.shape != gm.shape:
            gm = cv2.resize(gm, (pm.shape[1], pm.shape[0]), interpolation=cv2.INTER_NEAREST)

        pm_bin = pm > 0
        gm_bin = gm > 0
        inter = np.logical_and(pm_bin, gm_bin).sum()
        union = np.logical_or(pm_bin, gm_bin).sum()
        iou = float(inter / union) if union > 0 else 1.0
        scores.append(iou)

    if not scores:
        return None, "no overlapping mask filenames between prediction and GT"

    arr = np.array(scores, dtype=np.float32)
    metrics = {
        "JM": float(arr.mean()),
        "JR": float((arr >= threshold).mean()),
        "mask_frame_count": int(len(arr)),
    }
    return metrics, None


def compute_video_metrics(pred_frame_dir: Path, gt_frame_dir: Path) -> tuple[dict | None, str | None]:
    pred = list_images(pred_frame_dir)
    gt_map = {p.name: p for p in list_images(gt_frame_dir)}

    if not pred:
        return None, f"pred frame folder empty: {pred_frame_dir}"
    if not gt_map:
        return None, f"gt frame folder empty: {gt_frame_dir}"

    psnrs = []
    ssims = []
    for p in pred:
        gpath = gt_map.get(p.name)
        if gpath is None:
            continue

        pi = read_color(p)
        gi = read_color(gpath)
        if pi.shape != gi.shape:
            gi = cv2.resize(gi, (pi.shape[1], pi.shape[0]), interpolation=cv2.INTER_LINEAR)

        psnrs.append(float(cv2.PSNR(pi, gi)))
        ssims.append(float(ssim(pi, gi, channel_axis=2)))

    if not psnrs:
        return None, "no overlapping frame filenames between prediction and GT"

    return {
        "PSNR": float(np.mean(np.array(psnrs, dtype=np.float32))),
        "SSIM": float(np.mean(np.array(ssims, dtype=np.float32))),
        "video_frame_count": int(len(psnrs)),
    }, None


def save_visualizations(
    pred_frame_dir: Path,
    gt_frame_dir: Path,
    pred_mask_dir: Path,
    gt_mask_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_frames = list_images(pred_frame_dir)
    if not pred_frames:
        return

    first_pred = read_color(pred_frames[0])
    cv2.imwrite(str(out_dir / "sample_pred.png"), first_pred)

    gt_frames = list_images(gt_frame_dir)
    gt_map = {p.name: p for p in gt_frames}
    gt_first_path = gt_map.get(pred_frames[0].name) or (gt_frames[0] if gt_frames else None)

    if gt_first_path is not None:
        gt_img = read_color(gt_first_path)
        if gt_img.shape != first_pred.shape:
            gt_img = cv2.resize(gt_img, (first_pred.shape[1], first_pred.shape[0]), interpolation=cv2.INTER_LINEAR)
        concat = cv2.hconcat([first_pred, gt_img])
        cv2.imwrite(str(out_dir / "sample_pred_vs_gt.png"), concat)

    pred_masks = list_images(pred_mask_dir)
    gt_masks = list_images(gt_mask_dir)
    if pred_masks and gt_masks:
        pm = read_gray(pred_masks[0])
        gm_map = {p.name: p for p in gt_masks}
        gm_path = gm_map.get(pred_masks[0].name) or gt_masks[0]
        gm = read_gray(gm_path)

        if pm.shape != gm.shape:
            gm = cv2.resize(gm, (pm.shape[1], pm.shape[0]), interpolation=cv2.INTER_NEAREST)

        pm3 = cv2.cvtColor(pm, cv2.COLOR_GRAY2BGR)
        gm3 = cv2.cvtColor(gm, cv2.COLOR_GRAY2BGR)
        mask_concat = cv2.hconcat([pm3, gm3])
        cv2.imwrite(str(out_dir / "sample_mask_vs_gt.png"), mask_concat)


def ensure_pred_frames_exist(pred_root: Path, dataset: str) -> Path:
    pred_frame_dir = pred_root / dataset / "frames"
    frames = list_images(pred_frame_dir)
    if not frames:
        raise RuntimeError(
            f"Dataset '{dataset}' prediction frames missing or empty: {pred_frame_dir}. "
            "Expected normalized frame outputs at pred_root/<dataset>/frames/."
        )
    return pred_frame_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Phase 0 evaluation entry for JM/JR/PSNR/SSIM with optional missing GT support."
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
    eval_cfg = config.get("evaluation", {})

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
        "datasets": {},
        "aggregate": {},
    }

    rows = []
    jm_vals, jr_vals, psnr_vals, ssim_vals = [], [], [], []

    for dataset in selected_names:
        pred_frame_dir = ensure_pred_frames_exist(args.pred_root, dataset)
        pred_mask_dir = args.pred_root / dataset / "masks"
        gt_frame_dir = args.gt_root / dataset / "frames"
        gt_mask_dir = args.gt_root / dataset / "masks"

        ds_result = {
            "status": "ok",
            "notes": [],
            "metrics": {},
            "paths": {
                "pred_frames": str(pred_frame_dir),
                "pred_masks": str(pred_mask_dir),
                "gt_frames": str(gt_frame_dir),
                "gt_masks": str(gt_mask_dir),
            },
        }

        gt_available = gt_frame_dir.exists() and bool(list_images(gt_frame_dir))

        if not gt_available and not allow_missing_gt:
            raise RuntimeError(
                f"Dataset '{dataset}' GT frames missing at {gt_frame_dir} and allow_missing_gt=false"
            )

        if not gt_available:
            ds_result["status"] = "gt_missing"
            ds_result["notes"].append("GT missing, metrics skipped")
        else:
            video_metrics, video_note = compute_video_metrics(pred_frame_dir, gt_frame_dir)
            if video_metrics is not None:
                ds_result["metrics"].update(video_metrics)
                psnr_vals.append(video_metrics["PSNR"])
                ssim_vals.append(video_metrics["SSIM"])
            else:
                ds_result["notes"].append(f"video metrics skipped: {video_note}")

            mask_metrics, mask_note = compute_mask_metrics(pred_mask_dir, gt_mask_dir, jr_threshold)
            if mask_metrics is not None:
                ds_result["metrics"].update(mask_metrics)
                jm_vals.append(mask_metrics["JM"])
                jr_vals.append(mask_metrics["JR"])
            else:
                ds_result["notes"].append(f"mask metrics skipped: {mask_note}")

        if save_viz:
            save_visualizations(pred_frame_dir, gt_frame_dir, pred_mask_dir, gt_mask_dir, figures_dir / dataset)

        summary["datasets"][dataset] = ds_result

        rows.append(
            {
                "dataset": dataset,
                "status": ds_result["status"],
                "JM": ds_result["metrics"].get("JM", ""),
                "JR": ds_result["metrics"].get("JR", ""),
                "PSNR": ds_result["metrics"].get("PSNR", ""),
                "SSIM": ds_result["metrics"].get("SSIM", ""),
                "mask_frame_count": ds_result["metrics"].get("mask_frame_count", ""),
                "video_frame_count": ds_result["metrics"].get("video_frame_count", ""),
                "notes": " | ".join(ds_result["notes"]),
            }
        )

    summary["aggregate"] = {
        "datasets_evaluated": len(selected_names),
        "JM": float(np.mean(np.array(jm_vals))) if jm_vals else None,
        "JR": float(np.mean(np.array(jr_vals))) if jr_vals else None,
        "PSNR": float(np.mean(np.array(psnr_vals))) if psnr_vals else None,
        "SSIM": float(np.mean(np.array(ssim_vals))) if ssim_vals else None,
    }

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
                "PSNR",
                "SSIM",
                "mask_frame_count",
                "video_frame_count",
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
