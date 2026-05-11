#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.video_io import (
    compute_mask_coverage_from_dir_or_video,
    count_video_frames,
    dataset_video_paths,
    resolve_output_policy,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_mandatory_datasets(config: dict) -> list[str]:
    mandatory = config.get("datasets", {}).get("mandatory", {}) or {}
    if not isinstance(mandatory, dict) or not mandatory:
        raise ValueError("No mandatory datasets found under datasets.mandatory")
    return list(mandatory.keys())


def list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def get_wild_coverage_thresholds(config: dict) -> tuple[float, float]:
    selection_cfg = ((config.get("evaluation", {}) or {}).get("selection", {}) or {})
    mean_cfg = selection_cfg.get("min_mean_mask_ratio_by_dataset", {}) or {}
    active_cfg = selection_cfg.get("min_active_frame_ratio_by_dataset", {}) or {}
    if not isinstance(mean_cfg, dict):
        mean_cfg = {}
    if not isinstance(active_cfg, dict):
        active_cfg = {}
    return float(mean_cfg.get("wild", 0.0)), float(active_cfg.get("wild", 0.0))


def check_outputs(
    pred_root: Path,
    datasets: list[str],
    output_policy: dict,
    wild_min_mean_mask_ratio: float,
    wild_min_active_frame_ratio: float,
) -> list[str]:
    issues: list[str] = []
    for ds in datasets:
        ds_root = pred_root / ds
        frame_dir = ds_root / "frames"
        mask_dir = ds_root / "masks"
        frames = list_images(frame_dir)
        masks = list_images(mask_dir)
        restored_video, mask_video = dataset_video_paths(ds_root, output_policy)
        frame_video_count = count_video_frames(restored_video)
        mask_video_count = count_video_frames(mask_video)

        frame_count = len(frames) if frames else frame_video_count
        mask_count = len(masks) if masks else mask_video_count
        if frame_count <= 0:
            issues.append(
                f"{ds}: no prediction frames found in dir/video ({frame_dir}, {restored_video})"
            )
        if mask_count <= 0:
            issues.append(
                f"{ds}: no prediction masks found in dir/video ({mask_dir}, {mask_video})"
            )
        if frame_count > 0 and mask_count > 0 and frame_count != mask_count:
            issues.append(f"{ds}: frame/mask count mismatch ({frame_count} vs {mask_count})")

    if "wild" in datasets:
        wild_ds_root = pred_root / "wild"
        _, wild_mask_video = dataset_video_paths(wild_ds_root, output_policy)
        coverage = compute_mask_coverage_from_dir_or_video(
            mask_dir=wild_ds_root / "masks",
            mask_video_path=wild_mask_video,
            threshold=int(((output_policy.get("mask_h264", {}) or {}).get("threshold", 127))),
        )
        if int(coverage.get("frame_count", 0)) <= 0:
            issues.append("wild: unable to compute mask coverage (no mask frames/video)")
            return issues

        mean_ratio = float(coverage.get("mean_mask_ratio", 0.0))
        active_ratio = float(coverage.get("active_frame_ratio", 0.0))

        if wild_min_mean_mask_ratio > 0.0 or wild_min_active_frame_ratio > 0.0:
            if mean_ratio < wild_min_mean_mask_ratio:
                issues.append(
                    f"wild: mean_mask_ratio too low ({mean_ratio:.6f} < {wild_min_mean_mask_ratio:.6f})"
                )
            if active_ratio < wild_min_active_frame_ratio:
                issues.append(
                    f"wild: active_frame_ratio too low ({active_ratio:.6f} < {wild_min_active_frame_ratio:.6f})"
                )
        elif mean_ratio <= 0.0 or active_ratio <= 0.0:
            issues.append("wild: all final masks are zero")
    return issues


def check_metrics(metrics_dir: Path) -> list[str]:
    issues: list[str] = []
    required = [
        metrics_dir / "summary.json",
        metrics_dir / "per_dataset.csv",
        metrics_dir / "phase1_ablation.csv",
        metrics_dir / "phase1_selection.json",
        metrics_dir / "phase1_acceptance_report.md",
    ]
    for p in required:
        if not p.exists():
            issues.append(f"missing required file: {p}")

    summary_path = metrics_dir / "summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            if not isinstance(summary.get("datasets", {}), dict):
                issues.append(f"invalid summary datasets field: {summary_path}")
            agg = summary.get("aggregate", {}) or {}
            for key in ["GT_Coverage", "JM", "JR", "MaskScore", "TCF", "FAST_VQA"]:
                if key not in agg:
                    issues.append(f"missing aggregate metric '{key}' in {summary_path}")
            for key in ["PSNR", "SSIM", "ROS", "BES"]:
                if key in agg:
                    issues.append(f"forbidden legacy metric '{key}' in {summary_path}")
        except Exception as e:
            issues.append(f"invalid summary.json ({e})")

    per_dataset_csv = metrics_dir / "per_dataset.csv"
    if per_dataset_csv.exists():
        try:
            with per_dataset_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = set(reader.fieldnames or [])
            for key in ["GT_Coverage", "JM", "JR", "MaskScore", "TCF", "FAST_VQA"]:
                if key not in headers:
                    issues.append(f"missing column '{key}' in {per_dataset_csv}")
            for key in ["PSNR", "SSIM", "ROS", "BES"]:
                if key in headers:
                    issues.append(f"forbidden legacy column '{key}' in {per_dataset_csv}")
        except Exception as e:
            issues.append(f"invalid per_dataset.csv ({e})")
    return issues


def check_failure_explanations(figures_dir: Path) -> list[str]:
    issues: list[str] = []
    explained = figures_dir / "failure_cases" / "failure_cases_explained.csv"
    if not explained.exists():
        return [f"missing failure case explanation csv: {explained}"]

    try:
        with explained.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        return [f"invalid failure_cases_explained.csv ({e})"]

    if not rows:
        issues.append(f"failure_cases_explained.csv has no rows: {explained}")
    for idx, row in enumerate(rows, start=2):
        if not str(row.get("explanation", "")).strip():
            issues.append(f"missing explanation at {explained}:{idx}")
            break
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 gate checker.")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--figures-root", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    config = load_config(args.config)
    output_policy = resolve_output_policy(config)
    wild_min_mean_mask_ratio, wild_min_active_frame_ratio = get_wild_coverage_thresholds(config)
    mandatory = get_mandatory_datasets(config)

    pred_root = args.pred_root / args.exp_id
    metrics_dir = args.metrics_root / args.exp_id
    figures_dir = args.figures_root / args.exp_id

    issues: list[str] = []
    issues.extend(
        check_outputs(
            pred_root=pred_root,
            datasets=mandatory,
            output_policy=output_policy,
            wild_min_mean_mask_ratio=wild_min_mean_mask_ratio,
            wild_min_active_frame_ratio=wild_min_active_frame_ratio,
        )
    )
    issues.extend(check_metrics(metrics_dir=metrics_dir))
    issues.extend(check_failure_explanations(figures_dir=figures_dir))

    if issues:
        print("FAIL: Phase 1 checks failed")
        for i, issue in enumerate(issues, start=1):
            print(f"{i}. {issue}")
        raise SystemExit(1)

    print("PASS: Phase 1 checks passed")
    print(f"exp_id={args.exp_id}, mandatory={mandatory}")


if __name__ == "__main__":
    main()
