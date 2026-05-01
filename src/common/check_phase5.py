#!/usr/bin/env python3
"""Phase 5 gate checker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.video_io import (
    compute_mask_coverage_from_dir_or_video,
    dataset_video_paths,
    resolve_output_policy,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_mandatory_datasets(config: dict) -> list[str]:
    mandatory = (config.get("datasets", {}) or {}).get("mandatory", {}) or {}
    return list(mandatory.keys())


def check_outputs(pred_root: Path, datasets: list[str], output_policy: dict) -> list[str]:
    issues: list[str] = []
    threshold = int(((output_policy.get("mask_h264", {}) or {}).get("threshold", 127)))
    for ds in datasets:
        ds_root = pred_root / ds
        if not ds_root.exists():
            issues.append(f"missing output directory: {ds_root}")
            continue
        restored, mask_video = dataset_video_paths(ds_root, output_policy)
        if not restored.exists():
            issues.append(f"missing restored video for {ds}: {restored}")
        if not mask_video.exists():
            issues.append(f"missing mask video for {ds}: {mask_video}")
        coverage = compute_mask_coverage_from_dir_or_video(
            mask_dir=ds_root / "masks",
            mask_video_path=mask_video,
            threshold=threshold,
        )
        if int(coverage.get("frame_count", 0)) <= 0:
            issues.append(f"{ds}: unable to load masks")
        elif float(coverage.get("active_frame_ratio", 0.0)) <= 0.0:
            issues.append(f"{ds}: all predicted masks are zero")
    return issues


def check_metrics(metrics_dir: Path) -> list[str]:
    issues: list[str] = []
    required = [
        "summary.json",
        "per_dataset.csv",
        "phase5_ablation.csv",
        "phase5_selection.json",
        "phase5_b_vs_g.csv",
        "phase5_run_meta.json",
        "phase5_acceptance_report.md",
    ]
    for name in required:
        if not (metrics_dir / name).exists():
            issues.append(f"missing required file: {metrics_dir / name}")

    summary_path = metrics_dir / "summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            agg = summary.get("aggregate", {}) or {}
            for key in ["JM", "JR", "ROS", "TCF", "BES", "Q_REMOVE"]:
                if key not in agg:
                    issues.append(f"summary.json missing aggregate metric: {key}")
        except Exception as e:
            issues.append(f"invalid summary.json: {e}")

    run_meta_path = metrics_dir / "phase5_run_meta.json"
    if run_meta_path.exists():
        try:
            with run_meta_path.open("r", encoding="utf-8") as f:
                run_meta = json.load(f)
            if not bool(run_meta.get("has_g_variants", False)):
                issues.append("phase5_run_meta.json has_g_variants must be true")
            valid_variants = {"G-low", "G-mid", "G-high", "G-hybrid"}
            if run_meta.get("g_final_variant") not in valid_variants:
                issues.append(
                    f"phase5_run_meta.json g_final_variant must be one of {valid_variants}, "
                    f"got: {run_meta.get('g_final_variant')}"
                )
        except Exception as e:
            issues.append(f"invalid phase5_run_meta.json: {e}")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 (Route G) gate checker.")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = load_config(config_path)
    mandatory = get_mandatory_datasets(cfg)
    output_policy = resolve_output_policy(cfg)

    pred_root = args.pred_root / args.exp_id
    if not args.pred_root.is_absolute():
        pred_root = REPO_ROOT / args.pred_root / args.exp_id
    metrics_dir = args.metrics_root / args.exp_id
    if not args.metrics_root.is_absolute():
        metrics_dir = REPO_ROOT / args.metrics_root / args.exp_id

    issues: list[str] = []
    issues.extend(check_outputs(pred_root=pred_root, datasets=mandatory, output_policy=output_policy))
    issues.extend(check_metrics(metrics_dir=metrics_dir))

    if issues:
        print("FAIL: Phase5 checks failed")
        for idx, item in enumerate(issues, start=1):
            print(f"  {idx}. {item}")
        raise SystemExit(1)

    print("PASS: Phase5 checks passed")
    print(f"exp_id={args.exp_id}, mandatory={mandatory}")


if __name__ == "__main__":
    main()
