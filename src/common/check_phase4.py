#!/usr/bin/env python3
"""Phase 4 gate checker."""
from __future__ import annotations

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
    dataset_video_paths,
    resolve_output_policy,
)


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_mandatory_datasets(config: dict) -> list[str]:
    mandatory = (config.get("datasets", {}) or {}).get("mandatory", {}) or {}
    if not isinstance(mandatory, dict) or not mandatory:
        raise ValueError("No mandatory datasets in datasets.mandatory")
    return list(mandatory.keys())


def check_outputs(pred_root: Path, datasets: list[str], output_policy: dict) -> list[str]:
    issues: list[str] = []
    threshold = int(((output_policy.get("mask_h264", {}) or {}).get("threshold", 127)))
    for ds in datasets:
        ds_root = pred_root / ds
        if not ds_root.exists():
            issues.append(f"missing output dataset directory: {ds_root}")
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
            issues.append(f"{ds}: unable to load masks for non-zero check")
        elif float(coverage.get("active_frame_ratio", 0.0)) <= 0.0:
            issues.append(f"{ds}: all predicted masks are zero")
    return issues


def check_metrics(metrics_dir: Path) -> list[str]:
    issues: list[str] = []
    required = [
        metrics_dir / "summary.json",
        metrics_dir / "per_dataset.csv",
        metrics_dir / "phase4_ablation.csv",
        metrics_dir / "phase4_selection.json",
        metrics_dir / "phase4_b_vs_f.csv",
        metrics_dir / "phase4_mask_priors.csv",
        metrics_dir / "phase4_acceptance_report.md",
        metrics_dir / "phase4_run_meta.json",
    ]
    for p in required:
        if not p.exists():
            issues.append(f"missing required file: {p}")

    summary_path = metrics_dir / "summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            agg = summary.get("aggregate", {}) or {}
            for key in ["JM", "JR", "ROS", "TCF", "BES"]:
                if key not in agg:
                    issues.append(f"summary missing aggregate metric: {key}")
        except Exception as e:
            issues.append(f"invalid summary.json ({e})")

    priors_csv = metrics_dir / "phase4_mask_priors.csv"
    if priors_csv.exists():
        try:
            with priors_csv.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            needed = {
                (row.get("dataset", ""), row.get("prior", ""))
                for row in rows
            }
            for ds in ["wild", "bmx-trees", "tennis"]:
                for prior in ["yolo", "vggt4d", "vggt4d_yolo"]:
                    if (ds, prior) not in needed:
                        issues.append(f"phase4_mask_priors.csv missing row: dataset={ds}, prior={prior}")
            for row in rows:
                if row.get("prior", "") == "vggt4d" and not row.get("source", "").startswith("vggt4d_with_bbest_backend_"):
                    issues.append(
                        "phase4_mask_priors.csv vggt4d rows must use source=vggt4d_with_bbest_backend_<backend>"
                    )
                    break
        except Exception as e:
            issues.append(f"invalid phase4_mask_priors.csv ({e})")

    run_meta_path = metrics_dir / "phase4_run_meta.json"
    if run_meta_path.exists():
        try:
            with run_meta_path.open("r", encoding="utf-8") as f:
                run_meta = json.load(f)
            if not bool(run_meta.get("has_f_stages", False)):
                issues.append("phase4_run_meta.json has_f_stages=false")
            if run_meta.get("phase4_final_policy") != "force_vggt4d_prior":
                issues.append("phase4_run_meta.json phase4_final_policy must be force_vggt4d_prior")
            for key in ["final_best", "f_best"]:
                spec = run_meta.get(key, {}) or {}
                if spec.get("f_source_key", "") != "vggt4d":
                    issues.append(f"phase4_run_meta.json {key}.f_source_key must be vggt4d")
            if run_meta.get("phase4_bbest_backend") not in {"sam2", "trackanything"}:
                issues.append("phase4_run_meta.json phase4_bbest_backend must be sam2 or trackanything")
        except Exception as e:
            issues.append(f"invalid phase4_run_meta.json ({e})")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 (Route F) gate checker.")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/metrics"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    mandatory = get_mandatory_datasets(cfg)
    output_policy = resolve_output_policy(cfg)

    pred_root = args.pred_root / args.exp_id
    metrics_dir = args.metrics_root / args.exp_id

    issues: list[str] = []
    issues.extend(check_outputs(pred_root=pred_root, datasets=mandatory, output_policy=output_policy))
    issues.extend(check_metrics(metrics_dir=metrics_dir))

    if issues:
        print("FAIL: Phase4 checks failed")
        for idx, item in enumerate(issues, start=1):
            print(f"  {idx}. {item}")
        raise SystemExit(1)

    print("PASS: Phase4 checks passed")
    print(f"exp_id={args.exp_id}, mandatory={mandatory}")


if __name__ == "__main__":
    main()
