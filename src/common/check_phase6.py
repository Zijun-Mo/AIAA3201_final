#!/usr/bin/env python3
"""Phase 6 gate checker."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.video_io import compute_mask_coverage_from_dir_or_video, dataset_video_paths, resolve_output_policy


def str2bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = value.strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid bool value: {value}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_mandatory_datasets(config: dict[str, Any]) -> list[str]:
    mandatory = (config.get("datasets", {}) or {}).get("mandatory", {}) or {}
    return list(mandatory.keys())


def check_outputs(pred_root: Path, datasets: list[str], output_policy: dict[str, Any]) -> list[str]:
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


def check_metrics(metrics_dir: Path) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    required = [
        "summary.json",
        "per_dataset.csv",
        "phase6_ablation.csv",
        "phase6_selection.json",
        "phase6_b_vs_h.csv",
        "phase6_efh_jmjr.csv",
        "phase6_pareto.csv",
        "phase6_run_meta.json",
        "phase6_acceptance_report.md",
    ]
    for name in required:
        p = metrics_dir / name
        if not p.exists():
            issues.append(f"missing required file: {p}")

    summary: dict[str, Any] = {}
    summary_path = metrics_dir / "summary.json"
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            agg = summary.get("aggregate", {}) or {}
            for key in ["GT_Coverage", "JM", "JR", "MaskScore", "TCF", "FAST_VQA"]:
                if key not in agg:
                    issues.append(f"summary.json missing aggregate metric: {key}")
            for key in ["ROS", "BES"]:
                if key in agg:
                    issues.append(f"summary.json contains removed metric: {key}")
        except Exception as e:
            issues.append(f"invalid summary.json: {e}")

    run_meta_path = metrics_dir / "phase6_run_meta.json"
    if run_meta_path.exists():
        try:
            with run_meta_path.open("r", encoding="utf-8") as f:
                run_meta = json.load(f)
            if run_meta.get("phase_label") != "phase6":
                issues.append("phase6_run_meta.json phase_label must be phase6")
            if not bool(run_meta.get("has_phase6_stages", False)):
                issues.append("phase6_run_meta.json has_phase6_stages must be true")
            if run_meta.get("phase6_final_policy") != "global_non_oracle_maskscore":
                issues.append("phase6_run_meta.json phase6_final_policy must be global_non_oracle_maskscore")
            if not bool(run_meta.get("phase6_best_is_global_non_oracle", False)):
                issues.append("phase6_run_meta.json phase6_best_is_global_non_oracle must be true")
            phase6_best = run_meta.get("phase6_best", {}) or {}
            if phase6_best.get("stage") not in {"H0", "H1"}:
                issues.append(f"phase6_best.stage must be one of H0/H1, got: {phase6_best.get('stage')}")
            sam3_perm = run_meta.get("sam3_permission", {}) or {}
            if not bool(sam3_perm.get("checked", False)):
                issues.append("sam3_permission.checked must be true for Phase 6")
            selection_datasets = set(run_meta.get("selection_datasets", []) or [])
            datasets_meta = (summary.get("datasets", {}) or {}) if isinstance(summary, dict) else {}
            gt_missing_selected = [
                ds for ds in sorted(selection_datasets) if (datasets_meta.get(ds, {}) or {}).get("status") == "gt_missing"
            ]
            if gt_missing_selected:
                issues.append(
                    "Phase 6 MaskScore selection_datasets must exclude GT-missing datasets: "
                    + ",".join(gt_missing_selected)
                )
        except Exception as e:
            issues.append(f"invalid phase6_run_meta.json: {e}")

    efh_path = metrics_dir / "phase6_efh_jmjr.csv"
    if efh_path.exists():
        try:
            with efh_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if "GT_Coverage" not in (reader.fieldnames or []):
                    issues.append("phase6_efh_jmjr.csv missing GT_Coverage column")
                if "FAST_VQA" not in (reader.fieldnames or []):
                    issues.append("phase6_efh_jmjr.csv missing FAST_VQA column")
                rows = list(reader)
            by_method = {row.get("method", ""): row for row in rows}
            for method in ["B-best", "E-best", "F-best", "Phase6-best"]:
                if method not in by_method:
                    issues.append(f"phase6_efh_jmjr.csv missing method row: {method}")
            h_row = by_method.get("Phase6-best", {})
            e_row = by_method.get("E-best", {})
            f_row = by_method.get("F-best", {})
            try:
                h_score = float(h_row.get("MaskScore", "nan"))
                e_score = float(e_row.get("MaskScore", "nan"))
                f_score = float(f_row.get("MaskScore", "nan"))
                ref_score = max(e_score, f_score)
                if math.isfinite(ref_score) and h_score < ref_score:
                    warnings.append(
                        f"Phase6-best MaskScore {h_score:.6f} is below max(E-best,F-best) {ref_score:.6f}; report the stacked core change as not improving both baselines"
                    )
            except ValueError:
                pass
        except Exception as e:
            issues.append(f"invalid phase6_efh_jmjr.csv: {e}")

    for table_name in ["phase6_ablation.csv", "phase6_b_vs_h.csv", "phase6_pareto.csv"]:
        table_path = metrics_dir / table_name
        if table_path.exists():
            try:
                with table_path.open("r", encoding="utf-8", newline="") as f:
                    fieldnames = csv.DictReader(f).fieldnames or []
                if not any("GT_Coverage" == name or name.endswith("_GT_Coverage") for name in fieldnames):
                    issues.append(f"{table_name} missing GT_Coverage field")
                if "MaskScore" not in fieldnames and not any(name.endswith("_MaskScore") for name in fieldnames):
                    issues.append(f"{table_name} missing MaskScore field")
                if not any("FAST_VQA" == name or name.endswith("_FAST_VQA") for name in fieldnames):
                    issues.append(f"{table_name} missing FAST_VQA field")
            except Exception as e:
                issues.append(f"invalid {table_name}: {e}")

    return issues, warnings


def check_sam3_permission(metrics_dir: Path, strict_sam3_permission: bool) -> list[str]:
    if not strict_sam3_permission:
        return []
    run_meta_path = metrics_dir / "phase6_run_meta.json"
    if not run_meta_path.exists():
        return [f"missing run meta for SAM3 permission check: {run_meta_path}"]
    try:
        with run_meta_path.open("r", encoding="utf-8") as f:
            run_meta = json.load(f)
    except Exception as e:
        return [f"invalid phase6_run_meta.json ({e})"]
    sam3_perm = run_meta.get("sam3_permission", {}) or {}
    if not bool(sam3_perm.get("passed", False)):
        return ["sam3_permission.passed is false under strict mode"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 (E/F core stacking) gate checker.")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs/metrics"))
    parser.add_argument("--strict-sam3-permission", type=str, default="true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = load_config(config_path)
    mandatory = get_mandatory_datasets(cfg)
    output_policy = resolve_output_policy(cfg)
    strict_sam3_permission = str2bool(args.strict_sam3_permission, default=True)

    pred_root = args.pred_root / args.exp_id
    if not args.pred_root.is_absolute():
        pred_root = REPO_ROOT / args.pred_root / args.exp_id
    metrics_dir = args.metrics_root / args.exp_id
    if not args.metrics_root.is_absolute():
        metrics_dir = REPO_ROOT / args.metrics_root / args.exp_id

    issues: list[str] = []
    warnings: list[str] = []
    issues.extend(check_outputs(pred_root=pred_root, datasets=mandatory, output_policy=output_policy))
    metric_issues, metric_warnings = check_metrics(metrics_dir=metrics_dir)
    issues.extend(metric_issues)
    warnings.extend(metric_warnings)
    issues.extend(check_sam3_permission(metrics_dir=metrics_dir, strict_sam3_permission=strict_sam3_permission))

    if issues:
        print("FAIL: Phase 6 checks failed")
        for idx, item in enumerate(issues, start=1):
            print(f"  {idx}. {item}")
        raise SystemExit(1)

    print("PASS: Phase 6 checks passed")
    print(f"exp_id={args.exp_id}, mandatory={mandatory}, strict_sam3_permission={strict_sam3_permission}")
    for warning in warnings:
        print(f"WARN: {warning}")


if __name__ == "__main__":
    main()
