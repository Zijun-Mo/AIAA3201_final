#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import cv2
import yaml


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


def check_outputs(pred_root: Path, datasets: list[str]) -> list[str]:
    issues: list[str] = []
    for ds in datasets:
        frame_dir = pred_root / ds / "frames"
        mask_dir = pred_root / ds / "masks"
        frames = list_images(frame_dir)
        masks = list_images(mask_dir)

        if not frames:
            issues.append(f"{ds}: missing/empty frames dir {frame_dir}")
        if not masks:
            issues.append(f"{ds}: missing/empty masks dir {mask_dir}")
        if frames and masks and len(frames) != len(masks):
            issues.append(f"{ds}: frame/mask count mismatch ({len(frames)} vs {len(masks)})")

    # Explicit Phase 1 gate: wild final masks must not be all-zero.
    wild_masks = list_images(pred_root / "wild" / "masks")
    if wild_masks:
        has_nonzero = False
        for p in wild_masks:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if int((m > 0).sum()) > 0:
                has_nonzero = True
                break
        if not has_nonzero:
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
        except Exception as e:
            issues.append(f"invalid summary.json ({e})")
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
    mandatory = get_mandatory_datasets(config)

    pred_root = args.pred_root / args.exp_id
    metrics_dir = args.metrics_root / args.exp_id
    figures_dir = args.figures_root / args.exp_id

    issues: list[str] = []
    issues.extend(check_outputs(pred_root=pred_root, datasets=mandatory))
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

