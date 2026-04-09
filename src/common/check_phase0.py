#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import yaml


REQUIRED_MANIFEST_KEYS = {
    "dataset",
    "raw_video",
    "processed_frames_dir",
    "target_fps",
    "target_resolution",
    "saved_frame_count",
    "frame_format",
    "frame_name_template",
    "generated_at_utc",
}


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_mandatory_datasets(config: dict) -> dict:
    datasets = config.get("datasets", {}).get("mandatory", {}) or {}
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("No mandatory datasets found under configs.base.yaml -> datasets.mandatory")
    return datasets


def check_manifest(dataset: str, manifest_path: Path) -> list[str]:
    issues = []
    if not manifest_path.exists():
        return [f"{dataset}: missing manifest {manifest_path}"]

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:
        return [f"{dataset}: invalid manifest JSON ({e})"]

    missing = sorted(list(REQUIRED_MANIFEST_KEYS - set(m.keys())))
    if missing:
        issues.append(f"{dataset}: manifest missing keys {missing}")

    frame_count = int(m.get("saved_frame_count", 0) or 0)
    if frame_count <= 0:
        issues.append(f"{dataset}: saved_frame_count must be > 0")

    return issues


def check_evaluation_outputs(exp_id: str, mandatory_names: list[str]) -> list[str]:
    issues = []
    metrics_dir = Path("outputs/metrics") / exp_id
    summary_path = metrics_dir / "summary.json"
    csv_path = metrics_dir / "per_dataset.csv"
    figures_dir = Path("outputs/figures") / exp_id

    if not summary_path.exists():
        issues.append(f"missing evaluation summary: {summary_path}")
    if not csv_path.exists():
        issues.append(f"missing per_dataset csv: {csv_path}")
    if not figures_dir.exists():
        issues.append(f"missing figures directory: {figures_dir}")

    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            ds_keys = set((summary.get("datasets") or {}).keys())
            missing_ds = [d for d in mandatory_names if d not in ds_keys]
            if missing_ds:
                issues.append(f"summary missing mandatory datasets: {missing_ds}")
        except Exception as e:
            issues.append(f"invalid summary.json ({e})")

    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                ds_set = {row.get("dataset", "") for row in reader}
            missing_ds = [d for d in mandatory_names if d not in ds_set]
            if missing_ds:
                issues.append(f"per_dataset.csv missing mandatory datasets: {missing_ds}")
        except Exception as e:
            issues.append(f"invalid per_dataset.csv ({e})")

    if figures_dir.exists():
        for d in mandatory_names:
            sample_pred = figures_dir / d / "sample_pred.png"
            if not sample_pred.exists():
                issues.append(f"missing sample visualization: {sample_pred}")

    return issues


def suggestion_for(issue: str) -> str:
    if "missing manifest" in issue or "saved_frame_count" in issue:
        return "Run: bash scripts/preprocess.sh --datasets mandatory --overwrite"
    if "summary" in issue or "per_dataset" in issue or "figures" in issue:
        return "Run: bash scripts/evaluate.sh --exp-id <exp_id> --datasets mandatory --pred-root data/processed --gt-root data/gt --allow-missing-gt true"
    return "Check config paths and generated outputs, then rerun Phase 0 scripts."


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 gate checker (no model inference).")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--exp-id", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    mandatory = get_mandatory_datasets(config)

    issues = []

    # Basic required directories
    for d in [
        Path("data/raw"),
        Path("data/processed"),
        Path("data/gt"),
        Path("outputs/metrics"),
        Path("outputs/figures"),
        Path("outputs/logs"),
    ]:
        if not d.exists():
            issues.append(f"missing required directory: {d}")

    # Manifest checks
    for name, ds_cfg in mandatory.items():
        frames_dir = Path(ds_cfg.get("processed_frames_dir", ""))
        manifest_path = frames_dir.parent / "manifest.json"
        issues.extend(check_manifest(name, manifest_path))

    # Evaluation outputs checks
    issues.extend(check_evaluation_outputs(args.exp_id, list(mandatory.keys())))

    if issues:
        print("FAIL: Phase 0 checks failed")
        for idx, issue in enumerate(issues, start=1):
            print(f"{idx}. {issue}")
        print("\nSuggested fixes:")
        uniq = []
        for issue in issues:
            s = suggestion_for(issue)
            if s not in uniq:
                uniq.append(s)
        for idx, s in enumerate(uniq, start=1):
            print(f"{idx}. {s}")
        raise SystemExit(1)

    print("PASS: Phase 0 checks passed")
    print(f"Checked exp_id={args.exp_id} with mandatory datasets={list(mandatory.keys())}")


if __name__ == "__main__":
    main()
