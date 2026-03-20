#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_datasets(config: dict) -> tuple[dict, list[str]]:
    datasets_cfg = config.get("datasets", {})
    mandatory = datasets_cfg.get("mandatory", {}) or {}
    optional = datasets_cfg.get("optional", {}) or {}

    if not isinstance(mandatory, dict) or not isinstance(optional, dict):
        raise ValueError("'datasets.mandatory' and 'datasets.optional' must be mappings.")

    all_map = {**mandatory, **optional}
    mandatory_names = list(mandatory.keys())
    return all_map, mandatory_names


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


def clean_existing_frames(frames_dir: Path, frame_format: str) -> None:
    pattern = f"*.{frame_format}"
    for p in frames_dir.glob(pattern):
        p.unlink()


def build_frame_name(template: str, index: int, frame_format: str) -> str:
    try:
        base = template % index
    except TypeError as e:
        raise ValueError(
            "Invalid frame_name_template. Expected printf style like 'frame_%06d'."
        ) from e

    if "." in Path(base).name:
        return base
    return f"{base}.{frame_format}"


def preprocess_one_dataset(name: str, ds_cfg: dict, preprocess_cfg: dict, overwrite: bool) -> dict:
    raw_video = Path(ds_cfg.get("raw_video", ""))
    frames_dir = Path(ds_cfg.get("processed_frames_dir", ""))

    if not raw_video:
        raise ValueError(f"Dataset '{name}' missing 'raw_video' in config.")
    if not frames_dir:
        raise ValueError(f"Dataset '{name}' missing 'processed_frames_dir' in config.")

    if not raw_video.exists():
        raise FileNotFoundError(
            f"Dataset '{name}': raw video not found: {raw_video}. "
            "Please place video file according to config."
        )

    target_fps = float(preprocess_cfg.get("target_fps", 24))
    target_width = int(preprocess_cfg.get("target_width", 1280))
    target_height = int(preprocess_cfg.get("target_height", 720))
    frame_format = str(preprocess_cfg.get("frame_format", "png")).lower()
    frame_template = str(preprocess_cfg.get("frame_name_template", "frame_%06d"))

    if frame_format not in {"png", "jpg", "jpeg"}:
        raise ValueError(f"Unsupported frame_format '{frame_format}'. Use png/jpg/jpeg.")

    frames_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        clean_existing_frames(frames_dir, frame_format)

    cap = cv2.VideoCapture(str(raw_video))
    if not cap.isOpened():
        raise RuntimeError(
            f"Dataset '{name}': unable to open video '{raw_video}'. "
            "The file may be corrupted or format unsupported."
        )

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if source_fps <= 0:
        source_fps = target_fps if target_fps > 0 else 24.0

    time_step = 1.0 / target_fps if target_fps > 0 else 0.0
    next_time = 0.0

    src_index = 0
    saved_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_time = src_index / source_fps
        should_save = time_step == 0.0 or current_time + 1e-9 >= next_time

        if should_save:
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            frame_name = build_frame_name(frame_template, saved_index, frame_format)
            out_path = frames_dir / frame_name
            if not cv2.imwrite(str(out_path), resized):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            saved_index += 1
            if time_step > 0:
                next_time += time_step

        src_index += 1

    cap.release()

    if saved_index == 0:
        raise RuntimeError(
            f"Dataset '{name}': zero frames generated. "
            "Check fps settings or source video integrity."
        )

    manifest_path = frames_dir.parent / "manifest.json"
    manifest = {
        "dataset": name,
        "raw_video": str(raw_video),
        "processed_frames_dir": str(frames_dir),
        "source_fps": source_fps,
        "target_fps": target_fps,
        "source_frame_count": source_frame_count,
        "saved_frame_count": saved_index,
        "source_resolution": [source_width, source_height],
        "target_resolution": [target_width, target_height],
        "frame_format": frame_format,
        "frame_name_template": frame_template,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 preprocessing: normalize videos into standardized frame folders."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--datasets",
        type=str,
        default="mandatory",
        help="mandatory | all | comma-separated dataset names",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing frames with target format before writing new frames.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    all_map, mandatory_names = collect_datasets(config)
    all_names = list(all_map.keys())
    selected_names = resolve_dataset_names(args.datasets, all_names, mandatory_names)

    preprocess_cfg = config.get("preprocess", {})

    manifests = []
    for name in selected_names:
        manifest = preprocess_one_dataset(
            name=name,
            ds_cfg=all_map[name],
            preprocess_cfg=preprocess_cfg,
            overwrite=args.overwrite,
        )
        manifests.append(manifest)
        print(
            f"[OK] {name}: frames={manifest['saved_frame_count']} "
            f"target={manifest['target_resolution']} fps={manifest['target_fps']}"
        )

    print("[DONE] Phase 0 preprocessing complete.")
    print(json.dumps({"processed_datasets": [m["dataset"] for m in manifests]}, indent=2))


if __name__ == "__main__":
    main()
