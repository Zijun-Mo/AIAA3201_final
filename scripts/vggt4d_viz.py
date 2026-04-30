#!/usr/bin/env python3
"""Run VGGT4D on raw videos and overlay masks onto output videos."""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data/raw"
VGGT4D_REPO = REPO_ROOT / "data/external/vggt4d"
OUT_DIR = REPO_ROOT / "outputs/vggt4d_viz"
VIDEOS = {"bmx-trees": 40, "tennis": 40, "wild": 10}


def extract_frames(video_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = out_dir / f"{idx:06d}.png"
        cv2.imwrite(str(p), frame)
        paths.append(p)
        idx += 1
    cap.release()
    return paths


def overlay_masks(frame_dir: Path, mask_dir: Path, video_out: Path, fps: float) -> None:
    frame_paths = sorted(frame_dir.glob("*.png"))
    mask_paths = sorted(mask_dir.glob("dynamic_mask_*.png"))
    if not mask_paths:
        mask_paths = sorted(mask_dir.glob("*.png"))

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(str(fp))
        if i < len(mask_paths):
            mask = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay = frame.copy()
                overlay[mask > 127] = (overlay[mask > 127] * 0.4 + np.array([0, 0, 200]) * 0.6).astype(np.uint8)
                frame = overlay
        writer.write(frame)

    writer.release()


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="vggt4d_frames_") as tmpdir:
        input_root = Path(tmpdir) / "input"
        mask_root = Path(tmpdir) / "masks"

        fps_map = {}
        for name in VIDEOS:
            vp = RAW_DIR / f"{name}.mp4"
            fps_map[name] = get_fps(vp)
            print(f"Extracting {name} ({fps_map[name]:.1f} fps)...")
            extract_frames(vp, input_root / name)

        for name, chunk_size in VIDEOS.items():
            print(f"Running VGGT4D on {name} (chunk_size={chunk_size})...")
            cmd = [
                "conda", "run", "-n", "vggt4d", "python",
                str(VGGT4D_REPO / "run_vggt4d_chunked.py"),
                "--input_dir", str(input_root),
                "--output_dir", str(mask_root),
                "--chunk_size", str(chunk_size),
                "--datasets", name,
            ]
            result = subprocess.run(cmd, cwd=str(VGGT4D_REPO), capture_output=False, text=True)
            if result.returncode != 0:
                print(f"  WARNING: VGGT4D failed for {name}, skipping overlay.")
                continue

            out_video = OUT_DIR / f"{name}_vggt4d.mp4"
            print(f"Overlaying masks for {name}...")
            overlay_masks(input_root / name, mask_root / name, out_video, fps_map[name])
            print(f"  -> {out_video}")

    print("Done.")


if __name__ == "__main__":
    main()
