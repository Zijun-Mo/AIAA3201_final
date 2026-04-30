#!/usr/bin/env python3
"""Run VGGT4D, organize masks, then evaluate JM/JR."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data/raw"
VGGT4D_REPO = REPO_ROOT / "data/external/vggt4d"
GT_ROOT = REPO_ROOT / "data/gt"
OUT_PRED = REPO_ROOT / "outputs/vggt4d_eval"
CHUNK_SIZES = {"bmx-trees": 40, "tennis": 40, "wild": 10}


def extract_frames(video_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(out_dir / f"frame_{idx:06d}.png"), frame)
        idx += 1
    cap.release()


def main():
    with tempfile.TemporaryDirectory(prefix="vggt4d_eval_") as tmpdir:
        input_root = Path(tmpdir) / "input"
        mask_tmp = Path(tmpdir) / "masks"

        for name in CHUNK_SIZES:
            print(f"Extracting {name}...")
            extract_frames(RAW_DIR / f"{name}.mp4", input_root / name)

        for name, chunk_size in CHUNK_SIZES.items():
            print(f"Running VGGT4D on {name} (chunk_size={chunk_size})...")
            cmd = [
                "conda", "run", "-n", "vggt4d", "python",
                str(VGGT4D_REPO / "run_vggt4d_chunked.py"),
                "--input_dir", str(input_root),
                "--output_dir", str(mask_tmp),
                "--chunk_size", str(chunk_size),
                "--datasets", name,
            ]
            r = subprocess.run(cmd, cwd=str(VGGT4D_REPO))
            if r.returncode != 0:
                print(f"VGGT4D failed for {name}")
                sys.exit(1)

            # rename dynamic_mask_XXXXXX.png -> frame_XXXXXX.png and copy to pred_root
            pred_mask_dir = OUT_PRED / name / "masks"
            pred_mask_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted((mask_tmp / name).glob("dynamic_mask_*.png")):
                idx_str = p.stem.replace("dynamic_mask_", "")
                shutil.copy(p, pred_mask_dir / f"frame_{idx_str}.png")

            # also copy frames for evaluate_experiment
            pred_frame_dir = OUT_PRED / name / "frames"
            pred_frame_dir.mkdir(parents=True, exist_ok=True)
            for p in sorted((input_root / name).glob("frame_*.png")):
                shutil.copy(p, pred_frame_dir / p.name)

    print("Running evaluation...")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src/common/evaluate_experiment.py"),
            "--pred-root", str(OUT_PRED),
            "--gt-root", str(GT_ROOT),
            "--exp-id", "vggt4d_only",
            "--output-dir", str(REPO_ROOT / "outputs/metrics/vggt4d_only"),
        ],
        cwd=str(REPO_ROOT),
    )
    if result.returncode != 0:
        sys.exit(1)

    summary = REPO_ROOT / "outputs/metrics/vggt4d_only/summary.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        agg = data.get("aggregate", {})
        print(f"\nJM = {agg.get('JM', 'N/A'):.4f}")
        print(f"JR = {agg.get('JR', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
