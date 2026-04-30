"""VGGT4D-driven dynamic prior generation for Route F (Phase 4)."""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


def _ensure_binary_mask(mask_u8: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    m = np.asarray(mask_u8, dtype=np.uint8)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((m > 127).astype(np.uint8) * 255)


def _zero_masks(frame_count: int, frame_shape: tuple[int, int]) -> list[np.ndarray]:
    h, w = frame_shape
    return [np.zeros((h, w), dtype=np.uint8) for _ in range(frame_count)]


def _compute_mask_stats(masks_u8: list[np.ndarray]) -> dict[str, float]:
    if not masks_u8:
        return {
            "mean_dynamic_ratio": 0.0,
            "active_frame_ratio": 0.0,
        }

    ratios = np.array([float((np.asarray(m) > 0).mean()) for m in masks_u8], dtype=np.float32)
    active = np.array([1.0 if r > 0.0 else 0.0 for r in ratios], dtype=np.float32)
    return {
        "mean_dynamic_ratio": float(ratios.mean()),
        "active_frame_ratio": float(active.mean()),
    }


def build_vggt4d_runner_command(
    *,
    repo_dir: Path,
    script_relpath: str,
    env_name: str,
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    datasets: list[str],
) -> list[str]:
    script_path = (repo_dir / script_relpath).resolve()
    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        str(script_path),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--chunk_size",
        str(max(1, int(chunk_size))),
    ]
    if datasets:
        cmd.extend(["--datasets", ",".join(datasets)])
    return cmd


def _write_dataset_inputs(
    datasets_frames_bgr: dict[str, list[np.ndarray]],
    input_root: Path,
) -> None:
    input_root.mkdir(parents=True, exist_ok=True)
    for ds, frames in datasets_frames_bgr.items():
        ds_dir = input_root / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        for idx, frame in enumerate(frames):
            ok = cv2.imwrite(str(ds_dir / f"{idx:06d}.png"), np.asarray(frame))
            if not ok:
                raise RuntimeError(f"Failed to write temporary frame: dataset={ds}, idx={idx}")


def _load_dataset_masks(
    *,
    mask_dir: Path,
    target_count: int,
    target_shape: tuple[int, int],
) -> list[np.ndarray]:
    mask_paths = sorted(mask_dir.glob("dynamic_mask_*.png"))
    if not mask_paths:
        mask_paths = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No VGGT4D masks found in {mask_dir}")

    out: list[np.ndarray] = []
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        out.append(_ensure_binary_mask(m, target_shape=target_shape))

    if not out:
        raise RuntimeError(f"VGGT4D masks unreadable in {mask_dir}")

    if len(out) > target_count:
        out = out[:target_count]
    while len(out) < target_count:
        h, w = target_shape
        out.append(np.zeros((h, w), dtype=np.uint8))

    return out


def generate_vggt4d_dynamic_priors(
    datasets_frames_bgr: dict[str, list[np.ndarray]],
    output_root: Path,
    cfg: dict[str, Any],
    logger_obj: logging.Logger | None = None,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    log = logger_obj or logger

    strict_backend = bool(cfg.get("strict_backend", True))
    env_name = str(cfg.get("env_name", "vggt4d"))
    repo_dir = (REPO_ROOT / str(cfg.get("repo_dir", "data/external/vggt4d"))).resolve()
    script_relpath = str(cfg.get("script_relpath", "run_vggt4d_chunked.py"))
    chunk_size = int(cfg.get("chunk_size", 40))
    hf_endpoint = str(cfg.get("hf_endpoint", "")).strip()

    if not datasets_frames_bgr:
        return {}, {
            "backend": "vggt4d",
            "env_name": env_name,
            "repo_dir": str(repo_dir),
            "script_relpath": script_relpath,
            "chunk_size": int(chunk_size),
            "strict_backend": strict_backend,
            "datasets": {},
        }

    for ds, frames in datasets_frames_bgr.items():
        if not frames:
            raise ValueError(f"Empty frame list for dataset={ds}")

    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint

    ds_names = list(datasets_frames_bgr.keys())

    with tempfile.TemporaryDirectory(prefix="vggt4d_input_") as tmpdir:
        input_root = Path(tmpdir)
        _write_dataset_inputs(datasets_frames_bgr=datasets_frames_bgr, input_root=input_root)

        cmd = build_vggt4d_runner_command(
            repo_dir=repo_dir,
            script_relpath=script_relpath,
            env_name=env_name,
            input_dir=input_root,
            output_dir=output_root,
            chunk_size=chunk_size,
            datasets=ds_names,
        )

        completed = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    if completed.returncode != 0:
        tail = ((completed.stderr or "") + "\n" + (completed.stdout or ""))[-2000:]
        if strict_backend:
            raise RuntimeError(f"VGGT4D prior generation failed in strict mode: {tail}")
        log.warning("VGGT4D prior generation failed in non-strict mode, use zero masks: %s", tail)

    priors: dict[str, list[np.ndarray]] = {}
    ds_meta: dict[str, dict[str, Any]] = {}

    for ds, frames in datasets_frames_bgr.items():
        h, w = frames[0].shape[:2]
        target_count = len(frames)

        masks: list[np.ndarray]
        if completed.returncode == 0:
            try:
                masks = _load_dataset_masks(
                    mask_dir=output_root / ds,
                    target_count=target_count,
                    target_shape=(h, w),
                )
            except Exception as e:
                if strict_backend:
                    raise RuntimeError(f"VGGT4D output loading failed for {ds}: {e}")
                log.warning("VGGT4D output loading failed for %s, use zero masks: %s", ds, e)
                masks = _zero_masks(target_count, (h, w))
        else:
            masks = _zero_masks(target_count, (h, w))

        priors[ds] = masks
        ds_meta[ds] = {
            "frame_count": int(target_count),
            "output_dir": str(output_root / ds),
            **_compute_mask_stats(masks),
        }

    meta = {
        "backend": "vggt4d",
        "env_name": env_name,
        "repo_dir": str(repo_dir),
        "script_relpath": script_relpath,
        "chunk_size": int(chunk_size),
        "strict_backend": strict_backend,
        "hf_endpoint": hf_endpoint,
        "runner_returncode": int(completed.returncode),
        "datasets": ds_meta,
    }
    return priors, meta
