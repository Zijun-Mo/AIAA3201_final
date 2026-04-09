from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_OUTPUT_POLICY: dict[str, Any] = {
    "video_only": True,
    "write_h264_videos": True,
    "auto_cleanup_intermediates": True,
    "restored_video_name": "restored_h264.mp4",
    "mask_video_name": "mask_h264.mp4",
    "restored_h264": {
        "crf": 23,
        "preset": "medium",
        "pix_fmt": "yuv420p",
    },
    "mask_h264": {
        "crf": 0,
        "preset": "medium",
        "pix_fmt": "gray",
        "threshold": 127,
    },
    "cleanup": {
        "remove_candidate_root": True,
        "remove_frame_dirs": True,
        "remove_mask_dirs": True,
    },
}


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y"}:
        return True
    if token in {"0", "false", "no", "n"}:
        return False
    return default


def resolve_output_policy(config: dict[str, Any] | None) -> dict[str, Any]:
    src: dict[str, Any] = {}
    if isinstance(config, dict):
        if isinstance(config.get("output_policy"), dict):
            src = config.get("output_policy", {}) or {}
        else:
            src = config

    restored_src = src.get("restored_h264", {}) or {}
    mask_src = src.get("mask_h264", {}) or {}
    cleanup_src = src.get("cleanup", {}) or {}
    restored_default = DEFAULT_OUTPUT_POLICY.get("restored_h264", {}) or {}
    mask_default = DEFAULT_OUTPUT_POLICY.get("mask_h264", {}) or {}
    cleanup_default = DEFAULT_OUTPUT_POLICY.get("cleanup", {}) or {}

    policy = {
        "video_only": _to_bool(src.get("video_only"), bool(DEFAULT_OUTPUT_POLICY["video_only"])),
        "write_h264_videos": _to_bool(
            src.get("write_h264_videos"),
            bool(DEFAULT_OUTPUT_POLICY["write_h264_videos"]),
        ),
        "auto_cleanup_intermediates": _to_bool(
            src.get("auto_cleanup_intermediates"),
            bool(DEFAULT_OUTPUT_POLICY["auto_cleanup_intermediates"]),
        ),
        "restored_video_name": str(
            src.get("restored_video_name", DEFAULT_OUTPUT_POLICY["restored_video_name"])
        ).strip()
        or str(DEFAULT_OUTPUT_POLICY["restored_video_name"]),
        "mask_video_name": str(src.get("mask_video_name", DEFAULT_OUTPUT_POLICY["mask_video_name"])).strip()
        or str(DEFAULT_OUTPUT_POLICY["mask_video_name"]),
        "restored_h264": {
            "crf": int(restored_src.get("crf", restored_default.get("crf", 23))),
            "preset": str(restored_src.get("preset", restored_default.get("preset", "medium"))).strip()
            or "medium",
            "pix_fmt": str(restored_src.get("pix_fmt", restored_default.get("pix_fmt", "yuv420p"))).strip()
            or "yuv420p",
        },
        "mask_h264": {
            "crf": int(mask_src.get("crf", mask_default.get("crf", 0))),
            "preset": str(mask_src.get("preset", mask_default.get("preset", "medium"))).strip() or "medium",
            "pix_fmt": str(mask_src.get("pix_fmt", mask_default.get("pix_fmt", "gray"))).strip() or "gray",
            "threshold": int(mask_src.get("threshold", mask_default.get("threshold", 127))),
        },
        "cleanup": {
            "remove_candidate_root": _to_bool(
                cleanup_src.get("remove_candidate_root"),
                bool(cleanup_default.get("remove_candidate_root", True)),
            ),
            "remove_frame_dirs": _to_bool(
                cleanup_src.get("remove_frame_dirs"),
                bool(cleanup_default.get("remove_frame_dirs", True)),
            ),
            "remove_mask_dirs": _to_bool(
                cleanup_src.get("remove_mask_dirs"),
                bool(cleanup_default.get("remove_mask_dirs", True)),
            ),
        },
    }
    if policy["video_only"] and not policy["write_h264_videos"]:
        policy["video_only"] = False
    return policy


def dataset_video_paths(dataset_root: Path, output_policy: dict[str, Any]) -> tuple[Path, Path]:
    restored_name = str(output_policy.get("restored_video_name", DEFAULT_OUTPUT_POLICY["restored_video_name"]))
    mask_name = str(output_policy.get("mask_video_name", DEFAULT_OUTPUT_POLICY["mask_video_name"]))
    return dataset_root / restored_name, dataset_root / mask_name


def decode_video_frames(video_path: Path, as_gray: bool = False) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    if not video_path.exists():
        return frames
    cap = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if as_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    finally:
        cap.release()
    return frames


def count_video_frames(video_path: Path) -> int:
    if not video_path.exists():
        return 0
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    return int(count)


def has_nonzero_mask_video(mask_video_path: Path, threshold: int = 127) -> bool:
    if not mask_video_path.exists():
        return False
    cap = cv2.VideoCapture(str(mask_video_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if int((gray > int(threshold)).sum()) > 0:
                return True
    finally:
        cap.release()
    return False


def compute_mask_coverage_from_dir_or_video(
    mask_dir: Path,
    mask_video_path: Path,
    threshold: int = 127,
) -> dict[str, Any]:
    threshold_i = int(threshold)
    image_exts = {".png", ".jpg", ".jpeg"}

    frame_count = 0
    sum_ratio = 0.0
    active_count = 0
    source = "none"

    mask_paths: list[Path] = []
    if mask_dir.exists() and mask_dir.is_dir():
        mask_paths = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in image_exts])

    if mask_paths:
        source = "mask_dir"
        for path in mask_paths:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            ratio = float((img > threshold_i).mean())
            frame_count += 1
            sum_ratio += ratio
            if ratio > 0.0:
                active_count += 1
    elif mask_video_path.exists():
        source = "mask_video"
        cap = cv2.VideoCapture(str(mask_video_path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ratio = float((gray > threshold_i).mean())
                frame_count += 1
                sum_ratio += ratio
                if ratio > 0.0:
                    active_count += 1
        finally:
            cap.release()

    if frame_count <= 0:
        return {
            "source": source,
            "frame_count": 0,
            "mean_mask_ratio": 0.0,
            "active_frame_ratio": 0.0,
        }

    return {
        "source": source,
        "frame_count": int(frame_count),
        "mean_mask_ratio": float(sum_ratio / float(frame_count)),
        "active_frame_ratio": float(active_count / float(frame_count)),
    }


def _encode_h264_raw_frames(
    frames: list[np.ndarray],
    out_path: Path,
    fps: float,
    crf: int,
    preset: str,
    output_pix_fmt: str,
    input_pix_fmt: str,
) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = np.asarray(frames[0])
    if first.ndim == 3:
        h, w = first.shape[:2]
    elif first.ndim == 2:
        h, w = first.shape
    else:
        raise ValueError(f"Unsupported frame shape for h264 encoding: {first.shape}")

    target_w = int(w)
    target_h = int(h)
    needs_even_pad = str(output_pix_fmt).lower() == "yuv420p" and (target_w % 2 != 0 or target_h % 2 != 0)
    if needs_even_pad:
        target_w += target_w % 2
        target_h += target_h % 2

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        input_pix_fmt,
        "-s",
        f"{target_w}x{target_h}",
        "-r",
        f"{float(fps):.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        str(output_pix_fmt),
        str(out_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stderr_data = b""
    try:
        assert proc.stdin is not None
        for frame in frames:
            arr = np.asarray(frame)
            if input_pix_fmt == "bgr24":
                if arr.ndim != 3 or arr.shape[2] != 3:
                    raise ValueError(f"Expected BGR frame shape (H,W,3), got {arr.shape}")
            elif input_pix_fmt == "gray":
                if arr.ndim != 2:
                    raise ValueError(f"Expected gray frame shape (H,W), got {arr.shape}")
            if arr.shape[0] != h or arr.shape[1] != w:
                raise ValueError(f"Inconsistent frame shape: expected {(h, w)}, got {arr.shape[:2]}")
            if needs_even_pad:
                pad_bottom = target_h - h
                pad_right = target_w - w
                border_type = cv2.BORDER_REPLICATE
                arr = cv2.copyMakeBorder(arr, 0, pad_bottom, 0, pad_right, border_type)
            arr = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.stderr is not None:
            stderr_data = proc.stderr.read()
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None and not proc.stderr.closed:
            proc.stderr.close()

    if proc.returncode != 0:
        err_tail = (stderr_data or b"")[-1200:].decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg h264 encode failed: {out_path} | {err_tail}")


def encode_dataset_h264_videos(
    dataset_root: Path,
    restored_frames_bgr: list[np.ndarray],
    masks_u8: list[np.ndarray],
    fps: float,
    output_policy: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    if not bool(output_policy.get("write_h264_videos", True)):
        return None, None

    restored_path, mask_path = dataset_video_paths(dataset_root=dataset_root, output_policy=output_policy)

    restored_cfg = output_policy.get("restored_h264", {}) or {}
    mask_cfg = output_policy.get("mask_h264", {}) or {}

    _encode_h264_raw_frames(
        frames=restored_frames_bgr,
        out_path=restored_path,
        fps=float(fps),
        crf=int(restored_cfg.get("crf", 23)),
        preset=str(restored_cfg.get("preset", "medium")),
        output_pix_fmt=str(restored_cfg.get("pix_fmt", "yuv420p")),
        input_pix_fmt="bgr24",
    )

    binary_masks = [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in masks_u8]
    _encode_h264_raw_frames(
        frames=binary_masks,
        out_path=mask_path,
        fps=float(fps),
        crf=int(mask_cfg.get("crf", 0)),
        preset=str(mask_cfg.get("preset", "medium")),
        output_pix_fmt=str(mask_cfg.get("pix_fmt", "gray")),
        input_pix_fmt="gray",
    )
    return restored_path, mask_path


def load_masks_by_names_with_video_fallback(
    *,
    mask_dir: Path,
    frame_names: list[str],
    frame_shape: tuple[int, int],
    mask_video_path: Path | None,
    threshold: int = 127,
) -> tuple[list[np.ndarray] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "mask_dir": str(mask_dir),
        "mask_video_path": str(mask_video_path) if mask_video_path is not None else None,
        "total_frames": int(len(frame_names)),
        "loaded_frames": 0,
        "missing_frames": 0,
        "source": "none",
    }

    h, w = frame_shape

    if mask_dir.exists():
        out: list[np.ndarray] = []
        loaded = 0
        missing = 0
        for name in frame_names:
            path = mask_dir / name
            if not path.exists():
                stem = Path(name).stem
                path = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt = mask_dir / f"{stem}{ext}"
                    if alt.exists():
                        path = alt
                        break
            if path is None:
                out.append(np.zeros((h, w), dtype=np.uint8))
                missing += 1
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                out.append(np.zeros((h, w), dtype=np.uint8))
                missing += 1
                continue
            if img.shape != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            out.append(((img > int(threshold)).astype(np.uint8) * 255))
            loaded += 1

        meta["source"] = "mask_dir"
        meta["loaded_frames"] = int(loaded)
        meta["missing_frames"] = int(missing)
        if loaded > 0:
            return out, meta

    if mask_video_path is not None and mask_video_path.exists():
        decoded = decode_video_frames(mask_video_path, as_gray=True)
        if decoded:
            out2: list[np.ndarray] = []
            for img in decoded:
                if img.shape != (h, w):
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
                out2.append(((img > int(threshold)).astype(np.uint8) * 255))

            if len(out2) < len(frame_names):
                out2.extend([out2[-1].copy() for _ in range(len(frame_names) - len(out2))])
            elif len(out2) > len(frame_names):
                out2 = out2[: len(frame_names)]

            meta["source"] = "mask_video"
            meta["loaded_frames"] = int(len(out2))
            meta["missing_frames"] = 0
            return out2, meta

    meta["status"] = "unavailable"
    return None, meta


def cleanup_named_subdirs(root: Path, subdirs: list[str]) -> list[str]:
    removed: list[str] = []
    for name in subdirs:
        p = root / str(name).strip()
        if p.exists() and p.is_dir():
            shutil.rmtree(p)
            removed.append(str(p))
    return removed


def cleanup_video_only_outputs(
    *,
    exp_pred_root: Path,
    datasets: list[str],
    output_policy: dict[str, Any],
) -> dict[str, Any]:
    policy = resolve_output_policy(output_policy)
    stats: dict[str, Any] = {
        "enabled": bool(policy.get("auto_cleanup_intermediates", True)),
        "video_only": bool(policy.get("video_only", True)),
        "removed_paths": [],
        "skipped_datasets_without_videos": [],
    }
    if not stats["enabled"]:
        return stats

    cleanup_cfg = policy.get("cleanup", {}) or {}
    removed: list[str] = []

    if bool(cleanup_cfg.get("remove_candidate_root", True)):
        candidate_root = exp_pred_root / "_candidates"
        if candidate_root.exists() and candidate_root.is_dir():
            shutil.rmtree(candidate_root)
            removed.append(str(candidate_root))

    if not stats["video_only"]:
        stats["removed_paths"] = removed
        return stats

    for ds in datasets:
        ds_root = exp_pred_root / ds
        restored_path, mask_path = dataset_video_paths(ds_root, policy)
        if not (restored_path.exists() and mask_path.exists()):
            stats["skipped_datasets_without_videos"].append(ds)
            continue
        if bool(cleanup_cfg.get("remove_frame_dirs", True)):
            frame_dir = ds_root / "frames"
            if frame_dir.exists() and frame_dir.is_dir():
                shutil.rmtree(frame_dir)
                removed.append(str(frame_dir))
        if bool(cleanup_cfg.get("remove_mask_dirs", True)):
            mask_dir = ds_root / "masks"
            if mask_dir.exists() and mask_dir.is_dir():
                shutil.rmtree(mask_dir)
                removed.append(str(mask_dir))

    stats["removed_paths"] = removed
    return stats
