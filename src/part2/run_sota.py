#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.part1.run_baseline import (
    maybe_install_ultralytics,
    resolve_device,
    restore_frames,
    run_segmentation,
    set_global_seed,
    str2bool,
    write_dataset_outputs,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SUPPORTED_MASK_BACKENDS = {"sam2", "trackanything"}
SUPPORTED_PROMPT_DETECTORS = {"yolo", "maskrcnn"}
DEFAULT_MMCV_VERSION = "2.2.0"


@dataclass
class DatasetPayload:
    frame_names: list[str]
    frames: list[np.ndarray]
    frame_dir: Path


@dataclass
class CandidateSpec:
    stage: str
    name: str
    mask_backend: str
    mask_variant: str
    neighbor_length: int
    ref_stride: int
    subvideo_length: int
    resize_ratio: float
    mask_dilation: int
    fp16: bool

    @property
    def candidate_id(self) -> str:
        raw = f"{self.stage}_{self.name}"
        return sanitize_name(raw)


@dataclass
class CandidateResult:
    spec: CandidateSpec
    candidate_root: Path
    eval_exp_id: str
    summary_path: Path
    aggregate: dict[str, Any]
    per_dataset: dict[str, Any]
    mask_stats: dict[str, dict[str, Any]]
    backend_meta: dict[str, dict[str, Any]]
    propainter_meta: dict[str, dict[str, Any]]


class CandidateExecutionError(RuntimeError):
    pass


def setup_logger(exp_id: str) -> tuple[logging.Logger, Path]:
    log_dir = REPO_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"phase2_{exp_id}.log"

    logger = logging.getLogger("phase2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger, log_path


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_repo_path(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def resolve_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def path_to_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def build_sam2_config_candidates(
    model_cfg_raw: str,
    model_cfg_path: Path | None,
    sam2_pkg_root: Path | None,
    sam2_repo_root: Path | None,
) -> list[str]:
    candidates: list[str] = []

    def add(value: str | None) -> None:
        if value is None:
            return
        token = str(value).strip().replace("\\", "/")
        if not token:
            return
        if token not in candidates:
            candidates.append(token)

    add(model_cfg_raw)

    if model_cfg_path is not None:
        cfg_posix = path_to_posix(model_cfg_path)
        add(cfg_posix)
        if "/configs/" in cfg_posix:
            tail = cfg_posix.split("/configs/", 1)[1]
            add(f"configs/{tail}")

    for root in [sam2_pkg_root, sam2_repo_root]:
        if root is None or model_cfg_path is None:
            continue
        try:
            rel = model_cfg_path.resolve().relative_to(root.resolve())
            rel_token = rel.as_posix()
            add(rel_token)
            if rel_token.startswith("sam2/"):
                add(rel_token[len("sam2/") :])
        except Exception:
            pass

    if not candidates:
        add("configs/sam2.1/sam2.1_hiera_t.yaml")

    # Prefer package-relative hydra names (e.g. configs/sam2.1/...)
    return sorted(
        candidates,
        key=lambda x: (
            0 if x.startswith("configs/") else 1,
            1 if Path(x).is_absolute() else 0,
            len(x),
        ),
    )


def is_mmcv_ready() -> bool:
    try:
        from mmcv.cnn import ConvModule  # type: ignore
        from mmcv.ops import ModulatedDeformConv2d  # type: ignore

        _ = ConvModule, ModulatedDeformConv2d
        return True
    except Exception:
        return False


def find_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists() and p.is_file():
            return p
    return None


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def collect_dataset_cfg(config: dict[str, Any]) -> tuple[dict[str, dict], list[str], list[str]]:
    datasets_cfg = config.get("datasets", {}) or {}
    mandatory = datasets_cfg.get("mandatory", {}) or {}
    optional = datasets_cfg.get("optional", {}) or {}

    if not isinstance(mandatory, dict) or not isinstance(optional, dict):
        raise ValueError("datasets.mandatory and datasets.optional must be mappings")

    merged = {**mandatory, **optional}
    all_names = list(merged.keys())
    mandatory_names = list(mandatory.keys())
    return merged, all_names, mandatory_names


def resolve_dataset_names(spec: str, all_names: list[str], mandatory_names: list[str]) -> list[str]:
    token = spec.strip().lower()
    if token == "all":
        return all_names
    if token == "mandatory":
        return mandatory_names

    requested = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [x for x in requested if x not in all_names]
    if unknown:
        raise ValueError(f"Unknown datasets in --datasets: {unknown}. Valid: {all_names}")
    return requested


def parse_backends(spec: str) -> list[str]:
    models: list[str] = []
    for token in [x.strip().lower() for x in spec.split(",") if x.strip()]:
        if token not in SUPPORTED_MASK_BACKENDS:
            raise ValueError(
                f"Unsupported mask backend '{token}'. Supported: {sorted(SUPPORTED_MASK_BACKENDS)}"
            )
        if token not in models:
            models.append(token)
    if not models:
        raise ValueError("At least one mask backend is required.")
    return models


def parse_stages(spec: str) -> list[str]:
    valid = ["B1", "B2", "B3", "B4", "B5"]
    if not spec:
        return valid
    stages = [x.strip().upper() for x in spec.split(",") if x.strip()]
    unknown = [x for x in stages if x not in valid]
    if unknown:
        raise ValueError(f"Unsupported --stages: {unknown}. Valid: {valid}")
    seen: set[str] = set()
    out: list[str] = []
    for s in stages:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def parse_prompt_detector(value: str) -> str:
    token = value.strip().lower()
    if token not in SUPPORTED_PROMPT_DETECTORS:
        raise ValueError(
            f"Unsupported prompt detector '{value}'. Supported: {sorted(SUPPORTED_PROMPT_DETECTORS)}"
        )
    return token


def load_dataset_payload(dataset_name: str, ds_cfg: dict[str, Any], max_frames: int | None) -> DatasetPayload:
    frame_dir = resolve_repo_path(Path(ds_cfg.get("processed_frames_dir", "")))
    if not frame_dir:
        raise ValueError(f"Dataset '{dataset_name}' missing processed_frames_dir in config.")

    frame_paths = list_images(frame_dir)
    if not frame_paths:
        raise RuntimeError(
            f"Dataset '{dataset_name}' has no processed frames in {frame_dir}. Run preprocess first."
        )

    if max_frames is not None and max_frames > 0:
        frame_paths = frame_paths[: max_frames]

    names: list[str] = []
    frames: list[np.ndarray] = []
    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {p}")
        names.append(p.name)
        frames.append(img)
    return DatasetPayload(frame_names=names, frames=frames, frame_dir=frame_dir)


def run_cmd(
    cmd: list[str],
    cwd: Path,
    logger: logging.Logger,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    logger.info("Run command: %s", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if completed.returncode != 0:
        logger.warning("Command failed (code=%s)", completed.returncode)
        if completed.stderr:
            logger.warning("stderr tail: %s", completed.stderr[-800:])
        if completed.stdout:
            logger.warning("stdout tail: %s", completed.stdout[-800:])
    return completed


def ensure_mmcv_runtime(
    *,
    auto_install: bool,
    logger: logging.Logger,
    mmcv_version: str,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "ready": is_mmcv_ready(),
        "install_state": "already_ready",
        "attempts": [],
        "requested_version": mmcv_version,
    }
    if meta["ready"]:
        return meta

    if not auto_install:
        meta["install_state"] = "missing_auto_install_disabled"
        return meta

    attempts: list[dict[str, Any]] = []

    if importlib.util.find_spec("mim") is not None:
        mim_cmd = [sys.executable, "-m", "mim", "install", "mmcv"]
        completed = run_cmd(cmd=mim_cmd, cwd=REPO_ROOT, logger=logger)
        attempts.append(
            {
                "installer": "mim",
                "command": " ".join(mim_cmd),
                "returncode": completed.returncode,
            }
        )
        if is_mmcv_ready():
            meta["ready"] = True
            meta["install_state"] = "installed_via_mim"
            meta["attempts"] = attempts
            return meta
    else:
        attempts.append(
            {
                "installer": "mim",
                "command": f"{sys.executable} -m mim install mmcv",
                "returncode": None,
                "status": "mim_module_missing",
            }
        )

    pip_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        f"mmcv=={mmcv_version}",
    ]
    completed = run_cmd(cmd=pip_cmd, cwd=REPO_ROOT, logger=logger)
    attempts.append(
        {
            "installer": "pip",
            "command": " ".join(pip_cmd),
            "returncode": completed.returncode,
        }
    )

    meta["ready"] = is_mmcv_ready()
    meta["install_state"] = (
        "installed_via_pip_no_build_isolation" if meta["ready"] else "install_failed"
    )
    meta["attempts"] = attempts
    return meta


def ensure_repo(
    repo_name: str,
    repo_url: str,
    repo_root: Path,
    auto_install: bool,
    logger: logging.Logger,
) -> tuple[Path | None, str]:
    target = repo_root / repo_name
    if target.exists() and (target / ".git").exists():
        return target, "existing"

    if not auto_install:
        return None, "missing_auto_install_disabled"

    repo_root.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", repo_url, str(target)]
    completed = run_cmd(cmd=cmd, cwd=REPO_ROOT, logger=logger)
    if completed.returncode != 0:
        return None, "clone_failed"
    return target, "cloned"


def maybe_pip_install_requirements(
    repo_path: Path,
    marker_name: str,
    logger: logging.Logger,
    auto_install: bool,
) -> tuple[bool, str]:
    marker = repo_path / marker_name
    if marker.exists():
        return True, "already_installed"

    req_path = repo_path / "requirements.txt"
    if not req_path.exists():
        marker.write_text(datetime.utcnow().isoformat() + "Z\n", encoding="utf-8")
        return True, "no_requirements"

    if not auto_install:
        return False, "auto_install_disabled"

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
    completed = run_cmd(cmd=cmd, cwd=repo_path, logger=logger)
    if completed.returncode != 0:
        return False, "pip_install_failed"

    marker.write_text(datetime.utcnow().isoformat() + "Z\n", encoding="utf-8")
    return True, "installed"


def ensure_propainter_ready(
    part2_cfg: dict[str, Any],
    external_root: Path,
    auto_install_missing: bool,
    logger: logging.Logger,
) -> tuple[Path | None, dict[str, Any]]:
    prop_cfg = part2_cfg.get("propainter", {}) or {}
    repo_url = str(prop_cfg.get("repo_url", "https://github.com/sczhou/ProPainter.git"))
    repo_root = external_root / "repos"

    repo_path, repo_status = ensure_repo(
        repo_name="ProPainter",
        repo_url=repo_url,
        repo_root=repo_root,
        auto_install=auto_install_missing,
        logger=logger,
    )
    meta: dict[str, Any] = {
        "repo_status": repo_status,
        "repo_url": repo_url,
        "repo_path": str(repo_path) if repo_path else None,
        "ready": False,
    }
    if repo_path is None:
        return None, meta

    ok, install_status = maybe_pip_install_requirements(
        repo_path=repo_path,
        marker_name=".deps_installed_phase2",
        logger=logger,
        auto_install=auto_install_missing,
    )
    meta["deps_status"] = install_status

    infer_path = repo_path / "inference_propainter.py"
    if not infer_path.exists():
        meta["ready"] = False
        meta["error"] = f"missing inference script: {infer_path}"
        return None, meta

    meta["ready"] = bool(ok)
    return repo_path, meta


def probe_backend_environment(
    backends: list[str],
    part2_cfg: dict[str, Any],
    external_root: Path,
    auto_install_missing: bool,
    logger: logging.Logger,
) -> dict[str, dict[str, Any]]:
    model_cfg = part2_cfg.get("models", {}) or {}
    repo_root = external_root / "repos"
    status: dict[str, dict[str, Any]] = {}

    for backend in backends:
        if backend == "sam2":
            module_ok = importlib.util.find_spec("sam2") is not None
            repo_url = str(model_cfg.get("sam2_repo", "https://github.com/facebookresearch/sam2.git"))
            repo_path, repo_state = ensure_repo(
                repo_name="sam2",
                repo_url=repo_url,
                repo_root=repo_root,
                auto_install=auto_install_missing,
                logger=logger,
            )
            pip_state = "not_needed"
            if not module_ok and repo_path is not None:
                editable_cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo_path)]
                if auto_install_missing:
                    completed = run_cmd(editable_cmd, cwd=repo_path, logger=logger)
                    pip_state = "installed" if completed.returncode == 0 else "install_failed"
                    module_ok = importlib.util.find_spec("sam2") is not None
                else:
                    pip_state = "auto_install_disabled"

            status[backend] = {
                "module_importable": module_ok,
                "repo_state": repo_state,
                "repo_path": str(repo_path) if repo_path else None,
                "pip_state": pip_state,
                "ready": module_ok or repo_path is not None,
            }

        elif backend == "trackanything":
            module_ok = importlib.util.find_spec("track_anything") is not None
            mmcv_version = str(model_cfg.get("trackanything_mmcv_version", DEFAULT_MMCV_VERSION))
            mmcv_meta = ensure_mmcv_runtime(
                auto_install=auto_install_missing,
                logger=logger,
                mmcv_version=mmcv_version,
            )
            repo_url = str(
                model_cfg.get("trackanything_repo", "https://github.com/gaomingqi/Track-Anything.git")
            )
            repo_path, repo_state = ensure_repo(
                repo_name="Track-Anything",
                repo_url=repo_url,
                repo_root=repo_root,
                auto_install=auto_install_missing,
                logger=logger,
            )
            pip_state = "not_needed"
            if repo_path is not None:
                ok, req_status = maybe_pip_install_requirements(
                    repo_path=repo_path,
                    marker_name=".deps_installed_phase2",
                    logger=logger,
                    auto_install=auto_install_missing,
                )
                pip_state = req_status
                module_ok = module_ok or ok

            status[backend] = {
                "module_importable": module_ok,
                "repo_state": repo_state,
                "repo_path": str(repo_path) if repo_path else None,
                "pip_state": pip_state,
                "mmcv_ready": bool(mmcv_meta.get("ready", False)),
                "mmcv_install_state": mmcv_meta.get("install_state"),
                "mmcv_install_attempts": mmcv_meta.get("attempts", []),
                "ready": module_ok or repo_path is not None,
            }

    return status


def union_instances_to_masks(
    instances_per_frame: list[list[dict[str, Any]]],
    frame_shape: tuple[int, int],
) -> list[np.ndarray]:
    h, w = frame_shape
    masks: list[np.ndarray] = []
    for instances in instances_per_frame:
        mask = np.zeros((h, w), dtype=np.uint8)
        for inst in instances:
            mask = np.maximum(mask, (inst["mask"] > 0).astype(np.uint8) * 255)
        masks.append(mask)
    return masks


def compute_mean_mask_ratio(masks_u8: list[np.ndarray]) -> float:
    if not masks_u8:
        return 0.0
    ratios = [float((m > 0).mean()) for m in masks_u8]
    return float(np.mean(np.array(ratios, dtype=np.float32)))


def read_mask_binary(path: Path, frame_shape: tuple[int, int]) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = frame_shape
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((img > 0).astype(np.uint8) * 255)


def load_external_masks_by_names(
    mask_dir: Path,
    frame_names: list[str],
    frame_shape: tuple[int, int],
) -> tuple[list[np.ndarray] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "mask_dir": str(mask_dir),
        "total_frames": int(len(frame_names)),
        "loaded_frames": 0,
        "missing_frames": 0,
    }
    if not mask_dir.exists():
        meta["status"] = "mask_dir_missing"
        return None, meta

    out: list[np.ndarray] = []
    loaded = 0
    missing = 0
    for name in frame_names:
        candidate = mask_dir / name
        path = candidate if candidate.exists() else None
        if path is None:
            stem = Path(name).stem
            for ext in [".png", ".jpg", ".jpeg"]:
                alt = mask_dir / f"{stem}{ext}"
                if alt.exists():
                    path = alt
                    break

        if path is None:
            h, w = frame_shape
            out.append(np.zeros((h, w), dtype=np.uint8))
            missing += 1
            continue

        mask = read_mask_binary(path=path, frame_shape=frame_shape)
        if mask is None:
            h, w = frame_shape
            out.append(np.zeros((h, w), dtype=np.uint8))
            missing += 1
            continue

        out.append(mask)
        loaded += 1

    meta["loaded_frames"] = int(loaded)
    meta["missing_frames"] = int(missing)
    meta["status"] = "ok" if loaded > 0 else "no_readable_masks"
    return (out if loaded > 0 else None), meta


def fuse_mask_pair(
    backend_mask: np.ndarray,
    prior_mask: np.ndarray,
    method: str,
) -> np.ndarray:
    m = method.strip().lower()
    if m in {"intersect", "intersection", "and"}:
        return (((backend_mask > 0) & (prior_mask > 0)).astype(np.uint8) * 255)
    if m in {"union", "or"}:
        return np.maximum(backend_mask, prior_mask)
    if m in {"phase1", "phase1_only", "prior"}:
        return prior_mask.copy()
    return backend_mask.copy()


def dataset_metric_aggregate(
    per_dataset: dict[str, Any],
    datasets_for_scoring: list[str] | None,
) -> dict[str, float]:
    keys = ["JM", "JR", "PSNR", "SSIM"]
    values: dict[str, list[float]] = {k: [] for k in keys}

    if datasets_for_scoring:
        names = [x for x in datasets_for_scoring if x in per_dataset]
    else:
        names = list(per_dataset.keys())

    for ds in names:
        metrics = (per_dataset.get(ds, {}) or {}).get("metrics", {}) or {}
        for key in keys:
            value = metrics.get(key, None)
            if value is None:
                continue
            values[key].append(float(value))

    return {
        key: (float(np.mean(np.array(vals, dtype=np.float32))) if vals else float("-inf"))
        for key, vals in values.items()
    }


def align_masks_to_frame_count(
    masks_u8: list[np.ndarray],
    frame_count: int,
    frame_shape: tuple[int, int],
) -> list[np.ndarray]:
    if frame_count <= 0:
        return []
    if not masks_u8:
        h, w = frame_shape
        return [np.zeros((h, w), dtype=np.uint8) for _ in range(frame_count)]

    out = [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in masks_u8]
    if len(out) < frame_count:
        out.extend([out[-1].copy() for _ in range(frame_count - len(out))])
    elif len(out) > frame_count:
        out = out[:frame_count]
    return out


def normalize_odd_kernel(kernel_size: int) -> int:
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    return k


def refine_masks(
    masks_u8: list[np.ndarray],
    morph_kernel: int,
    temporal_window: int,
) -> list[np.ndarray]:
    if not masks_u8:
        return masks_u8

    k = normalize_odd_kernel(morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    refined: list[np.ndarray] = []

    for m in masks_u8:
        c = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
        c = cv2.morphologyEx(c, cv2.MORPH_OPEN, kernel, iterations=1)
        c = cv2.dilate(c, kernel, iterations=1)
        refined.append((c > 0).astype(np.uint8) * 255)

    if temporal_window <= 0:
        return refined

    bool_stack = np.stack([(m > 0).astype(np.float32) for m in refined], axis=0)
    out: list[np.ndarray] = []
    n = bool_stack.shape[0]
    for idx in range(n):
        lo = max(0, idx - temporal_window)
        hi = min(n, idx + temporal_window + 1)
        avg = bool_stack[lo:hi].mean(axis=0)
        out.append((avg >= 0.5).astype(np.uint8) * 255)
    return out


def mask_to_bbox(mask_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_motion_fallback_box(frames: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    if not frames:
        return None
    if len(frames) < 2:
        h, w = frames[0].shape[:2]
        return int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)

    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = []
    for idx in range(1, len(gray)):
        diffs.append(cv2.absdiff(gray[idx], gray[idx - 1]))
    if not diffs:
        return None

    score_map = np.max(np.stack(diffs, axis=0), axis=0)
    thr = float(np.percentile(score_map.astype(np.float32), 97.5))
    if thr <= 0:
        thr = float(score_map.mean() + score_map.std())
    mask = (score_map >= thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask_to_bbox(mask)


def build_auto_prompts(
    instances_per_frame: list[list[dict[str, Any]]],
    frames: list[np.ndarray],
    max_prompts: int,
) -> dict[str, Any]:
    best_idx = 0
    best_area = -1.0
    best_instances: list[dict[str, Any]] = []

    for idx, instances in enumerate(instances_per_frame):
        total = 0.0
        for inst in instances:
            total += float((inst["mask"] > 0).sum())
        if total > best_area:
            best_area = total
            best_idx = idx
            best_instances = instances

    boxes: list[tuple[int, int, int, int]] = []
    ranked = sorted(best_instances, key=lambda x: float((x["mask"] > 0).sum()), reverse=True)
    for inst in ranked[: max(1, max_prompts)]:
        box = mask_to_bbox((inst["mask"] > 0).astype(np.uint8) * 255)
        if box is not None:
            boxes.append(box)

    source = "detector"
    if not boxes:
        fallback = build_motion_fallback_box(frames)
        if fallback is not None:
            boxes = [fallback]
            source = "motion_fallback"

    if not boxes and frames:
        h, w = frames[0].shape[:2]
        boxes = [(int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.8))]
        source = "center_fallback"

    return {
        "frame_idx": int(best_idx),
        "boxes": boxes,
        "source": source,
        "max_area": float(best_area),
    }


def prepare_sam2_video_input_dir(payload: DatasetPayload) -> tuple[Path, bool]:
    all_names = [p.name for p in list_images(payload.frame_dir)]
    full_match = len(all_names) == len(payload.frame_names) and all_names == payload.frame_names
    jpg_like = all(name.lower().endswith((".jpg", ".jpeg")) for name in payload.frame_names)
    numeric = all(Path(name).stem.isdigit() for name in payload.frame_names)

    if full_match and jpg_like and numeric:
        return payload.frame_dir, False

    tmp_dir = Path(tempfile.mkdtemp(prefix="phase2_sam2_frames_"))
    for idx, frame in enumerate(payload.frames):
        out_path = tmp_dir / f"{idx:05d}.jpg"
        ok = cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"failed to write SAM2 input frame: {out_path}")
    return tmp_dir, True


def build_backend_masks_from_detector_fallback(
    backend: str,
    detector_instances: list[list[dict[str, Any]]],
    frame_shape: tuple[int, int],
    prompt_boxes: list[tuple[int, int, int, int]] | None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    masks = union_instances_to_masks(detector_instances, frame_shape=frame_shape)
    meta: dict[str, Any] = {
        "backend": backend,
        "official_used": False,
        "fallback_used": True,
        "fallback_reason": "official_backend_unavailable_or_failed",
    }

    if backend == "trackanything":
        masks = refine_masks(masks_u8=masks, morph_kernel=5, temporal_window=1)
    else:
        masks = refine_masks(masks_u8=masks, morph_kernel=3, temporal_window=0)

    if compute_mean_mask_ratio(masks) <= 0.0 and prompt_boxes:
        h, w = frame_shape
        fallback_masks: list[np.ndarray] = []
        for _ in range(len(detector_instances)):
            m = np.zeros((h, w), dtype=np.uint8)
            for x1, y1, x2, y2 in prompt_boxes:
                x1c = max(0, min(w - 1, int(x1)))
                x2c = max(0, min(w - 1, int(x2)))
                y1c = max(0, min(h - 1, int(y1)))
                y2c = max(0, min(h - 1, int(y2)))
                if x2c <= x1c or y2c <= y1c:
                    continue
                m[y1c : y2c + 1, x1c : x2c + 1] = 255
            fallback_masks.append(m)
        masks = refine_masks(masks_u8=fallback_masks, morph_kernel=5, temporal_window=1)
        meta["fallback_reason"] = "detector_empty_use_prompt_boxes"
        meta["prompt_box_fallback"] = True

    meta["mean_mask_ratio"] = compute_mean_mask_ratio(masks)
    return masks, meta


def resolve_sam2_checkpoint(
    part2_cfg: dict[str, Any],
    backend_env: dict[str, dict[str, Any]],
) -> tuple[Path | None, str]:
    models_cfg = part2_cfg.get("models", {}) or {}
    backend_meta = backend_env.get("sam2", {}) or {}
    repo_path = resolve_optional_path(backend_meta.get("repo_path"))

    explicit = resolve_optional_path(models_cfg.get("sam2_checkpoint"))
    if explicit is not None and explicit.exists():
        return explicit, "config_explicit"

    candidates: list[Path] = []
    if repo_path is not None:
        candidates.extend(sorted((repo_path / "checkpoints").glob("*.pt")))

    local_ckpt_root = REPO_ROOT / "outputs" / "external" / "part2" / "checkpoints"
    candidates.extend(sorted(local_ckpt_root.glob("sam2*.pt")))
    candidates.extend(sorted(local_ckpt_root.glob("*.pt")))

    found = find_first_existing(candidates)
    if found is None:
        return None, "missing"
    return found, "auto_discovered"


def try_run_sam2_official(
    payload: DatasetPayload,
    prompt_frame_idx: int,
    prompt_boxes: list[tuple[int, int, int, int]],
    part2_cfg: dict[str, Any],
    backend_env: dict[str, dict[str, Any]],
    device: str,
) -> tuple[list[np.ndarray] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "backend": "sam2",
        "official_attempted": True,
        "official_used": False,
    }

    if not prompt_boxes:
        meta["official_error"] = "no_prompt_boxes"
        return None, meta

    ckpt, ckpt_source = resolve_sam2_checkpoint(part2_cfg=part2_cfg, backend_env=backend_env)
    if ckpt is None:
        meta["official_error"] = "sam2_checkpoint_missing"
        return None, meta
    meta["sam2_checkpoint"] = str(ckpt)
    meta["sam2_checkpoint_source"] = ckpt_source

    models_cfg = part2_cfg.get("models", {}) or {}
    model_cfg_raw = str(models_cfg.get("sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_t.yaml"))

    model_cfg_path = resolve_optional_path(model_cfg_raw)
    if model_cfg_path is None or not model_cfg_path.exists():
        repo_path = resolve_optional_path((backend_env.get("sam2", {}) or {}).get("repo_path"))
        if repo_path is not None:
            cand = repo_path / model_cfg_raw
            if cand.exists():
                model_cfg_path = cand

    try:
        import sam2
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as e:
        meta["official_error"] = f"sam2_import_error:{type(e).__name__}:{e}"
        return None, meta

    sam2_paths = list(getattr(sam2, "__path__", []))
    sam2_pkg_root = Path(sam2_paths[0]).resolve() if sam2_paths else None
    sam2_repo_root = resolve_optional_path((backend_env.get("sam2", {}) or {}).get("repo_path"))
    cfg_candidates = build_sam2_config_candidates(
        model_cfg_raw=model_cfg_raw,
        model_cfg_path=model_cfg_path if model_cfg_path is not None and model_cfg_path.exists() else None,
        sam2_pkg_root=sam2_pkg_root,
        sam2_repo_root=sam2_repo_root,
    )
    meta["sam2_model_cfg_candidates"] = cfg_candidates

    try:
        predictor = None
        last_cfg_err: Exception | None = None
        used_cfg = None
        for cfg_name in cfg_candidates:
            try:
                predictor = build_sam2_video_predictor(
                    cfg_name,
                    str(ckpt),
                    device=device,
                )
                used_cfg = cfg_name
                break
            except Exception as cfg_err:
                last_cfg_err = cfg_err
                continue

        if predictor is None:
            if last_cfg_err is None:
                raise RuntimeError("no_valid_sam2_config_candidate")
            raise last_cfg_err

        meta["sam2_model_cfg"] = str(used_cfg)
        sam2_video_dir, cleanup_sam2_video_dir = prepare_sam2_video_input_dir(payload)
        meta["sam2_video_input_dir"] = str(sam2_video_dir)
        meta["sam2_video_input_materialized"] = bool(cleanup_sam2_video_dir)
        try:
            try:
                state = predictor.init_state(video_path=str(sam2_video_dir))
            except TypeError:
                state = predictor.init_state(str(sam2_video_dir))

            with torch.inference_mode():
                for obj_id, (x1, y1, x2, y2) in enumerate(prompt_boxes, start=1):
                    box = np.array([x1, y1, x2, y2], dtype=np.float32)
                    try:
                        predictor.add_new_points_or_box(
                            state,
                            frame_idx=int(prompt_frame_idx),
                            obj_id=int(obj_id),
                            box=box,
                        )
                    except TypeError:
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=int(prompt_frame_idx),
                            obj_id=int(obj_id),
                            box=box,
                        )

                h, w = payload.frames[0].shape[:2]
                masks_u8 = [np.zeros((h, w), dtype=np.uint8) for _ in payload.frames]
                for out in predictor.propagate_in_video(state):
                    if not isinstance(out, (list, tuple)) or len(out) < 3:
                        continue
                    frame_idx, _obj_ids, mask_logits = out[0], out[1], out[2]
                    idx = int(frame_idx)
                    if idx < 0 or idx >= len(masks_u8):
                        continue
                    arr = mask_logits
                    if hasattr(arr, "detach"):
                        arr = arr.detach().cpu().numpy()
                    arr_np = np.asarray(arr)
                    frame_mask = np.zeros((h, w), dtype=np.uint8)
                    if arr_np.ndim == 2:
                        frame_mask = (arr_np > 0).astype(np.uint8) * 255
                    else:
                        for m in arr_np:
                            m2 = m[0] if getattr(m, "ndim", 0) == 3 else m
                            frame_mask = np.maximum(
                                frame_mask,
                                (np.asarray(m2) > 0).astype(np.uint8) * 255,
                            )
                    masks_u8[idx] = frame_mask
        finally:
            if cleanup_sam2_video_dir:
                shutil.rmtree(sam2_video_dir, ignore_errors=True)

        if compute_mean_mask_ratio(masks_u8) <= 0.0:
            meta["official_error"] = "sam2_output_all_zero"
            return None, meta

        meta["official_used"] = True
        meta["fallback_used"] = False
        meta["mean_mask_ratio"] = compute_mean_mask_ratio(masks_u8)
        return masks_u8, meta
    except Exception as e:
        meta["official_error"] = f"sam2_runtime_error:{type(e).__name__}:{e}"
        return None, meta


def resolve_trackanything_checkpoints(
    part2_cfg: dict[str, Any],
    backend_env: dict[str, dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, str], list[str]]:
    models_cfg = part2_cfg.get("models", {}) or {}
    backend_meta = backend_env.get("trackanything", {}) or {}
    repo_path = resolve_optional_path(backend_meta.get("repo_path"))

    out: dict[str, Path] = {}
    source: dict[str, str] = {}
    missing: list[str] = []

    ckpt_root_candidates: list[Path] = []
    if repo_path is not None:
        ckpt_root_candidates.append(repo_path / "checkpoints")
    ckpt_root_candidates.append(REPO_ROOT / "outputs" / "external" / "part2" / "checkpoints")

    required = {
        "sam": models_cfg.get("trackanything_sam_checkpoint"),
        "xmem": models_cfg.get("trackanything_xmem_checkpoint"),
        "e2fgvi": models_cfg.get("trackanything_e2fgvi_checkpoint"),
    }

    patterns = {
        "sam": ["sam*.pth", "sam*.pt"],
        "xmem": ["XMem*.pth", "xmem*.pth", "xmem*.pt"],
        "e2fgvi": ["E2FGVI*.pth", "e2fgvi*.pth", "e2fgvi*.pt"],
    }

    for key, raw in required.items():
        explicit = resolve_optional_path(raw)
        if explicit is not None and explicit.exists():
            out[key] = explicit
            source[key] = "config_explicit"
            continue

        cands: list[Path] = []
        for root in ckpt_root_candidates:
            if not root.exists():
                continue
            for pat in patterns[key]:
                cands.extend(sorted(root.glob(pat)))

        found = find_first_existing(cands)
        if found is None:
            missing.append(key)
        else:
            out[key] = found
            source[key] = "auto_discovered"

    return out, source, missing


def build_prompt_template_mask(
    frame_shape: tuple[int, int],
    prompt_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    h, w = frame_shape
    template = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in prompt_boxes:
        x1c = max(0, min(w - 1, int(x1)))
        x2c = max(0, min(w - 1, int(x2)))
        y1c = max(0, min(h - 1, int(y1)))
        y2c = max(0, min(h - 1, int(y2)))
        if x2c > x1c and y2c > y1c:
            template[y1c : y2c + 1, x1c : x2c + 1] = 1
    return template


def build_prompt_order(frame_count: int, prompt_frame_idx: int) -> list[int]:
    if frame_count <= 0:
        return []
    start = max(0, min(frame_count - 1, int(prompt_frame_idx)))
    return list(range(start, frame_count)) + list(range(0, start))


def run_trackanything_xmem_tracker(
    payload: DatasetPayload,
    repo_path: Path,
    ckpts: dict[str, Path],
    prompt_boxes: list[tuple[int, int, int, int]],
    prompt_frame_idx: int,
    device: str,
) -> list[np.ndarray]:
    tracker_device = "cuda:0" if device == "cuda" else "cpu"
    if not payload.frames:
        return []
    order = build_prompt_order(len(payload.frames), prompt_frame_idx=prompt_frame_idx)
    frames_rgb = [cv2.cvtColor(payload.frames[idx], cv2.COLOR_BGR2RGB) for idx in order]
    template = build_prompt_template_mask(payload.frames[order[0]].shape[:2], prompt_boxes)

    with pushd(repo_path):
        base_tracker_mod = importlib.import_module("tracker.base_tracker")
        BaseTracker = getattr(base_tracker_mod, "BaseTracker")
        tracker = BaseTracker(str(ckpts["xmem"]), tracker_device, None, None)
        try:
            masks_u8: list[np.ndarray] = []
            for idx, frame in enumerate(frames_rgb):
                if idx == 0:
                    mask, _logit, _painted = tracker.track(frame, template)
                else:
                    mask, _logit, _painted = tracker.track(frame)
                masks_u8.append((np.asarray(mask) > 0).astype(np.uint8) * 255)
        finally:
            try:
                tracker.clear_memory()
            except Exception:
                pass

    h, w = payload.frames[0].shape[:2]
    reordered = [np.zeros((h, w), dtype=np.uint8) for _ in payload.frames]
    if masks_u8:
        limit = min(len(order), len(masks_u8))
        for ordered_idx in range(limit):
            reordered[order[ordered_idx]] = masks_u8[ordered_idx]
        if limit < len(order):
            last = masks_u8[limit - 1]
            for ordered_idx in range(limit, len(order)):
                reordered[order[ordered_idx]] = last.copy()
    return reordered


def try_run_trackanything_official(
    payload: DatasetPayload,
    prompt_frame_idx: int,
    prompt_boxes: list[tuple[int, int, int, int]],
    part2_cfg: dict[str, Any],
    backend_env: dict[str, dict[str, Any]],
    device: str,
) -> tuple[list[np.ndarray] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "backend": "trackanything",
        "official_attempted": True,
        "official_used": False,
        "prompt_frame_idx": int(prompt_frame_idx),
    }

    if not prompt_boxes:
        meta["official_error"] = "no_prompt_boxes"
        return None, meta

    repo_path = resolve_optional_path((backend_env.get("trackanything", {}) or {}).get("repo_path"))
    if repo_path is None or not repo_path.exists():
        meta["official_error"] = "trackanything_repo_missing"
        return None, meta
    meta["trackanything_repo"] = str(repo_path)

    ckpts, ckpt_sources, missing = resolve_trackanything_checkpoints(
        part2_cfg=part2_cfg,
        backend_env=backend_env,
    )
    if missing:
        meta["official_error"] = f"trackanything_checkpoint_missing:{','.join(missing)}"
        meta["checkpoint_sources"] = ckpt_sources
        return None, meta
    meta["checkpoint_sources"] = ckpt_sources

    repo_str = str(repo_path)
    tracker_str = str(repo_path / "tracker")
    inpainter_str = str(repo_path / "inpainter")
    for p in [repo_str, tracker_str, inpainter_str]:
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        masks_u8: list[np.ndarray] | None = None
        mode = ""
        wrapper_error = ""
        track_meta = backend_env.get("trackanything", {}) or {}
        mmcv_ready = bool(track_meta.get("mmcv_ready", False)) or is_mmcv_ready()

        # Prefer Track-Anything wrapper when mmcv runtime is available.
        if mmcv_ready:
            try:
                ta_mod = importlib.import_module("track_anything")
                TrackingAnything = getattr(ta_mod, "TrackingAnything")
                args_ns = argparse.Namespace(
                    device=("cuda:0" if device == "cuda" else "cpu"),
                    sam_model_type=str(
                        (part2_cfg.get("models", {}) or {}).get("trackanything_sam_model_type", "vit_h")
                    ),
                    port=6080,
                    debug=False,
                    mask_save=False,
                )
                with pushd(repo_path):
                    engine = TrackingAnything(
                        str(ckpts["sam"]),
                        str(ckpts["xmem"]),
                        str(ckpts["e2fgvi"]),
                        args_ns,
                    )
                    order = build_prompt_order(len(payload.frames), prompt_frame_idx=prompt_frame_idx)
                    frames_rgb = [cv2.cvtColor(payload.frames[idx], cv2.COLOR_BGR2RGB) for idx in order]
                    template = build_prompt_template_mask(payload.frames[order[0]].shape[:2], prompt_boxes)
                    masks, _logits, _painted = engine.generator(frames_rgb, template)
                    try:
                        engine.xmem.clear_memory()
                    except Exception:
                        pass
                    ordered_masks = [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in masks]
                    h, w = payload.frames[0].shape[:2]
                    masks_u8 = [np.zeros((h, w), dtype=np.uint8) for _ in payload.frames]
                    if ordered_masks:
                        limit = min(len(order), len(ordered_masks))
                        for ordered_idx in range(limit):
                            masks_u8[order[ordered_idx]] = ordered_masks[ordered_idx]
                        if limit < len(order):
                            last = ordered_masks[limit - 1]
                            for ordered_idx in range(limit, len(order)):
                                masks_u8[order[ordered_idx]] = last.copy()
                    mode = "trackinganything_wrapper"
            except Exception as e:
                wrapper_error = f"wrapper_error:{type(e).__name__}:{e}"
                masks_u8 = None

        # Fallback inside official local Track-Anything stack: direct XMem tracker.
        if masks_u8 is None:
            masks_u8 = run_trackanything_xmem_tracker(
                payload=payload,
                repo_path=repo_path,
                ckpts=ckpts,
                prompt_boxes=prompt_boxes,
                prompt_frame_idx=prompt_frame_idx,
                device=device,
            )
            mode = "xmem_tracker"
            if wrapper_error:
                meta["wrapper_error"] = wrapper_error
            if not mmcv_ready:
                meta["wrapper_skipped_reason"] = "mmcv_unavailable_use_xmem_tracker"
                mmcv_state = track_meta.get("mmcv_install_state")
                if mmcv_state:
                    meta["mmcv_install_state"] = mmcv_state

        if not masks_u8:
            meta["official_error"] = "trackanything_empty_output"
            return None, meta
        if len(masks_u8) != len(payload.frames):
            # Align to frame length to avoid downstream mismatch.
            n = min(len(masks_u8), len(payload.frames))
            masks_u8 = masks_u8[:n] + [masks_u8[-1].copy() for _ in range(len(payload.frames) - n)]
        if compute_mean_mask_ratio(masks_u8) <= 0.0:
            meta["official_error"] = "trackanything_output_all_zero"
            return None, meta

        meta["official_mode"] = mode
        meta["official_used"] = True
        meta["fallback_used"] = False
        meta["mean_mask_ratio"] = compute_mean_mask_ratio(masks_u8)
        return masks_u8, meta
    except Exception as e:
        meta["official_error"] = f"trackanything_runtime_error:{type(e).__name__}:{e}"
        return None, meta


def generate_backend_masks_with_priority(
    backend: str,
    payload: DatasetPayload,
    detector_instances: list[list[dict[str, Any]]],
    prompt_frame_idx: int,
    prompt_boxes: list[tuple[int, int, int, int]],
    part2_cfg: dict[str, Any],
    backend_env: dict[str, dict[str, Any]],
    device: str,
    logger: logging.Logger,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    official_masks: list[np.ndarray] | None = None
    official_meta: dict[str, Any]

    if backend == "sam2":
        official_masks, official_meta = try_run_sam2_official(
            payload=payload,
            prompt_frame_idx=prompt_frame_idx,
            prompt_boxes=prompt_boxes,
            part2_cfg=part2_cfg,
            backend_env=backend_env,
            device=device,
        )
    elif backend == "trackanything":
        official_masks, official_meta = try_run_trackanything_official(
            payload=payload,
            prompt_frame_idx=prompt_frame_idx,
            prompt_boxes=prompt_boxes,
            part2_cfg=part2_cfg,
            backend_env=backend_env,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if official_masks is not None:
        logger.info(
            "Official backend succeeded: backend=%s ratio=%.6f",
            backend,
            compute_mean_mask_ratio(official_masks),
        )
        return official_masks, official_meta

    fallback_masks, fallback_meta = build_backend_masks_from_detector_fallback(
        backend=backend,
        detector_instances=detector_instances,
        frame_shape=payload.frames[0].shape[:2],
        prompt_boxes=prompt_boxes,
    )
    fallback_meta["official_attempted"] = True
    fallback_meta["official_error"] = official_meta.get("official_error", "unknown")
    fallback_meta["official_used"] = False
    logger.warning(
        "Fallback activated (non-silent): backend=%s reason=%s",
        backend,
        fallback_meta["official_error"],
    )
    return fallback_masks, fallback_meta


def prepare_prompt_instances(
    payload: DatasetPayload,
    part1_cfg: dict[str, Any],
    dynamic_classes: set[str],
    prompt_detector: str,
    device: str,
    auto_install_missing: bool,
    logger: logging.Logger,
) -> tuple[list[list[dict[str, Any]]], str]:
    chosen = prompt_detector

    if chosen == "yolo":
        if not maybe_install_ultralytics(auto_install=auto_install_missing, logger=logger):
            logger.warning("YOLO unavailable for prompt detection, fallback to maskrcnn.")
            chosen = "maskrcnn"

    try:
        inst = run_segmentation(
            model_name=chosen,
            payload=payload,
            part1_cfg=part1_cfg,
            dynamic_classes=dynamic_classes,
            device=device,
            auto_install_missing=auto_install_missing,
            logger=logger,
        )
        return inst, chosen
    except Exception:
        if chosen != "maskrcnn":
            logger.warning("Prompt detector %s failed, fallback to maskrcnn", chosen)
            inst = run_segmentation(
                model_name="maskrcnn",
                payload=payload,
                part1_cfg=part1_cfg,
                dynamic_classes=dynamic_classes,
                device=device,
                auto_install_missing=auto_install_missing,
                logger=logger,
            )
            return inst, "maskrcnn"
        raise


def ensure_unique_input_dir(
    payload: DatasetPayload,
    dataset_name: str,
    target_root: Path,
) -> Path:
    input_dir = target_root / dataset_name
    if input_dir.exists():
        shutil.rmtree(input_dir)

    input_dir.parent.mkdir(parents=True, exist_ok=True)
    all_names = [p.name for p in list_images(payload.frame_dir)]
    full_match = len(all_names) == len(payload.frame_names) and all_names == payload.frame_names

    if full_match:
        # Fast path: all frames are used, so directory-level symlink/copy is safe.
        try:
            os.symlink(payload.frame_dir, input_dir, target_is_directory=True)
        except Exception:
            shutil.copytree(payload.frame_dir, input_dir)
        return input_dir

    # Subset path: only materialize selected frames to keep frame/mask counts aligned.
    input_dir.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(payload.frame_names):
        src = payload.frame_dir / name
        dst = input_dir / name
        if src.exists():
            try:
                os.symlink(src, dst)
                continue
            except Exception:
                try:
                    os.link(src, dst)
                    continue
                except Exception:
                    shutil.copy2(src, dst)
                    continue

        if idx < len(payload.frames):
            cv2.imwrite(str(dst), payload.frames[idx])

    return input_dir


def save_masks(mask_dir: Path, frame_names: list[str], masks_u8: list[np.ndarray]) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in zip(frame_names, masks_u8):
        cv2.imwrite(str(mask_dir / name), mask)


def find_best_image_dir(root: Path) -> tuple[Path | None, int]:
    if not root.exists():
        return None, 0

    candidate_dirs = [root] + [p for p in root.rglob("*") if p.is_dir()]
    best_dir = None
    best_count = 0
    for d in candidate_dirs:
        count = len(list_images(d))
        if count > best_count:
            best_count = count
            best_dir = d
    return best_dir, best_count


def read_frames_from_mp4(video_path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def run_propainter_once(
    repo_path: Path,
    input_frame_dir: Path,
    input_mask_dir: Path,
    output_root: Path,
    profile: dict[str, Any],
    target_fps: float,
    logger: logging.Logger,
) -> tuple[bool, str, list[np.ndarray], Path | None]:
    infer_script = repo_path / "inference_propainter.py"
    if not infer_script.exists():
        return False, "missing_inference_script", [], None

    cmd = [
        sys.executable,
        str(infer_script),
        "-i",
        str(input_frame_dir),
        "-m",
        str(input_mask_dir),
        "-o",
        str(output_root),
        "--neighbor_length",
        str(int(profile.get("neighbor_length", 10))),
        "--ref_stride",
        str(int(profile.get("ref_stride", 10))),
        "--subvideo_length",
        str(int(profile.get("subvideo_length", 80))),
        "--resize_ratio",
        str(float(profile.get("resize_ratio", 1.0))),
        "--mask_dilation",
        str(int(profile.get("mask_dilation", 4))),
        "--save_fps",
        str(int(round(target_fps))),
        "--save_frames",
    ]

    width = int(profile.get("width", -1))
    height = int(profile.get("height", -1))
    if width > 0 and height > 0:
        cmd.extend(["--width", str(width), "--height", str(height)])

    if bool(profile.get("fp16", True)):
        cmd.append("--fp16")

    completed = run_cmd(cmd=cmd, cwd=repo_path, logger=logger)
    raw_output_dir = output_root / input_frame_dir.name

    if completed.returncode != 0:
        err = (completed.stderr or "") + "\n" + (completed.stdout or "")
        err_l = err.lower()
        if "out of memory" in err_l or "cuda out of memory" in err_l or "oom" in err_l:
            return False, "oom", [], raw_output_dir
        return False, "failed", [], raw_output_dir

    frame_dir, count = find_best_image_dir(raw_output_dir)
    if frame_dir is not None and count > 0:
        frames = []
        for p in list_images(frame_dir):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        if frames:
            return True, "ok", frames, raw_output_dir

    mp4_candidates = sorted(raw_output_dir.rglob("*.mp4"))
    if mp4_candidates:
        frames = read_frames_from_mp4(mp4_candidates[0])
        if frames:
            return True, "ok_mp4", frames, raw_output_dir

    return False, "no_output_frames", [], raw_output_dir


def run_propainter_with_fallback(
    payload: DatasetPayload,
    dataset_name: str,
    masks_u8: list[np.ndarray],
    candidate_root: Path,
    propainter_repo: Path | None,
    part2_cfg: dict[str, Any],
    profile_override: dict[str, Any] | None,
    target_fps: float,
    logger: logging.Logger,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    prop_cfg = part2_cfg.get("propainter", {}) or {}
    fallback_cfg = part2_cfg.get("fallback", {}) or {}

    base_profile = {
        "name": "balanced",
        **(prop_cfg.get("balanced_profile", {}) or {}),
    }
    if profile_override:
        base_profile = {**base_profile, **profile_override}
        base_profile["name"] = str(profile_override.get("name", base_profile.get("name", "balanced")))
    degraded_profile = {
        "name": "degraded",
        **(prop_cfg.get("degraded_profile", {}) or {}),
    }

    meta: dict[str, Any] = {
        "status": "unknown",
        "profile": base_profile,
        "fallback_profile": None,
        "raw_output_dir": None,
    }

    if propainter_repo is None:
        method = str(fallback_cfg.get("inpaint_method", "ns")).lower()
        inpaint_method = "telea" if method == "telea" else "ns"
        restored = restore_frames(
            frames=payload.frames,
            masks_u8=masks_u8,
            inpaint_method=inpaint_method,
            inpaint_radius=float(fallback_cfg.get("inpaint_radius", 3.0)),
            temporal_window=int(fallback_cfg.get("temporal_window", 1)),
        )
        meta["status"] = "fallback_cv2"
        meta["reason"] = "propainter_unavailable"
        return restored, meta

    input_root = candidate_root / "_propainter_inputs"
    input_dir = ensure_unique_input_dir(payload=payload, dataset_name=dataset_name, target_root=input_root)

    mask_input_dir = candidate_root / "_propainter_masks" / dataset_name
    save_masks(mask_input_dir, payload.frame_names, masks_u8)

    raw_out_root = candidate_root / "_propainter_raw"
    ok, reason, frames, raw_dir = run_propainter_once(
        repo_path=propainter_repo,
        input_frame_dir=input_dir,
        input_mask_dir=mask_input_dir,
        output_root=raw_out_root,
        profile=base_profile,
        target_fps=target_fps,
        logger=logger,
    )
    meta["raw_output_dir"] = str(raw_dir) if raw_dir else None
    meta["reason"] = reason

    if ok:
        meta["status"] = "ok"
        return frames, meta

    if reason == "oom":
        meta["fallback_profile"] = degraded_profile
        ok2, reason2, frames2, raw_dir2 = run_propainter_once(
            repo_path=propainter_repo,
            input_frame_dir=input_dir,
            input_mask_dir=mask_input_dir,
            output_root=raw_out_root,
            profile=degraded_profile,
            target_fps=target_fps,
            logger=logger,
        )
        meta["raw_output_dir"] = str(raw_dir2) if raw_dir2 else meta["raw_output_dir"]
        meta["reason"] = reason2
        if ok2:
            meta["status"] = "ok_fallback_profile"
            meta["profile"] = degraded_profile
            return frames2, meta

    if bool(fallback_cfg.get("enable_cv2_inpaint_fallback", True)):
        method = str(fallback_cfg.get("inpaint_method", "ns")).lower()
        inpaint_method = "telea" if method == "telea" else "ns"
        restored = restore_frames(
            frames=payload.frames,
            masks_u8=masks_u8,
            inpaint_method=inpaint_method,
            inpaint_radius=float(fallback_cfg.get("inpaint_radius", 3.0)),
            temporal_window=int(fallback_cfg.get("temporal_window", 1)),
        )
        meta["status"] = "fallback_cv2"
        return restored, meta

    raise CandidateExecutionError(
        f"ProPainter failed for dataset={dataset_name}, reason={meta.get('reason')}"
    )


def maybe_fuse_with_phase1_prior(
    *,
    dataset_name: str,
    frame_names: list[str],
    frame_shape: tuple[int, int],
    backend_masks: list[np.ndarray],
    part2_cfg: dict[str, Any],
    phase1_exp_id: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    fusion_cfg = part2_cfg.get("mask_fusion", {}) or {}
    enabled = bool(fusion_cfg.get("enable_phase1_prior", False))
    meta: dict[str, Any] = {
        "enabled": enabled,
        "applied": False,
        "dataset": dataset_name,
    }
    if not enabled:
        meta["reason"] = "disabled"
        return backend_masks, meta

    skip_datasets = set(
        str(x).strip() for x in (fusion_cfg.get("skip_datasets", []) or []) if str(x).strip()
    )
    if dataset_name in skip_datasets:
        meta["reason"] = "dataset_skipped"
        return backend_masks, meta

    exp_token = str(fusion_cfg.get("phase1_exp_id", "")).strip() or phase1_exp_id
    if not exp_token:
        meta["reason"] = "phase1_exp_id_missing"
        return backend_masks, meta

    phase1_mask_dir = REPO_ROOT / "outputs" / "videos" / exp_token / dataset_name / "masks"
    prior_masks, load_meta = load_external_masks_by_names(
        mask_dir=phase1_mask_dir,
        frame_names=frame_names,
        frame_shape=frame_shape,
    )
    meta["phase1_exp_id"] = exp_token
    meta["phase1_mask_dir"] = str(phase1_mask_dir)
    meta["load_meta"] = load_meta

    if prior_masks is None:
        meta["reason"] = "phase1_masks_unavailable"
        return backend_masks, meta

    n = min(len(backend_masks), len(prior_masks))
    method = str(fusion_cfg.get("method", "intersection")).strip().lower()
    per_frame_low_ratio = float(fusion_cfg.get("per_frame_low_ratio", 1e-6))
    empty_fallback = str(fusion_cfg.get("empty_frame_fallback", "phase1")).strip().lower()
    mean_min_ratio = float(fusion_cfg.get("mean_min_ratio", 1e-4))
    low_mean_fallback = str(fusion_cfg.get("low_mean_fallback", "phase1")).strip().lower()

    fused: list[np.ndarray] = []
    low_ratio_fallback_count = 0
    for idx in range(n):
        bm = ((np.asarray(backend_masks[idx]) > 0).astype(np.uint8) * 255)
        pm = ((np.asarray(prior_masks[idx]) > 0).astype(np.uint8) * 255)
        fm = fuse_mask_pair(bm, pm, method=method)
        if float((fm > 0).mean()) <= per_frame_low_ratio:
            low_ratio_fallback_count += 1
            if empty_fallback in {"backend", "base"}:
                fm = bm
            elif empty_fallback in {"union", "or"}:
                fm = np.maximum(bm, pm)
            else:
                fm = pm
        fused.append(((fm > 0).astype(np.uint8) * 255))

    if len(backend_masks) > n:
        fused.extend([((np.asarray(m) > 0).astype(np.uint8) * 255) for m in backend_masks[n:]])

    ratio_after = compute_mean_mask_ratio(fused)
    if ratio_after <= mean_min_ratio:
        if low_mean_fallback in {"backend", "base"}:
            fused = [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in backend_masks]
        elif low_mean_fallback in {"union", "or"}:
            merged: list[np.ndarray] = []
            n2 = min(len(backend_masks), len(prior_masks))
            for idx in range(n2):
                bm = ((np.asarray(backend_masks[idx]) > 0).astype(np.uint8) * 255)
                pm = ((np.asarray(prior_masks[idx]) > 0).astype(np.uint8) * 255)
                merged.append(np.maximum(bm, pm))
            if len(backend_masks) > n2:
                merged.extend(
                    [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in backend_masks[n2:]]
                )
            fused = merged
        else:
            n2 = min(len(backend_masks), len(prior_masks))
            prior_only = [((np.asarray(prior_masks[idx]) > 0).astype(np.uint8) * 255) for idx in range(n2)]
            if len(backend_masks) > n2:
                prior_only.extend(
                    [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in backend_masks[n2:]]
                )
            fused = prior_only
        ratio_after = compute_mean_mask_ratio(fused)
        meta["low_mean_replaced"] = True
    else:
        meta["low_mean_replaced"] = False

    meta.update(
        {
            "applied": True,
            "method": method,
            "per_frame_low_ratio": per_frame_low_ratio,
            "empty_frame_fallback": empty_fallback,
            "mean_min_ratio": mean_min_ratio,
            "low_mean_fallback": low_mean_fallback,
            "low_ratio_fallback_count": int(low_ratio_fallback_count),
            "mean_mask_ratio_before": compute_mean_mask_ratio(backend_masks),
            "mean_mask_ratio_after": ratio_after,
        }
    )
    return fused, meta


def run_evaluation(
    config_path: Path,
    datasets: list[str],
    pred_root: Path,
    gt_root: Path,
    eval_exp_id: str,
    allow_missing_gt: bool,
    save_visualization: bool,
    logger: logging.Logger,
) -> tuple[Path, dict[str, Any]]:
    cmd = [
        sys.executable,
        "src/common/evaluate_experiment.py",
        "--config",
        str(config_path),
        "--exp-id",
        eval_exp_id,
        "--datasets",
        ",".join(datasets),
        "--pred-root",
        str(pred_root),
        "--gt-root",
        str(gt_root),
        "--allow-missing-gt",
        "true" if allow_missing_gt else "false",
        "--save-visualization",
        "true" if save_visualization else "false",
    ]
    logger.info("Evaluate: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    summary_path = REPO_ROOT / "outputs" / "metrics" / eval_exp_id / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Evaluation summary missing: {summary_path}")
    return summary_path, read_json(summary_path)


def metric_or_neg_inf(agg: dict[str, Any], key: str) -> float:
    value = agg.get(key, None)
    if value is None:
        return float("-inf")
    return float(value)


def stage_score(stage: str, agg: dict[str, Any], mean_mask_ratio: float) -> tuple[float, float, float, float, float]:
    jm = metric_or_neg_inf(agg, "JM")
    jr = metric_or_neg_inf(agg, "JR")
    psnr = metric_or_neg_inf(agg, "PSNR")
    ssim = metric_or_neg_inf(agg, "SSIM")

    if stage in {"B1", "B3"}:
        return (jm, jr, psnr, ssim, -abs(mean_mask_ratio - 0.1))
    return (psnr, ssim, jm, jr, -abs(mean_mask_ratio - 0.1))


def select_best(
    stage: str,
    entries: list[CandidateResult],
    score_datasets: list[str] | None = None,
) -> CandidateResult:
    if not entries:
        raise ValueError(f"No candidate results to select from in stage {stage}")

    def score(entry: CandidateResult) -> tuple[float, float, float, float, float]:
        if score_datasets:
            ratios = [
                (entry.mask_stats.get(ds, {}) or {}).get("mean_mask_ratio", 0.0) for ds in score_datasets
            ]
        else:
            ratios = [v.get("mean_mask_ratio", 0.0) for v in entry.mask_stats.values()]
        mean_ratio = float(np.mean(np.array(ratios, dtype=np.float32))) if ratios else 0.0
        agg = dataset_metric_aggregate(per_dataset=entry.per_dataset, datasets_for_scoring=score_datasets)
        return stage_score(stage, agg, mean_ratio)

    return max(entries, key=score)


def copy_b_best(best_candidate_root: Path, final_root: Path, datasets: list[str]) -> None:
    for ds in datasets:
        src = best_candidate_root / ds
        if not src.exists():
            raise RuntimeError(f"B-best candidate missing dataset output: {src}")
        dst = final_root / ds
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def classify_failure_case(
    dataset: str,
    psnr: float,
    edge_diff: float,
    texture_ratio: float,
    backend_fallback: bool,
    propainter_fallback: bool,
) -> str:
    if backend_fallback:
        return "mask_backend_fallback"
    if propainter_fallback:
        return "propainter_profile_fallback"
    if psnr < 24.0 and edge_diff > 0.06:
        return "boundary_residue"
    if psnr < 24.0 and texture_ratio < 0.75:
        return "texture_loss"
    if dataset == "wild":
        return "wild_domain_gap_suspected"
    return "temporal_flicker_suspected"


def build_failure_case_index(
    pred_root: Path,
    gt_root: Path,
    datasets: list[str],
    out_dir: Path,
    backend_meta: dict[str, dict[str, Any]],
    propainter_meta: dict[str, dict[str, Any]],
    top_k: int = 3,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows_explained: list[dict[str, Any]] = []

    for ds in datasets:
        pred_dir = pred_root / ds / "frames"
        gt_dir = gt_root / ds / "frames"
        pred_paths = list_images(pred_dir)
        gt_map = {p.name: p for p in list_images(gt_dir)}
        if not pred_paths or not gt_map:
            continue

        scores: list[tuple[float, Path, Path]] = []
        for p in pred_paths:
            g = gt_map.get(p.name)
            if g is None:
                continue
            pi = cv2.imread(str(p), cv2.IMREAD_COLOR)
            gi = cv2.imread(str(g), cv2.IMREAD_COLOR)
            if pi is None or gi is None:
                continue
            if pi.shape != gi.shape:
                gi = cv2.resize(gi, (pi.shape[1], pi.shape[0]), interpolation=cv2.INTER_LINEAR)
            psnr = float(cv2.PSNR(pi, gi))
            scores.append((psnr, p, g))

        if not scores:
            continue

        scores.sort(key=lambda x: x[0])
        chosen = scores[: max(1, top_k)]
        for rank, (psnr, p, g) in enumerate(chosen, start=1):
            pi = cv2.imread(str(p), cv2.IMREAD_COLOR)
            gi = cv2.imread(str(g), cv2.IMREAD_COLOR)
            if pi is None or gi is None:
                continue
            if pi.shape != gi.shape:
                gi = cv2.resize(gi, (pi.shape[1], pi.shape[0]), interpolation=cv2.INTER_LINEAR)

            diff = cv2.absdiff(pi, gi)
            edge_p = cv2.Canny(pi, 80, 160)
            edge_g = cv2.Canny(gi, 80, 160)
            edge_diff = float(np.mean(np.abs(edge_p.astype(np.float32) - edge_g.astype(np.float32))) / 255.0)
            texture_ratio = float(diff.std() / (gi.std() + 1e-6))

            panel = np.concatenate([pi, gi, diff], axis=1)
            out_img = out_dir / f"{ds}_{p.stem}_rank{rank}_psnr{psnr:.2f}.png"
            cv2.imwrite(str(out_img), panel)

            rows.append(
                {
                    "dataset": ds,
                    "frame": p.name,
                    "rank": rank,
                    "psnr": psnr,
                    "compare_image": str(out_img),
                }
            )

            b_meta = backend_meta.get(ds, {})
            p_meta = propainter_meta.get(ds, {})
            explanation = classify_failure_case(
                dataset=ds,
                psnr=psnr,
                edge_diff=edge_diff,
                texture_ratio=texture_ratio,
                backend_fallback=bool(b_meta.get("fallback_used", False)),
                propainter_fallback=str(p_meta.get("status", "")).startswith("ok_fallback")
                or str(p_meta.get("status", "")) == "fallback_cv2",
            )
            rows_explained.append(
                {
                    "dataset": ds,
                    "frame": p.name,
                    "rank": rank,
                    "psnr": psnr,
                    "edge_diff": edge_diff,
                    "texture_ratio": texture_ratio,
                    "explanation": explanation,
                    "compare_image": str(out_img),
                }
            )

    csv_path = out_dir / "failure_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "frame", "rank", "psnr", "compare_image"])
        writer.writeheader()
        writer.writerows(rows)

    explained_csv_path = out_dir / "failure_cases_explained.csv"
    with explained_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "frame",
                "rank",
                "psnr",
                "edge_diff",
                "texture_ratio",
                "explanation",
                "compare_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_explained)

    return csv_path, explained_csv_path


def write_ablation_outputs(
    exp_metrics_dir: Path,
    all_results: list[CandidateResult],
    stage_best_map: dict[str, CandidateResult],
    final_best: CandidateResult,
) -> tuple[Path, Path]:
    exp_metrics_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for r in all_results:
        ratios = [v.get("mean_mask_ratio", 0.0) for v in r.mask_stats.values()]
        mean_ratio = float(np.mean(np.array(ratios, dtype=np.float32))) if ratios else 0.0
        rows.append(
            {
                "stage": r.spec.stage,
                "candidate": r.spec.name,
                "mask_backend": r.spec.mask_backend,
                "mask_variant": r.spec.mask_variant,
                "neighbor_length": r.spec.neighbor_length,
                "ref_stride": r.spec.ref_stride,
                "subvideo_length": r.spec.subvideo_length,
                "resize_ratio": r.spec.resize_ratio,
                "mask_dilation": r.spec.mask_dilation,
                "fp16": int(r.spec.fp16),
                "JM": r.aggregate.get("JM"),
                "JR": r.aggregate.get("JR"),
                "PSNR": r.aggregate.get("PSNR"),
                "SSIM": r.aggregate.get("SSIM"),
                "mean_mask_ratio": mean_ratio,
                "pred_root": str(r.candidate_root),
                "eval_exp_id": r.eval_exp_id,
                "is_stage_best": int(stage_best_map.get(r.spec.stage) is r),
                "is_final_best": int(final_best is r),
            }
        )

    csv_path = exp_metrics_dir / "phase2_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "candidate",
                "mask_backend",
                "mask_variant",
                "neighbor_length",
                "ref_stride",
                "subvideo_length",
                "resize_ratio",
                "mask_dilation",
                "fp16",
                "JM",
                "JR",
                "PSNR",
                "SSIM",
                "mean_mask_ratio",
                "pred_root",
                "eval_exp_id",
                "is_stage_best",
                "is_final_best",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    selection = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "stage_best": {k: asdict(v.spec) for k, v in stage_best_map.items()},
        "final_best": asdict(final_best.spec),
    }
    selection_path = exp_metrics_dir / "phase2_selection.json"
    write_json(selection_path, selection)

    return csv_path, selection_path


def write_a_vs_b_comparison(
    exp_metrics_dir: Path,
    phase1_exp_id: str,
    phase2_summary: dict[str, Any],
) -> tuple[Path | None, dict[str, Any]]:
    phase1_summary_path = REPO_ROOT / "outputs" / "metrics" / phase1_exp_id / "summary.json"
    if not phase1_summary_path.exists():
        return None, {"status": "phase1_summary_missing", "phase1_exp_id": phase1_exp_id}

    phase1_summary = read_json(phase1_summary_path)
    a_ds = phase1_summary.get("datasets", {}) or {}
    b_ds = phase2_summary.get("datasets", {}) or {}

    rows: list[dict[str, Any]] = []
    all_names = sorted(set(a_ds.keys()) | set(b_ds.keys()))
    for ds in all_names:
        am = (a_ds.get(ds, {}) or {}).get("metrics", {}) or {}
        bm = (b_ds.get(ds, {}) or {}).get("metrics", {}) or {}
        row = {
            "dataset": ds,
            "A_JM": am.get("JM"),
            "B_JM": bm.get("JM"),
            "delta_JM": None if am.get("JM") is None or bm.get("JM") is None else float(bm.get("JM")) - float(am.get("JM")),
            "A_JR": am.get("JR"),
            "B_JR": bm.get("JR"),
            "delta_JR": None if am.get("JR") is None or bm.get("JR") is None else float(bm.get("JR")) - float(am.get("JR")),
            "A_PSNR": am.get("PSNR"),
            "B_PSNR": bm.get("PSNR"),
            "delta_PSNR": None
            if am.get("PSNR") is None or bm.get("PSNR") is None
            else float(bm.get("PSNR")) - float(am.get("PSNR")),
            "A_SSIM": am.get("SSIM"),
            "B_SSIM": bm.get("SSIM"),
            "delta_SSIM": None
            if am.get("SSIM") is None or bm.get("SSIM") is None
            else float(bm.get("SSIM")) - float(am.get("SSIM")),
        }
        rows.append(row)

    a_agg = phase1_summary.get("aggregate", {}) or {}
    b_agg = phase2_summary.get("aggregate", {}) or {}
    rows.append(
        {
            "dataset": "__aggregate__",
            "A_JM": a_agg.get("JM"),
            "B_JM": b_agg.get("JM"),
            "delta_JM": None if a_agg.get("JM") is None or b_agg.get("JM") is None else float(b_agg.get("JM")) - float(a_agg.get("JM")),
            "A_JR": a_agg.get("JR"),
            "B_JR": b_agg.get("JR"),
            "delta_JR": None if a_agg.get("JR") is None or b_agg.get("JR") is None else float(b_agg.get("JR")) - float(a_agg.get("JR")),
            "A_PSNR": a_agg.get("PSNR"),
            "B_PSNR": b_agg.get("PSNR"),
            "delta_PSNR": None
            if a_agg.get("PSNR") is None or b_agg.get("PSNR") is None
            else float(b_agg.get("PSNR")) - float(a_agg.get("PSNR")),
            "A_SSIM": a_agg.get("SSIM"),
            "B_SSIM": b_agg.get("SSIM"),
            "delta_SSIM": None
            if a_agg.get("SSIM") is None or b_agg.get("SSIM") is None
            else float(b_agg.get("SSIM")) - float(a_agg.get("SSIM")),
        }
    )

    out_path = exp_metrics_dir / "phase2_a_vs_b.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "A_JM",
                "B_JM",
                "delta_JM",
                "A_JR",
                "B_JR",
                "delta_JR",
                "A_PSNR",
                "B_PSNR",
                "delta_PSNR",
                "A_SSIM",
                "B_SSIM",
                "delta_SSIM",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return out_path, {"status": "ok", "phase1_exp_id": phase1_exp_id, "phase1_summary": str(phase1_summary_path)}


def write_acceptance_report(
    report_path: Path,
    exp_id: str,
    aggregate: dict[str, Any],
    stage_best_map: dict[str, CandidateResult],
    final_best: CandidateResult,
    per_dataset: dict[str, Any],
    phase1_exp_id: str,
    a_vs_b_path: Path | None,
    failure_explained_csv: Path,
    seed: int,
    strict_dual_run: bool,
) -> None:
    lines: list[str] = []
    lines.append(f"# Phase 2 Acceptance Report: `{exp_id}`")
    lines.append("")
    lines.append("## Final Aggregate")
    lines.append("")
    lines.append("| JM | JR | PSNR | SSIM |")
    lines.append("| ---: | ---: | ---: | ---: |")
    lines.append(
        f"| {aggregate.get('JM')} | {aggregate.get('JR')} | {aggregate.get('PSNR')} | {aggregate.get('SSIM')} |"
    )
    lines.append("")

    lines.append("## Stage Best")
    lines.append("")
    lines.append("| Stage | Candidate | Backend | Variant | Neighbor | RefStride | Subvideo | Resize |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for stage in ["B1", "B2", "B3", "B4", "B5"]:
        r = stage_best_map.get(stage)
        if r is None:
            continue
        s = r.spec
        lines.append(
            f"| {stage} | {s.name} | {s.mask_backend} | {s.mask_variant} | {s.neighbor_length} | {s.ref_stride} | {s.subvideo_length} | {s.resize_ratio} |"
        )
    lines.append("")

    lines.append("## Final B-best")
    lines.append("")
    s = final_best.spec
    lines.append(
        f"- `{s.name}`: backend={s.mask_backend}, variant={s.mask_variant}, neighbor={s.neighbor_length}, ref_stride={s.ref_stride}, subvideo={s.subvideo_length}, resize={s.resize_ratio}, fp16={s.fp16}"
    )
    lines.append("")

    lines.append("## Per-Dataset Metrics")
    lines.append("")
    lines.append("| Dataset | JM | JR | PSNR | SSIM |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for ds_name, ds_payload in per_dataset.items():
        metrics = ds_payload.get("metrics", {}) if isinstance(ds_payload, dict) else {}
        lines.append(
            f"| {ds_name} | {metrics.get('JM')} | {metrics.get('JR')} | {metrics.get('PSNR')} | {metrics.get('SSIM')} |"
        )
    lines.append("")

    lines.append("## A-best vs B-best")
    lines.append("")
    lines.append(f"- Phase1 reference exp: `{phase1_exp_id}`")
    if a_vs_b_path is not None:
        lines.append(f"- Comparison CSV: `{a_vs_b_path}`")
    else:
        lines.append("- Comparison CSV: unavailable (Phase1 summary missing)")
    lines.append("")

    lines.append("## Acceptance Checks")
    lines.append("")
    lines.append(f"- Seed set and logged: `{seed}`")
    lines.append(f"- strict_dual_run: `{str(strict_dual_run).lower()}`")
    lines.append(f"- Failure case explanations: `{failure_explained_csv}`")
    lines.append(
        "- Acceptance policy: completeness & reproducibility first; if B-best is not globally better than A-best, report domain gap and failure reasons."
    )
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 SOTA baseline (B1-B5): SAM2/TrackAnything + ProPainter."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--datasets", type=str, default="mandatory", help="mandatory | all | csv")
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/videos"))

    parser.add_argument("--mask-models", type=str, default="sam2,trackanything")
    parser.add_argument("--prompt-detector", type=str, default=None, help="yolo|maskrcnn")
    parser.add_argument("--auto-install-missing", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--phase1-exp-id", type=str, default=None)
    parser.add_argument("--strict-dual-run", type=str, default=None)
    parser.add_argument("--stages", type=str, default="B1,B2,B3,B4,B5")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def run_candidate(
    spec: CandidateSpec,
    datasets: list[str],
    dataset_payloads: dict[str, DatasetPayload],
    prompt_instances_cache: dict[str, list[list[dict[str, Any]]]],
    prompt_meta_cache: dict[str, dict[str, Any]],
    backend_env: dict[str, dict[str, Any]],
    device: str,
    backend_mask_cache: dict[tuple[str, str], tuple[list[np.ndarray], dict[str, Any]]],
    out_root: Path,
    part2_cfg: dict[str, Any],
    phase1_exp_id: str,
    propainter_repo: Path | None,
    target_fps: float,
    logger: logging.Logger,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    candidate_root = out_root / spec.stage / spec.candidate_id
    candidate_root.mkdir(parents=True, exist_ok=True)

    mask_post_cfg = part2_cfg.get("mask_post", {}) or {}

    mask_stats_map: dict[str, dict[str, Any]] = {}
    backend_meta_map: dict[str, dict[str, Any]] = {}
    propainter_meta_map: dict[str, dict[str, Any]] = {}

    for ds in datasets:
        payload = dataset_payloads[ds]
        cache_key = (spec.mask_backend, ds)
        if cache_key in backend_mask_cache:
            base_masks, b_meta = backend_mask_cache[cache_key]
            masks_u8 = [m.copy() for m in base_masks]
            backend_meta = dict(b_meta)
        else:
            detector_instances = prompt_instances_cache[ds]
            prompt_frame_idx = int((prompt_meta_cache.get(ds, {}) or {}).get("frame_idx", 0))
            prompt_boxes = [
                tuple(int(v) for v in box)
                for box in (prompt_meta_cache.get(ds, {}) or {}).get("boxes", [])
                if isinstance(box, (list, tuple)) and len(box) == 4
            ]
            masks_u8, backend_meta = generate_backend_masks_with_priority(
                backend=spec.mask_backend,
                payload=payload,
                detector_instances=detector_instances,
                prompt_frame_idx=prompt_frame_idx,
                prompt_boxes=prompt_boxes,
                part2_cfg=part2_cfg,
                backend_env=backend_env,
                device=device,
                logger=logger,
            )
            backend_mask_cache[cache_key] = ([m.copy() for m in masks_u8], dict(backend_meta))

        if spec.mask_variant == "refined":
            masks_u8 = refine_masks(
                masks_u8,
                morph_kernel=int(mask_post_cfg.get("refined_kernel", 5)),
                temporal_window=int(mask_post_cfg.get("temporal_smooth_window", 1)),
            )
            backend_meta["variant_refined"] = True
        else:
            backend_meta["variant_refined"] = False

        original_mask_count = len(masks_u8)
        masks_u8 = align_masks_to_frame_count(
            masks_u8=masks_u8,
            frame_count=len(payload.frame_names),
            frame_shape=payload.frames[0].shape[:2],
        )
        if original_mask_count != len(masks_u8):
            backend_meta["mask_count_aligned"] = {
                "before": int(original_mask_count),
                "after": int(len(masks_u8)),
            }

        masks_u8, fusion_meta = maybe_fuse_with_phase1_prior(
            dataset_name=ds,
            frame_names=payload.frame_names,
            frame_shape=payload.frames[0].shape[:2],
            backend_masks=masks_u8,
            part2_cfg=part2_cfg,
            phase1_exp_id=phase1_exp_id,
        )
        backend_meta["phase1_mask_fusion"] = fusion_meta

        mask_ratio = compute_mean_mask_ratio(masks_u8)
        mask_stats_map[ds] = {
            "mean_mask_ratio": mask_ratio,
            "frame_count": len(masks_u8),
        }

        restored_frames, p_meta = run_propainter_with_fallback(
            payload=payload,
            dataset_name=ds,
            masks_u8=masks_u8,
            candidate_root=candidate_root,
            propainter_repo=propainter_repo,
            part2_cfg=part2_cfg,
            profile_override={
                "name": spec.name,
                "neighbor_length": spec.neighbor_length,
                "ref_stride": spec.ref_stride,
                "subvideo_length": spec.subvideo_length,
                "resize_ratio": spec.resize_ratio,
                "mask_dilation": spec.mask_dilation,
                "fp16": spec.fp16,
            },
            target_fps=target_fps,
            logger=logger,
        )

        if not restored_frames:
            raise CandidateExecutionError(
                f"No restored frames produced for dataset={ds}, candidate={spec.name}"
            )

        n = min(len(payload.frame_names), len(masks_u8), len(restored_frames))
        if n <= 0:
            raise CandidateExecutionError(
                f"Invalid output lengths for dataset={ds}, candidate={spec.name}"
            )

        write_dataset_outputs(
            out_root=candidate_root,
            dataset_name=ds,
            frame_names=payload.frame_names[:n],
            restored_frames=restored_frames[:n],
            masks_u8=masks_u8[:n],
            target_fps=target_fps,
            save_mp4=bool(part2_cfg.get("save_mp4", False)),
        )

        backend_meta_map[ds] = backend_meta
        propainter_meta_map[ds] = p_meta
        mask_stats_map[ds]["backend_fallback_used"] = bool(backend_meta.get("fallback_used", False))
        mask_stats_map[ds]["propainter_status"] = p_meta.get("status")

    write_json(
        candidate_root / "candidate_config.json",
        {
            "candidate": asdict(spec),
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "mask_stats": mask_stats_map,
            "backend_meta": backend_meta_map,
            "propainter_meta": propainter_meta_map,
        },
    )

    logger.info(
        "Candidate complete: stage=%s name=%s root=%s",
        spec.stage,
        spec.name,
        candidate_root,
    )

    return candidate_root, mask_stats_map, backend_meta_map, propainter_meta_map


def main() -> None:
    args = parse_args()
    exp_id = args.exp_id or f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger, log_path = setup_logger(exp_id)
    logger.info("Phase 2 SOTA start | exp_id=%s", exp_id)

    config_path = resolve_repo_path(Path(args.config))
    config = read_yaml(config_path)
    part1_cfg = config.get("part1", {}) or {}
    part2_cfg = config.get("part2", {}) or {}
    default_phase1_exp = str(part2_cfg.get("phase1_reference_exp_id", "phase1_fix_acceptance_20260323"))
    phase1_exp_id = args.phase1_exp_id or default_phase1_exp

    ds_map, all_names, mandatory_names = collect_dataset_cfg(config)
    selected_datasets = resolve_dataset_names(args.datasets, all_names, mandatory_names)
    logger.info("Datasets: %s", selected_datasets)

    selection_cfg = part2_cfg.get("selection", {}) or {}
    exclude_for_scoring = set(
        str(x).strip() for x in (selection_cfg.get("exclude_datasets", []) or []) if str(x).strip()
    )
    stage_score_datasets = [ds for ds in selected_datasets if ds not in exclude_for_scoring]
    if not stage_score_datasets:
        stage_score_datasets = list(selected_datasets)
    logger.info(
        "Stage selection datasets: %s (excluded=%s)",
        stage_score_datasets,
        sorted(exclude_for_scoring),
    )

    stages = parse_stages(args.stages)
    logger.info("Stages to run: %s", stages)

    runtime_cfg = part2_cfg.get("runtime", {}) or {}
    auto_install_missing = str2bool(
        args.auto_install_missing,
        default=bool(runtime_cfg.get("auto_install_missing", True)),
    )
    strict_dual_run = str2bool(
        args.strict_dual_run,
        default=bool(runtime_cfg.get("strict_dual_run", False)),
    )

    seed_value = int(args.seed) if args.seed is not None else int(config.get("project", {}).get("seed", 42))
    seed_meta = set_global_seed(seed_value, logger)

    device = resolve_device(runtime_cfg=runtime_cfg, logger=logger)
    logger.info("Device: %s", device)

    default_prompt_detector = str((part2_cfg.get("prompt", {}) or {}).get("detector", "yolo"))
    prompt_detector = parse_prompt_detector(args.prompt_detector or default_prompt_detector)

    mask_backends = parse_backends(args.mask_models)

    external_root = REPO_ROOT / "outputs" / "external" / "part2"
    external_root.mkdir(parents=True, exist_ok=True)

    backend_env = probe_backend_environment(
        backends=mask_backends,
        part2_cfg=part2_cfg,
        external_root=external_root,
        auto_install_missing=auto_install_missing,
        logger=logger,
    )

    propainter_repo, propainter_meta = ensure_propainter_ready(
        part2_cfg=part2_cfg,
        external_root=external_root,
        auto_install_missing=auto_install_missing,
        logger=logger,
    )

    if strict_dual_run:
        missing = [k for k in mask_backends if not bool((backend_env.get(k, {}) or {}).get("ready", False))]
        if missing:
            raise RuntimeError(
                f"strict_dual_run=true but backend not ready: {missing}. backend_env={backend_env}"
            )

    target_fps = float(config.get("preprocess", {}).get("target_fps", 24.0))
    gt_root = resolve_repo_path(Path(config.get("paths", {}).get("gt_data_dir", "data/gt")))

    eval_cfg = config.get("evaluation", {}) or {}
    allow_missing_gt = bool(eval_cfg.get("allow_missing_gt", True))
    save_visualization = bool(eval_cfg.get("save_visualization", True))

    pred_root_base = resolve_repo_path(Path(args.pred_root))
    exp_pred_root = pred_root_base / exp_id
    candidate_root = exp_pred_root / "_candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)

    dynamic_classes = set(
        x.strip().lower()
        for x in (part2_cfg.get("prompt", {}) or {}).get(
            "dynamic_classes",
            ["person", "bicycle", "motorcycle", "car", "bus", "truck"],
        )
        if str(x).strip()
    )

    max_frames = int(args.max_frames) if args.max_frames is not None and int(args.max_frames) > 0 else None

    dataset_payloads: dict[str, DatasetPayload] = {}
    prompt_instances_cache: dict[str, list[list[dict[str, Any]]]] = {}
    prompt_meta: dict[str, dict[str, Any]] = {}

    for ds in selected_datasets:
        payload = load_dataset_payload(ds, ds_map[ds], max_frames=max_frames)
        dataset_payloads[ds] = payload
        logger.info("Loaded %s frames for %s", len(payload.frames), ds)

        instances, actual_detector = prepare_prompt_instances(
            payload=payload,
            part1_cfg=part1_cfg,
            dynamic_classes=dynamic_classes,
            prompt_detector=prompt_detector,
            device=device,
            auto_install_missing=auto_install_missing,
            logger=logger,
        )
        prompt_instances_cache[ds] = instances

        prompt_cfg = part2_cfg.get("prompt", {}) or {}
        prompt_info = build_auto_prompts(
            instances_per_frame=instances,
            frames=payload.frames,
            max_prompts=int(prompt_cfg.get("max_prompts", 3)),
        )
        prompt_meta[ds] = {
            "detector_requested": prompt_detector,
            "detector_used": actual_detector,
            **prompt_info,
        }

    # Candidate grids
    prop_cfg = part2_cfg.get("propainter", {}) or {}
    balanced = {
        "neighbor_length": int((prop_cfg.get("balanced_profile", {}) or {}).get("neighbor_length", 10)),
        "ref_stride": int((prop_cfg.get("balanced_profile", {}) or {}).get("ref_stride", 10)),
        "subvideo_length": int((prop_cfg.get("balanced_profile", {}) or {}).get("subvideo_length", 80)),
        "resize_ratio": float((prop_cfg.get("balanced_profile", {}) or {}).get("resize_ratio", 1.0)),
        "mask_dilation": int((prop_cfg.get("balanced_profile", {}) or {}).get("mask_dilation", 4)),
        "fp16": bool((prop_cfg.get("balanced_profile", {}) or {}).get("fp16", True)),
    }

    b4_grid = (prop_cfg.get("profile_grid", []) or [])
    if not isinstance(b4_grid, list) or not b4_grid:
        b4_grid = [
            {
                "name": "balanced",
                "neighbor_length": balanced["neighbor_length"],
                "ref_stride": balanced["ref_stride"],
                "subvideo_length": balanced["subvideo_length"],
                "resize_ratio": balanced["resize_ratio"],
                "mask_dilation": balanced["mask_dilation"],
                "fp16": balanced["fp16"],
            },
            {
                "name": "balanced_lowmem",
                "neighbor_length": max(4, balanced["neighbor_length"] - 2),
                "ref_stride": balanced["ref_stride"] + 4,
                "subvideo_length": max(24, balanced["subvideo_length"] - 16),
                "resize_ratio": min(1.0, balanced["resize_ratio"] * 0.85),
                "mask_dilation": balanced["mask_dilation"],
                "fp16": True,
            },
        ]

    all_results: list[CandidateResult] = []
    stage_best_map: dict[str, CandidateResult] = {}
    backend_mask_cache: dict[tuple[str, str], tuple[list[np.ndarray], dict[str, Any]]] = {}

    def execute_stage(stage: str, specs: list[CandidateSpec], stage_datasets: list[str]) -> CandidateResult:
        stage_results: list[CandidateResult] = []

        for spec in specs:
            try:
                c_root, mask_stats, backend_meta_map, propainter_meta_map = run_candidate(
                    spec=spec,
                    datasets=stage_datasets,
                    dataset_payloads=dataset_payloads,
                    prompt_instances_cache=prompt_instances_cache,
                    prompt_meta_cache=prompt_meta,
                    backend_env=backend_env,
                    device=device,
                    backend_mask_cache=backend_mask_cache,
                    out_root=candidate_root,
                    part2_cfg=part2_cfg,
                    phase1_exp_id=phase1_exp_id,
                    propainter_repo=propainter_repo,
                    target_fps=target_fps,
                    logger=logger,
                )
            except Exception as e:
                logger.warning(
                    "Candidate failed: stage=%s candidate=%s reason=%s",
                    stage,
                    spec.name,
                    str(e),
                )
                logger.debug(traceback.format_exc())
                if strict_dual_run and stage == "B1":
                    raise
                continue

            eval_exp_id = f"{exp_id}__{stage}__{spec.candidate_id}"
            summary_path, summary = run_evaluation(
                config_path=config_path,
                datasets=stage_datasets,
                pred_root=c_root,
                gt_root=gt_root,
                eval_exp_id=eval_exp_id,
                allow_missing_gt=allow_missing_gt,
                save_visualization=save_visualization,
                logger=logger,
            )
            result = CandidateResult(
                spec=spec,
                candidate_root=c_root,
                eval_exp_id=eval_exp_id,
                summary_path=summary_path,
                aggregate=summary.get("aggregate", {}),
                per_dataset=summary.get("datasets", {}),
                mask_stats=mask_stats,
                backend_meta=backend_meta_map,
                propainter_meta=propainter_meta_map,
            )
            stage_results.append(result)
            all_results.append(result)

            logger.info(
                "[%s] candidate=%s -> JM=%.4f JR=%.4f PSNR=%.4f SSIM=%.4f",
                stage,
                spec.name,
                metric_or_neg_inf(result.aggregate, "JM"),
                metric_or_neg_inf(result.aggregate, "JR"),
                metric_or_neg_inf(result.aggregate, "PSNR"),
                metric_or_neg_inf(result.aggregate, "SSIM"),
            )

        if not stage_results:
            raise RuntimeError(f"No successful candidate in {stage}")

        if stage == "B1" and strict_dual_run:
            seen_backends = {r.spec.mask_backend for r in stage_results}
            required = set(mask_backends)
            if not required.issubset(seen_backends):
                raise RuntimeError(
                    f"strict_dual_run=true but B1 missing backend runs. required={required}, seen={seen_backends}"
                )

        best = select_best(stage, stage_results, score_datasets=stage_score_datasets)
        stage_best_map[stage] = best
        logger.info("[%s] best candidate=%s", stage, best.spec.name)
        return best

    best_b1: CandidateResult | None = None
    best_b2: CandidateResult | None = None
    best_b3: CandidateResult | None = None
    best_b4: CandidateResult | None = None
    best_b5: CandidateResult | None = None

    if "B1" in stages:
        b1_specs = [
            CandidateSpec(
                stage="B1",
                name=backend,
                mask_backend=backend,
                mask_variant="coarse",
                neighbor_length=balanced["neighbor_length"],
                ref_stride=balanced["ref_stride"],
                subvideo_length=balanced["subvideo_length"],
                resize_ratio=balanced["resize_ratio"],
                mask_dilation=balanced["mask_dilation"],
                fp16=balanced["fp16"],
            )
            for backend in mask_backends
        ]
        best_b1 = execute_stage("B1", b1_specs, selected_datasets)

    if "B2" in stages:
        anchor = best_b1 or stage_best_map.get("B1")
        if anchor is None:
            raise RuntimeError("B2 requires B1 best result")
        b2_specs = [
            CandidateSpec(
                stage="B2",
                name="propainter_base",
                mask_backend=anchor.spec.mask_backend,
                mask_variant="coarse",
                neighbor_length=balanced["neighbor_length"],
                ref_stride=balanced["ref_stride"],
                subvideo_length=balanced["subvideo_length"],
                resize_ratio=balanced["resize_ratio"],
                mask_dilation=balanced["mask_dilation"],
                fp16=balanced["fp16"],
            )
        ]
        best_b2 = execute_stage("B2", b2_specs, selected_datasets)

    if "B3" in stages:
        anchor = best_b2 or stage_best_map.get("B2")
        if anchor is None:
            raise RuntimeError("B3 requires B2 best result")
        b3_specs = [
            CandidateSpec(
                stage="B3",
                name="coarse_mask",
                mask_backend=anchor.spec.mask_backend,
                mask_variant="coarse",
                neighbor_length=anchor.spec.neighbor_length,
                ref_stride=anchor.spec.ref_stride,
                subvideo_length=anchor.spec.subvideo_length,
                resize_ratio=anchor.spec.resize_ratio,
                mask_dilation=anchor.spec.mask_dilation,
                fp16=anchor.spec.fp16,
            ),
            CandidateSpec(
                stage="B3",
                name="refined_mask",
                mask_backend=anchor.spec.mask_backend,
                mask_variant="refined",
                neighbor_length=anchor.spec.neighbor_length,
                ref_stride=anchor.spec.ref_stride,
                subvideo_length=anchor.spec.subvideo_length,
                resize_ratio=anchor.spec.resize_ratio,
                mask_dilation=anchor.spec.mask_dilation,
                fp16=anchor.spec.fp16,
            ),
        ]
        best_b3 = execute_stage("B3", b3_specs, selected_datasets)

    if "B4" in stages:
        anchor = best_b3 or stage_best_map.get("B3")
        if anchor is None:
            raise RuntimeError("B4 requires B3 best result")
        b4_specs: list[CandidateSpec] = []
        for item in b4_grid:
            if not isinstance(item, dict):
                continue
            name = sanitize_name(str(item.get("name", f"profile_{len(b4_specs)}")))
            b4_specs.append(
                CandidateSpec(
                    stage="B4",
                    name=name,
                    mask_backend=anchor.spec.mask_backend,
                    mask_variant=anchor.spec.mask_variant,
                    neighbor_length=int(item.get("neighbor_length", anchor.spec.neighbor_length)),
                    ref_stride=int(item.get("ref_stride", anchor.spec.ref_stride)),
                    subvideo_length=int(item.get("subvideo_length", anchor.spec.subvideo_length)),
                    resize_ratio=float(item.get("resize_ratio", anchor.spec.resize_ratio)),
                    mask_dilation=int(item.get("mask_dilation", anchor.spec.mask_dilation)),
                    fp16=bool(item.get("fp16", anchor.spec.fp16)),
                )
            )
        if not b4_specs:
            raise RuntimeError("B4 profile_grid resolved to empty candidate list")
        best_b4 = execute_stage("B4", b4_specs, selected_datasets)

    if "B5" in stages:
        anchor = best_b4 or stage_best_map.get("B4")
        if anchor is None:
            raise RuntimeError("B5 requires B4 best result")
        b5_datasets = mandatory_names if args.datasets.strip().lower() == "mandatory" else selected_datasets
        b5_specs = [
            CandidateSpec(
                stage="B5",
                name="b_best_finalize",
                mask_backend=anchor.spec.mask_backend,
                mask_variant=anchor.spec.mask_variant,
                neighbor_length=anchor.spec.neighbor_length,
                ref_stride=anchor.spec.ref_stride,
                subvideo_length=anchor.spec.subvideo_length,
                resize_ratio=anchor.spec.resize_ratio,
                mask_dilation=anchor.spec.mask_dilation,
                fp16=anchor.spec.fp16,
            )
        ]
        best_b5 = execute_stage("B5", b5_specs, b5_datasets)

    final_best = best_b5 or best_b4 or best_b3 or best_b2 or best_b1
    if final_best is None:
        raise RuntimeError("Phase 2 finished without any successful candidate")

    # Materialize B-best to outputs/videos/<exp_id>/<dataset> for downstream compatibility.
    copy_b_best(
        best_candidate_root=final_best.candidate_root,
        final_root=exp_pred_root,
        datasets=(mandatory_names if args.datasets.strip().lower() == "mandatory" else selected_datasets),
    )

    # Final evaluation under exp_id.
    final_datasets = mandatory_names if args.datasets.strip().lower() == "mandatory" else selected_datasets
    final_summary_path, final_summary = run_evaluation(
        config_path=config_path,
        datasets=final_datasets,
        pred_root=exp_pred_root,
        gt_root=gt_root,
        eval_exp_id=exp_id,
        allow_missing_gt=allow_missing_gt,
        save_visualization=save_visualization,
        logger=logger,
    )

    exp_metrics_dir = REPO_ROOT / "outputs" / "metrics" / exp_id
    ablation_csv, selection_json = write_ablation_outputs(
        exp_metrics_dir=exp_metrics_dir,
        all_results=all_results,
        stage_best_map=stage_best_map,
        final_best=final_best,
    )

    figures_dir = REPO_ROOT / "outputs" / "figures" / exp_id
    failure_csv, failure_explained_csv = build_failure_case_index(
        pred_root=exp_pred_root,
        gt_root=gt_root,
        datasets=final_datasets,
        out_dir=figures_dir / "failure_cases",
        backend_meta=final_best.backend_meta,
        propainter_meta=final_best.propainter_meta,
        top_k=3,
    )

    a_vs_b_path, a_vs_b_meta = write_a_vs_b_comparison(
        exp_metrics_dir=exp_metrics_dir,
        phase1_exp_id=phase1_exp_id,
        phase2_summary=final_summary,
    )

    report_path = exp_metrics_dir / "phase2_acceptance_report.md"
    write_acceptance_report(
        report_path=report_path,
        exp_id=exp_id,
        aggregate=final_summary.get("aggregate", {}),
        stage_best_map=stage_best_map,
        final_best=final_best,
        per_dataset=final_summary.get("datasets", {}),
        phase1_exp_id=phase1_exp_id,
        a_vs_b_path=a_vs_b_path,
        failure_explained_csv=failure_explained_csv,
        seed=seed_value,
        strict_dual_run=strict_dual_run,
    )

    write_json(
        exp_metrics_dir / "phase2_run_meta.json",
        {
            "exp_id": exp_id,
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "config": str(config_path),
            "datasets": final_datasets,
            "seed": seed_value,
            "seed_meta": seed_meta,
            "device": device,
            "auto_install_missing": auto_install_missing,
            "strict_dual_run": strict_dual_run,
            "mask_backends": mask_backends,
            "prompt_detector": prompt_detector,
            "prompt_meta": prompt_meta,
            "backend_environment": backend_env,
            "propainter_environment": propainter_meta,
            "phase1_exp_id": phase1_exp_id,
            "phase1_compare_meta": a_vs_b_meta,
            "ablation_csv": str(ablation_csv),
            "selection_json": str(selection_json),
            "summary_json": str(final_summary_path),
            "failure_csv": str(failure_csv),
            "failure_explained_csv": str(failure_explained_csv),
            "a_vs_b_csv": str(a_vs_b_path) if a_vs_b_path else None,
            "acceptance_report": str(report_path),
            "log_path": str(log_path),
            "final_best": asdict(final_best.spec),
        },
    )

    logger.info("B-best summary: %s", final_summary_path)
    logger.info("Phase2 ablation csv: %s", ablation_csv)
    logger.info("Phase2 selection json: %s", selection_json)
    logger.info("Failure case csv: %s", failure_csv)
    logger.info("Failure case explained csv: %s", failure_explained_csv)
    if a_vs_b_path is not None:
        logger.info("A vs B compare csv: %s", a_vs_b_path)
    logger.info("Acceptance report: %s", report_path)


if __name__ == "__main__":
    main()
