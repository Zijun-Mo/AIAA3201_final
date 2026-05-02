#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.part1.run_baseline import resolve_device, set_global_seed, str2bool, write_dataset_outputs
from src.part2.run_sota import (
    DatasetPayload,
    build_prior_prompt_anchors_from_masks,
    build_failure_case_index,
    collect_dataset_cfg,
    ensure_propainter_ready,
    generate_backend_masks_with_priority,
    load_dataset_payload,
    metric_or_neg_inf,
    normalize_mask_propagation_cfg,
    probe_backend_environment,
    read_json,
    read_yaml,
    resolve_dataset_names,
    resolve_repo_path,
    run_evaluation,
    run_propainter_with_fallback,
    sanitize_name,
    write_json,
)
from src.common.remove_quality import DynamicObjectDetector
from src.common.video_io import (
    cleanup_named_subdirs,
    cleanup_video_only_outputs,
    dataset_video_paths,
    load_masks_by_names_with_video_fallback,
    resolve_output_policy,
)
from src.part3.motion_flow import (
    apply_trajectory_consistency,
    compute_video_flow_consistency,
    compute_video_instance_motion_scores,
    compute_video_motion_maps,
    mask_flow_reliability,
)
from src.part3.mask_fusion import SUPPORTED_FUSION_METHODS, apply_fusion
from src.part3.vggt_prior import generate_vggt4d_dynamic_priors

SUPPORTED_STAGES = ["E1", "E2", "E3", "E4", "F1", "F2", "F3", "F4", "F5"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass
class CandidateSpec:
    stage: str
    name: str
    source_stage: str
    e1_profile: dict[str, Any]
    temporal_window: int
    use_sam3: bool
    f_source_key: str = ""
    f_fusion_method: str = ""
    f_fusion_cfg: dict[str, Any] = field(default_factory=dict)
    f_trajectory_cfg: dict[str, Any] = field(default_factory=dict)
    f_use_bidirectional: bool = False

    @property
    def candidate_id(self) -> str:
        return sanitize_name(f"{self.stage}_{self.name}")


@dataclass
class CandidateResult:
    spec: CandidateSpec
    candidate_root: Path
    eval_exp_id: str
    summary_path: Path
    aggregate: dict[str, Any]
    per_dataset: dict[str, Any]
    mask_stats: dict[str, dict[str, Any]]
    stage_mask_meta: dict[str, dict[str, Any]]
    propainter_meta: dict[str, dict[str, Any]]


class CandidateExecutionError(RuntimeError):
    pass


def setup_logger(exp_id: str) -> tuple[logging.Logger, Path]:
    log_dir = REPO_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"phase3_{exp_id}.log"

    logger = logging.getLogger("phase3")
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


def list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def parse_stages(spec: str) -> list[str]:
    if not spec:
        return list(SUPPORTED_STAGES)
    seen: set[str] = set()
    for token in [x.strip().upper() for x in spec.split(",") if x.strip()]:
        if token not in SUPPORTED_STAGES:
            raise ValueError(f"Unsupported stage '{token}'. Valid: {SUPPORTED_STAGES}")
        seen.add(token)
    return [s for s in SUPPORTED_STAGES if s in seen]


def ensure_binary_mask(mask: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    h, w = frame_shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((mask > 0).astype(np.uint8) * 255)


def compute_mean_mask_ratio(masks_u8: list[np.ndarray]) -> float:
    if not masks_u8:
        return 0.0
    vals = [float((m > 0).mean()) for m in masks_u8]
    return float(np.mean(np.array(vals, dtype=np.float32)))


def compute_active_frame_ratio(masks_u8: list[np.ndarray]) -> float:
    if not masks_u8:
        return 0.0
    active = [1.0 if int((np.asarray(m) > 0).sum()) > 0 else 0.0 for m in masks_u8]
    return float(np.mean(np.array(active, dtype=np.float32)))


def mask_bbox(mask_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask_u8) > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def build_prior_prompt_from_masks(
    masks_u8: list[np.ndarray],
    frame_shape: tuple[int, int],
    max_prompts: int,
    min_area_ratio: float = 1e-6,
) -> tuple[int, list[tuple[int, int, int, int]], list[list[dict[str, Any]]], dict[str, Any]]:
    """Compatibility wrapper around the shared Phase2 prior prompt builder."""
    h, w = frame_shape
    binary_masks = [ensure_binary_mask(np.asarray(m), (h, w)) for m in masks_u8]
    instances_per_frame: list[list[dict[str, Any]]] = []
    for mask in binary_masks:
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        frame_instances: list[dict[str, Any]] = []
        for label in range(1, int(num_labels)):
            comp_mask = ((labels == label).astype(np.uint8) * 255)
            frame_instances.append({"mask": comp_mask, "score": 1.0, "class_name": "vggt4d_prior"})
        instances_per_frame.append(frame_instances)

    anchors, anchor_meta = build_prior_prompt_anchors_from_masks(
        masks_u8=binary_masks,
        frame_shape=frame_shape,
        max_anchors=1,
        max_prompts_per_anchor=max_prompts,
        min_area_ratio=min_area_ratio,
        min_anchor_gap_ratio=0.0,
        source_name="vggt4d",
    )
    if not anchors:
        meta = {
            "prompt_source": "empty",
            "prompt_frame_idx": 0,
            "prompt_boxes": [],
            "prompt_box_count": 0,
            "max_area": 0.0,
            "min_area_ratio": float(min_area_ratio),
            "max_prompts": int(max_prompts),
            "anchor_meta": anchor_meta,
        }
        return 0, [], instances_per_frame, meta

    first = anchors[0]
    boxes = [tuple(int(v) for v in box) for box in first.get("boxes", [])]
    meta = {
        "prompt_source": first.get("source", "vggt4d_connected_components"),
        "prompt_frame_idx": int(first["frame_idx"]),
        "prompt_boxes": [list(map(int, box)) for box in boxes],
        "prompt_box_count": int(len(boxes)),
        "max_area": float(first.get("total_area", 0.0)),
        "min_area_ratio": float(min_area_ratio),
        "max_prompts": int(max_prompts),
        "anchor_meta": anchor_meta,
    }
    return int(first["frame_idx"]), boxes, instances_per_frame, meta


def run_bbest_backend_postprocess_on_prior(
    *,
    dataset_name: str,
    payload: DatasetPayload,
    prior_masks_u8: list[np.ndarray],
    backend: str,
    part2_cfg: dict[str, Any],
    backend_env: dict[str, dict[str, Any]],
    device: str,
    max_prompts: int,
    min_area_ratio: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    backend_key = backend.strip().lower()
    prop_cfg = normalize_mask_propagation_cfg(part2_cfg)
    anchors, anchor_meta = build_prior_prompt_anchors_from_masks(
        masks_u8=prior_masks_u8,
        frame_shape=payload.frames[0].shape[:2],
        max_anchors=int(prop_cfg.get("max_anchors", 3)),
        max_prompts_per_anchor=max_prompts,
        min_area_ratio=min_area_ratio,
        min_anchor_gap_ratio=float(prop_cfg.get("min_anchor_gap_ratio", 0.16)),
        fallback_frames=payload.frames,
        source_name="vggt4d",
    )
    meta: dict[str, Any] = {
        "bbest_backend": backend_key,
        "prompt_meta": anchor_meta,
        "anchors": anchors,
        "official_used": False,
    }
    if not anchors:
        raise CandidateExecutionError(f"VGGT4D prior could not produce prompt boxes for dataset={dataset_name}")

    if backend_key not in {"sam2", "trackanything"}:
        raise CandidateExecutionError(f"Unsupported B-best mask backend for Route F: {backend}")

    prompt_frame_idx = int(anchors[0].get("frame_idx", 0))
    prompt_boxes = [tuple(int(v) for v in box) for box in anchors[0].get("boxes", [])]
    instances = [[{"mask": m, "score": 1.0, "class_name": "vggt4d_prior"}] for m in prior_masks_u8]
    masks, official_meta = generate_backend_masks_with_priority(
        backend=backend_key,
        payload=payload,
        detector_instances=instances,
        prompt_frame_idx=prompt_frame_idx,
        prompt_boxes=prompt_boxes,
        prompt_anchors=anchors,
        part2_cfg=part2_cfg,
        backend_env=backend_env,
        device=device,
        logger=logging.getLogger("phase3"),
    )
    meta["backend_meta"] = official_meta
    if masks is None or not bool(official_meta.get("official_used", False)):
        reason = official_meta.get("official_error", "unknown")
        raise CandidateExecutionError(
            f"B-best backend postprocess failed for dataset={dataset_name}, backend={backend_key}, reason={reason}"
        )

    out = align_masks_to_frame_count(
        masks_u8=masks,
        frame_count=len(payload.frame_names),
        frame_shape=payload.frames[0].shape[:2],
    )
    meta["official_used"] = True
    meta["mean_mask_ratio"] = compute_mean_mask_ratio(out)
    meta["active_frame_ratio"] = compute_active_frame_ratio(out)
    return out, meta


def normalize_odd_kernel(value: int) -> int:
    k = max(1, int(value))
    if k % 2 == 0:
        k += 1
    return k


def apply_morph_profile(
    masks_u8: list[np.ndarray],
    opening_kernel: int,
    closing_kernel: int,
    dilation_kernel: int,
    smoothing_kernel: int,
) -> list[np.ndarray]:
    if not masks_u8:
        return []

    ok = normalize_odd_kernel(opening_kernel)
    ck = normalize_odd_kernel(closing_kernel)
    dk = normalize_odd_kernel(dilation_kernel)
    sk = max(0, int(smoothing_kernel))

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))

    out: list[np.ndarray] = []
    for m in masks_u8:
        b = ((np.asarray(m) > 0).astype(np.uint8) * 255)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, open_kernel, iterations=1)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        b = cv2.dilate(b, dilate_kernel, iterations=1)

        if sk > 1:
            blur_k = normalize_odd_kernel(sk)
            x = cv2.GaussianBlur(b.astype(np.float32), (blur_k, blur_k), 0)
            b = ((x >= 127.5).astype(np.uint8) * 255)

        out.append(((b > 0).astype(np.uint8) * 255))
    return out


def temporal_smooth_masks(masks_u8: list[np.ndarray], temporal_window: int) -> list[np.ndarray]:
    if not masks_u8:
        return []
    tw = max(0, int(temporal_window))
    if tw <= 0:
        return [((np.asarray(m) > 0).astype(np.uint8) * 255) for m in masks_u8]

    stack = np.stack([(np.asarray(m) > 0).astype(np.float32) for m in masks_u8], axis=0)
    n = stack.shape[0]
    out: list[np.ndarray] = []
    for idx in range(n):
        lo = max(0, idx - tw)
        hi = min(n, idx + tw + 1)
        avg = stack[lo:hi].mean(axis=0)
        out.append((avg >= 0.5).astype(np.uint8) * 255)
    return out


def load_masks_by_frame_names(
    mask_dir: Path,
    frame_names: list[str],
    frame_shape: tuple[int, int],
    mask_video_path: Path | None = None,
    threshold: int = 127,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    masks, meta = load_masks_by_names_with_video_fallback(
        mask_dir=mask_dir,
        frame_names=frame_names,
        frame_shape=frame_shape,
        mask_video_path=mask_video_path,
        threshold=threshold,
    )
    if masks is None:
        raise FileNotFoundError(
            f"Masks unavailable from dir/video: dir={mask_dir}, video={mask_video_path}"
        )
    meta["mean_mask_ratio"] = compute_mean_mask_ratio(masks)
    meta["active_frame_ratio"] = compute_active_frame_ratio(masks)
    return masks, meta


def align_masks_to_frame_count(
    masks_u8: list[np.ndarray],
    frame_count: int,
    frame_shape: tuple[int, int],
) -> list[np.ndarray]:
    if frame_count <= 0:
        return []
    h, w = frame_shape
    out = [ensure_binary_mask(np.asarray(m), (h, w)) for m in masks_u8]
    if not out:
        return [np.zeros((h, w), dtype=np.uint8) for _ in range(frame_count)]
    if len(out) < frame_count:
        out.extend([out[-1].copy() for _ in range(frame_count - len(out))])
    elif len(out) > frame_count:
        out = out[:frame_count]
    return out


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
            logger.warning("stderr tail: %s", completed.stderr[-1000:])
        if completed.stdout:
            logger.warning("stdout tail: %s", completed.stdout[-1000:])
    return completed


def build_hf_env(sam3_cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    hf_endpoint = str(sam3_cfg.get("hf_endpoint", "https://huggingface.co")).strip()
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    return env


def ensure_conda_env(
    env_name: str,
    auto_install: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "env_name": env_name,
        "exists": False,
        "created": False,
    }

    probe = run_cmd(["conda", "run", "-n", env_name, "python", "-V"], cwd=REPO_ROOT, logger=logger)
    if probe.returncode == 0:
        meta["exists"] = True
        return meta

    if not auto_install:
        meta["reason"] = "env_missing_auto_install_disabled"
        return meta

    create = run_cmd(["conda", "create", "-y", "-n", env_name, "python=3.10"], cwd=REPO_ROOT, logger=logger)
    if create.returncode != 0:
        meta["reason"] = "env_create_failed"
        return meta

    meta["created"] = True
    probe2 = run_cmd(["conda", "run", "-n", env_name, "python", "-V"], cwd=REPO_ROOT, logger=logger)
    meta["exists"] = probe2.returncode == 0
    if not meta["exists"]:
        meta["reason"] = "env_probe_failed_after_create"
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
    completed = run_cmd(["git", "clone", "--depth", "1", repo_url, str(target)], cwd=REPO_ROOT, logger=logger)
    if completed.returncode != 0:
        return None, "clone_failed"
    return target, "cloned"


def ensure_sam3_repo(
    part3_cfg: dict[str, Any],
    external_root: Path,
    sam3_env_name: str,
    auto_install_missing: bool,
    logger: logging.Logger,
) -> tuple[Path | None, dict[str, Any]]:
    e3_cfg = ((part3_cfg.get("e3", {}) or {}).get("sam3", {}) or {})
    repo_url = str(e3_cfg.get("repo_url", "https://github.com/facebookresearch/sam3.git"))
    repo_root = external_root / "repos"

    repo_path, repo_state = ensure_repo(
        repo_name="sam3",
        repo_url=repo_url,
        repo_root=repo_root,
        auto_install=auto_install_missing,
        logger=logger,
    )

    meta: dict[str, Any] = {
        "repo_url": repo_url,
        "repo_state": repo_state,
        "repo_path": str(repo_path) if repo_path else None,
    }

    env_meta = ensure_conda_env(env_name=sam3_env_name, auto_install=auto_install_missing, logger=logger)
    meta["env"] = env_meta
    if not env_meta.get("exists", False):
        meta["ready"] = False
        meta["reason"] = env_meta.get("reason", "sam3_env_unavailable")
        return None, meta

    if repo_path is None:
        meta["ready"] = False
        meta["reason"] = "sam3_repo_missing"
        return None, meta

    if auto_install_missing:
        hf_env = build_hf_env(e3_cfg)
        install_cmds = [
            ["conda", "run", "-n", sam3_env_name, "python", "-m", "pip", "install", "--upgrade", "pip"],
            [
                "conda",
                "run",
                "-n",
                sam3_env_name,
                "python",
                "-m",
                "pip",
                "install",
                "numpy",
                "opencv-python",
                "huggingface_hub",
                "pyyaml",
                "einops",
                "omegaconf",
                "hydra-core",
                "imageio",
                "imageio-ffmpeg",
                "pycocotools",
                "psutil",
            ],
            [
                "conda",
                "run",
                "-n",
                sam3_env_name,
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                str(repo_path),
            ],
        ]
        install_codes: list[int] = []
        for cmd in install_cmds:
            completed = run_cmd(
                cmd,
                cwd=repo_path,
                logger=logger,
                env=hf_env,
            )
            install_codes.append(int(completed.returncode))
            if completed.returncode != 0:
                break
        meta["install_returncodes"] = install_codes
        meta["pip_editable_install_returncode"] = int(install_codes[-1]) if install_codes else None
        meta["ready"] = bool(install_codes and install_codes[-1] == 0)
        if not meta["ready"]:
            meta["reason"] = "sam3_install_failed"
    else:
        verify = run_cmd(
            [
                "conda",
                "run",
                "-n",
                sam3_env_name,
                "python",
                "-c",
                "import importlib.util as u; print('ok' if u.find_spec('sam3') else 'missing')",
            ],
            cwd=repo_path,
            logger=logger,
        )
        meta["sam3_import_probe_returncode"] = int(verify.returncode)
        meta["pip_editable_install_returncode"] = None
        meta["ready"] = bool(verify.returncode == 0)
        if not meta["ready"]:
            meta["reason"] = "sam3_not_importable_in_env"
    return repo_path, meta


def decide_e3_flow(permission_passed: bool, runtime_error: str | None) -> str:
    if not permission_passed:
        return "abort"
    if runtime_error:
        return "skip"
    return "continue"


def parse_selection_excludes(arg_value: str | None, cfg_excludes: list[Any]) -> list[str]:
    if arg_value is not None and arg_value.strip():
        return [x.strip() for x in arg_value.split(",") if x.strip()]
    return [str(x).strip() for x in cfg_excludes if str(x).strip()]


def normalize_selection_primary_metric(value: Any, default: str = "auto") -> str:
    mode = str(value if value is not None else default).strip().lower()
    if mode not in {"mask", "quality", "auto"}:
        mode = str(default).strip().lower()
    if mode not in {"mask", "quality", "auto"}:
        mode = "auto"
    return mode


def dataset_metric_aggregate(
    per_dataset: dict[str, Any],
    datasets_for_scoring: list[str] | None,
) -> dict[str, float]:
    keys = ["JM", "JR", "ROS", "TCF", "BES"]
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


def stage_score(
    stage: str,
    agg: dict[str, float],
    mean_mask_ratio: float,
    primary_metric: str = "auto",
) -> tuple[float, float, float, float]:
    jm = float(agg.get("JM", float("-inf")))
    jr = float(agg.get("JR", float("-inf")))
    tcf = float(agg.get("TCF", float("inf")))
    if not np.isfinite(tcf):
        tcf = float("inf")
    mode = normalize_selection_primary_metric(primary_metric, default="auto")
    if mode == "mask":
        return (jm, jr, -tcf, -abs(mean_mask_ratio - 0.1))
    if mode == "quality":
        return (-tcf, jm, jr, -abs(mean_mask_ratio - 0.1))
    if stage in {"E1", "E2", "E3", "F1", "F2", "F3"}:
        return (jm, jr, -tcf, -abs(mean_mask_ratio - 0.1))
    return (-tcf, jm, jr, -abs(mean_mask_ratio - 0.1))


def parse_selection_coverage_constraints(selection_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    mean_cfg = selection_cfg.get("min_mean_mask_ratio_by_dataset", {}) or {}
    active_cfg = selection_cfg.get("min_active_frame_ratio_by_dataset", {}) or {}
    constraints: dict[str, dict[str, float]] = {}

    if not isinstance(mean_cfg, dict):
        mean_cfg = {}
    if not isinstance(active_cfg, dict):
        active_cfg = {}

    for ds in sorted(set([str(k).strip() for k in mean_cfg.keys()] + [str(k).strip() for k in active_cfg.keys()])):
        if not ds:
            continue
        req: dict[str, float] = {}
        if ds in mean_cfg:
            req["min_mean_mask_ratio"] = float(mean_cfg.get(ds, 0.0))
        if ds in active_cfg:
            req["min_active_frame_ratio"] = float(active_cfg.get(ds, 0.0))
        constraints[ds] = req
    return constraints


def candidate_meets_coverage_constraints(
    entry: CandidateResult,
    coverage_constraints: dict[str, dict[str, float]],
    score_datasets: list[str] | None = None,
) -> bool:
    if not coverage_constraints:
        return True
    for ds, req in coverage_constraints.items():
        if score_datasets and ds not in score_datasets:
            continue
        ds_stats = entry.mask_stats.get(ds, {}) or {}
        if not ds_stats:
            continue
        mean_ratio = float(ds_stats.get("mean_mask_ratio", 0.0))
        active_ratio = float(
            ds_stats.get(
                "active_frame_ratio",
                1.0 if mean_ratio > 0.0 else 0.0,
            )
        )
        if "min_mean_mask_ratio" in req and mean_ratio < float(req["min_mean_mask_ratio"]):
            return False
        if "min_active_frame_ratio" in req and active_ratio < float(req["min_active_frame_ratio"]):
            return False
    return True


def select_best(
    stage: str,
    entries: list[CandidateResult],
    score_datasets: list[str],
    coverage_constraints: dict[str, dict[str, float]] | None = None,
    enforce_if_candidate_available: bool = True,
    primary_metric: str = "auto",
    logger: logging.Logger | None = None,
) -> CandidateResult:
    if not entries:
        raise ValueError(f"No entries to select in stage {stage}")

    pool = entries
    constraints = coverage_constraints or {}
    if constraints:
        eligible = [
            e
            for e in entries
            if candidate_meets_coverage_constraints(
                e,
                coverage_constraints=constraints,
                score_datasets=score_datasets,
            )
        ]
        if eligible:
            pool = eligible
        elif enforce_if_candidate_available:
            if logger is not None:
                logger.warning(
                    "[%s] No candidate met coverage constraints=%s, fallback to all candidates.",
                    stage,
                    constraints,
                )
        else:
            raise ValueError(f"No candidate meets coverage constraints in stage {stage}: {constraints}")

    def score(entry: CandidateResult) -> tuple[float, float, float, float]:
        ratios = [
            (entry.mask_stats.get(ds, {}) or {}).get("mean_mask_ratio", 0.0) for ds in score_datasets
        ]
        mean_ratio = float(np.mean(np.array(ratios, dtype=np.float32))) if ratios else 0.0
        agg = dataset_metric_aggregate(entry.per_dataset, score_datasets)
        return stage_score(stage, agg, mean_ratio, primary_metric=primary_metric)

    return max(pool, key=score)


def select_f_route_best(
    entries: list[CandidateResult],
    score_datasets: list[str],
    coverage_constraints: dict[str, dict[str, float]] | None = None,
    enforce_if_candidate_available: bool = True,
    logger: logging.Logger | None = None,
) -> CandidateResult:
    """Select the final Route F output from successful F-stage winners.

    Phase 4 is meant to export the VGGT4D-prior result. YOLO/B-best and fusion
    candidates are ablations only, so final videos and final metrics must come
    from a pure VGGT4D-prior candidate, not from the best-scoring fusion row.
    """
    vggt4d_entries = [
        entry
        for entry in entries
        if entry.spec.f_source_key.strip().lower() == "vggt4d"
    ]
    if vggt4d_entries:
        forced = select_best(
            stage="F1",
            entries=vggt4d_entries,
            score_datasets=score_datasets,
            coverage_constraints=None,
            enforce_if_candidate_available=True,
            primary_metric="mask",
            logger=logger,
        )
        if logger is not None:
            logger.info(
                "[F-best] force VGGT4D-prior final output: candidate=%s stage=%s",
                forced.spec.name,
                forced.spec.stage,
            )
        return forced

    raise ValueError("Route F final output requires a successful VGGT4D-prior candidate (f_source_key=vggt4d)")


def resolve_phase2_reference(
    phase2_exp_id_arg: str | None,
    part3_cfg: dict[str, Any],
) -> tuple[str, Path, Path]:
    refs = part3_cfg.get("references", {}) or {}
    b_alias = str(refs.get("b_best_alias", "B-best"))

    if phase2_exp_id_arg and phase2_exp_id_arg.strip():
        token = phase2_exp_id_arg.strip()
    else:
        token = b_alias

    pred_root = REPO_ROOT / "outputs" / "videos" / token
    metrics_root = REPO_ROOT / "outputs" / "metrics" / token
    return token, pred_root, metrics_root


def resolve_phase1_reference(part3_cfg: dict[str, Any]) -> tuple[str, Path]:
    refs = part3_cfg.get("references", {}) or {}
    a_alias = str(refs.get("a_best_alias", "A-best"))
    metrics_root = REPO_ROOT / "outputs" / "metrics" / a_alias
    return a_alias, metrics_root


def check_sam3_permission(
    sam3_cfg: dict[str, Any],
    sam3_env_name: str,
    strict_permission: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    checkpoint_cfg = sam3_cfg.get("checkpoint", {}) or {}
    local_path = str(checkpoint_cfg.get("local_path", "")).strip()
    hf_repo_id = str(checkpoint_cfg.get("hf_repo_id", "")).strip()
    hf_filename = str(checkpoint_cfg.get("hf_filename", "")).strip()

    meta: dict[str, Any] = {
        "strict_permission": bool(strict_permission),
        "sam3_env_name": sam3_env_name,
        "local_checkpoint": local_path,
        "hf_repo_id": hf_repo_id,
        "hf_filename": hf_filename,
        "hf_endpoint": str(sam3_cfg.get("hf_endpoint", "https://huggingface.co")).strip(),
        "passed": False,
        "reason": "",
    }

    if local_path:
        lp = resolve_repo_path(Path(local_path))
        if lp.exists():
            meta["passed"] = True
            meta["reason"] = "local_checkpoint_exists"
            meta["resolved_local_checkpoint"] = str(lp)
            return meta

    if not hf_repo_id:
        meta["reason"] = "hf_repo_id_missing"
        if strict_permission:
            raise RuntimeError("SAM3 permission check failed: hf_repo_id missing and no local checkpoint")
        return meta

    test_filename = hf_filename or "config.json"
    meta["hf_test_filename"] = test_filename
    check_code = (
        "from huggingface_hub import HfApi, hf_hub_download\n"
        f"api=HfApi(); info=api.model_info('{hf_repo_id}')\n"
        f"path=hf_hub_download(repo_id='{hf_repo_id}', filename='{test_filename}')\n"
        "print('ok', info.id, path)\n"
    )

    cmd = ["conda", "run", "--no-capture-output", "-n", sam3_env_name, "python", "-c", check_code]
    completed = run_cmd(cmd=cmd, cwd=REPO_ROOT, logger=logger, env=build_hf_env(sam3_cfg))

    if completed.returncode == 0:
        meta["passed"] = True
        meta["reason"] = "hf_download_access_ok"
    else:
        meta["passed"] = False
        meta["reason"] = "hf_download_access_failed"
        meta["stderr_tail"] = (completed.stderr or "")[-1000:]

    if strict_permission and not meta["passed"]:
        raise RuntimeError(
            "SAM3 permission check failed (strict mode): unable to access checkpoint/model repo"
        )

    return meta


def run_sam3_refine_subprocess(
    *,
    repo_path: Path,
    sam3_env_name: str,
    dataset_name: str,
    frame_dir: Path,
    input_mask_dir: Path,
    output_mask_dir: Path,
    sam3_cfg: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    checkpoint_cfg = sam3_cfg.get("checkpoint", {}) or {}
    local_path = str(checkpoint_cfg.get("local_path", "")).strip()
    checkpoint_arg = ""
    if local_path:
        resolved = resolve_repo_path(Path(local_path))
        if resolved.exists():
            checkpoint_arg = str(resolved)
        else:
            logger.warning("SAM3 local checkpoint not found, fallback to backend default: %s", resolved)

    prompt_text_by_dataset = sam3_cfg.get("prompt_text_by_dataset", {}) or {}
    if isinstance(prompt_text_by_dataset, dict):
        prompt_text = str(prompt_text_by_dataset.get(dataset_name, sam3_cfg.get("prompt_text", "")))
    else:
        prompt_text = str(sam3_cfg.get("prompt_text", ""))
    device = str(sam3_cfg.get("device", "cuda"))
    model_cfg = str(sam3_cfg.get("model_cfg", "")).strip()

    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        sam3_env_name,
        "python",
        str(REPO_ROOT / "src" / "part3" / "sam3_refine_worker.py"),
        "--repo-path",
        str(repo_path),
        "--dataset-name",
        dataset_name,
        "--input-frames",
        str(frame_dir),
        "--input-masks",
        str(input_mask_dir),
        "--output-masks",
        str(output_mask_dir),
        "--prompt-text",
        prompt_text,
        "--device",
        device,
    ]
    if checkpoint_arg:
        cmd.extend(["--checkpoint", checkpoint_arg])
    if model_cfg:
        cmd.extend(["--model-cfg", str(model_cfg)])

    completed = run_cmd(cmd=cmd, cwd=REPO_ROOT, logger=logger, env=build_hf_env(sam3_cfg))
    return {
        "returncode": int(completed.returncode),
        "stdout_tail": (completed.stdout or "")[-1000:],
        "stderr_tail": (completed.stderr or "")[-1000:],
        "status": "ok" if completed.returncode == 0 else "failed",
    }


def copy_candidate_outputs(best_candidate_root: Path, final_root: Path, datasets: list[str]) -> None:
    for ds in datasets:
        src = best_candidate_root / ds
        if not src.exists():
            raise RuntimeError(f"Candidate output missing for dataset {ds}: {src}")
        dst = final_root / ds
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def write_ablation_outputs(
    metrics_dir: Path,
    all_results: list[CandidateResult],
    stage_best_map: dict[str, CandidateResult],
    final_best: CandidateResult,
    phase_label: str = "phase3",
) -> tuple[Path, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for r in all_results:
        ratios = [v.get("mean_mask_ratio", 0.0) for v in r.mask_stats.values()]
        mean_ratio = float(np.mean(np.array(ratios, dtype=np.float32))) if ratios else 0.0
        active_ratios = [
            v.get("active_frame_ratio", 1.0 if float(v.get("mean_mask_ratio", 0.0)) > 0 else 0.0)
            for v in r.mask_stats.values()
        ]
        active_ratio = float(np.mean(np.array(active_ratios, dtype=np.float32))) if active_ratios else 0.0

        row = {
            "stage": r.spec.stage,
            "candidate": r.spec.name,
            "source_stage": r.spec.source_stage,
            "use_sam3": int(r.spec.use_sam3),
            "opening_kernel": r.spec.e1_profile.get("opening_kernel", ""),
            "closing_kernel": r.spec.e1_profile.get("closing_kernel", ""),
            "dilation_kernel": r.spec.e1_profile.get("dilation_kernel", ""),
            "smoothing_kernel": r.spec.e1_profile.get("smoothing_kernel", ""),
            "temporal_window": r.spec.temporal_window,
            "f_source_key": r.spec.f_source_key,
            "f_fusion_method": r.spec.f_fusion_method,
            "JM": r.aggregate.get("JM"),
            "JR": r.aggregate.get("JR"),
            "ROS": r.aggregate.get("ROS"),
            "TCF": r.aggregate.get("TCF"),
            "BES": r.aggregate.get("BES"),
            "mean_mask_ratio": mean_ratio,
            "active_frame_ratio": active_ratio,
            "pred_root": str(r.candidate_root),
            "eval_exp_id": r.eval_exp_id,
            "is_stage_best": int(stage_best_map.get(r.spec.stage) is r),
            "is_final_best": int(final_best is r),
        }
        rows.append(row)

    csv_path = metrics_dir / f"{phase_label}_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "candidate",
                "source_stage",
                "use_sam3",
                "opening_kernel",
                "closing_kernel",
                "dilation_kernel",
                "smoothing_kernel",
                "temporal_window",
                "f_source_key",
                "f_fusion_method",
                "JM",
                "JR",
                "ROS",
                "TCF",
                "BES",
                "mean_mask_ratio",
                "active_frame_ratio",
                "pred_root",
                "eval_exp_id",
                "is_stage_best",
                "is_final_best",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    selection_path = metrics_dir / f"{phase_label}_selection.json"
    write_json(
        selection_path,
        {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "stage_best": {k: asdict(v.spec) for k, v in stage_best_map.items()},
            "final_best": asdict(final_best.spec),
        },
    )
    return csv_path, selection_path


def write_b_vs_e_comparison(
    metrics_dir: Path,
    phase2_summary: dict[str, Any],
    phase2_ref_token: str,
    phase3_summary: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    b_ds = phase2_summary.get("datasets", {}) or {}
    e_ds = phase3_summary.get("datasets", {}) or {}

    rows: list[dict[str, Any]] = []
    names = sorted(set(b_ds.keys()) | set(e_ds.keys()))
    for ds in names:
        bm = (b_ds.get(ds, {}) or {}).get("metrics", {}) or {}
        em = (e_ds.get(ds, {}) or {}).get("metrics", {}) or {}
        row = {
            "dataset": ds,
            "B_JM": bm.get("JM"),
            "E_JM": em.get("JM"),
            "delta_JM": None if bm.get("JM") is None or em.get("JM") is None else float(em.get("JM")) - float(bm.get("JM")),
            "B_JR": bm.get("JR"),
            "E_JR": em.get("JR"),
            "delta_JR": None if bm.get("JR") is None or em.get("JR") is None else float(em.get("JR")) - float(bm.get("JR")),
            "B_ROS": bm.get("ROS"),
            "E_ROS": em.get("ROS"),
            "delta_ROS": None if bm.get("ROS") is None or em.get("ROS") is None else float(em.get("ROS")) - float(bm.get("ROS")),
            "B_TCF": bm.get("TCF"),
            "E_TCF": em.get("TCF"),
            "delta_TCF": None if bm.get("TCF") is None or em.get("TCF") is None else float(em.get("TCF")) - float(bm.get("TCF")),
            "B_BES": bm.get("BES"),
            "E_BES": em.get("BES"),
            "delta_BES": None if bm.get("BES") is None or em.get("BES") is None else float(em.get("BES")) - float(bm.get("BES")),
        }
        rows.append(row)

    b_agg = phase2_summary.get("aggregate", {}) or {}
    e_agg = phase3_summary.get("aggregate", {}) or {}
    rows.append(
        {
            "dataset": "__aggregate__",
            "B_JM": b_agg.get("JM"),
            "E_JM": e_agg.get("JM"),
            "delta_JM": None if b_agg.get("JM") is None or e_agg.get("JM") is None else float(e_agg.get("JM")) - float(b_agg.get("JM")),
            "B_JR": b_agg.get("JR"),
            "E_JR": e_agg.get("JR"),
            "delta_JR": None if b_agg.get("JR") is None or e_agg.get("JR") is None else float(e_agg.get("JR")) - float(b_agg.get("JR")),
            "B_ROS": b_agg.get("ROS"),
            "E_ROS": e_agg.get("ROS"),
            "delta_ROS": None if b_agg.get("ROS") is None or e_agg.get("ROS") is None else float(e_agg.get("ROS")) - float(b_agg.get("ROS")),
            "B_TCF": b_agg.get("TCF"),
            "E_TCF": e_agg.get("TCF"),
            "delta_TCF": None if b_agg.get("TCF") is None or e_agg.get("TCF") is None else float(e_agg.get("TCF")) - float(b_agg.get("TCF")),
            "B_BES": b_agg.get("BES"),
            "E_BES": e_agg.get("BES"),
            "delta_BES": None if b_agg.get("BES") is None or e_agg.get("BES") is None else float(e_agg.get("BES")) - float(b_agg.get("BES")),
        }
    )

    out_path = metrics_dir / "phase3_b_vs_e.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "B_JM",
                "E_JM",
                "delta_JM",
                "B_JR",
                "E_JR",
                "delta_JR",
                "B_ROS",
                "E_ROS",
                "delta_ROS",
                "B_TCF",
                "E_TCF",
                "delta_TCF",
                "B_BES",
                "E_BES",
                "delta_BES",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "status": "ok",
        "phase2_reference": phase2_ref_token,
    }
    return out_path, meta


def write_b_vs_f_comparison(
    metrics_dir: Path,
    phase2_summary: dict[str, Any],
    phase2_ref_token: str,
    phase4_summary: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    b_ds = phase2_summary.get("datasets", {}) or {}
    f_ds = phase4_summary.get("datasets", {}) or {}

    rows: list[dict[str, Any]] = []
    names = sorted(set(b_ds.keys()) | set(f_ds.keys()))
    metric_keys = ["JM", "JR", "ROS", "TCF", "BES"]
    for ds in names:
        bm = (b_ds.get(ds, {}) or {}).get("metrics", {}) or {}
        fm = (f_ds.get(ds, {}) or {}).get("metrics", {}) or {}
        row: dict[str, Any] = {"dataset": ds}
        for key in metric_keys:
            row[f"B_{key}"] = bm.get(key)
            row[f"F_{key}"] = fm.get(key)
            if bm.get(key) is not None and fm.get(key) is not None:
                row[f"delta_{key}"] = float(fm[key]) - float(bm[key])
            else:
                row[f"delta_{key}"] = None
        rows.append(row)

    b_agg = phase2_summary.get("aggregate", {}) or {}
    f_agg = phase4_summary.get("aggregate", {}) or {}
    agg_row: dict[str, Any] = {"dataset": "__aggregate__"}
    for key in metric_keys:
        agg_row[f"B_{key}"] = b_agg.get(key)
        agg_row[f"F_{key}"] = f_agg.get(key)
        if b_agg.get(key) is not None and f_agg.get(key) is not None:
            agg_row[f"delta_{key}"] = float(f_agg[key]) - float(b_agg[key])
        else:
            agg_row[f"delta_{key}"] = None
    rows.append(agg_row)

    fieldnames = ["dataset"]
    for key in metric_keys:
        fieldnames.extend([f"B_{key}", f"F_{key}", f"delta_{key}"])

    out_path = metrics_dir / "phase4_b_vs_f.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path, {"status": "ok", "phase2_reference": phase2_ref_token}


def write_phase4_mask_priors(
    metrics_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    out_path = metrics_dir / "phase4_mask_priors.csv"
    fields = [
        "dataset",
        "prior",
        "mean_mask_ratio",
        "active_frame_ratio",
        "frame_count",
        "source",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    return out_path


def write_acceptance_report(
    report_path: Path,
    exp_id: str,
    phase_label: str,
    aggregate: dict[str, Any],
    stage_best_map: dict[str, CandidateResult],
    final_best: CandidateResult,
    per_dataset: dict[str, Any],
    phase2_ref_token: str,
    compare_csv: Path,
    failure_explained_csv: Path,
    seed: int,
    selection_datasets: list[str],
    e3_permission_meta: dict[str, Any],
    e3_skipped_reason: str | None,
    priors_csv: Path | None = None,
) -> None:
    lines: list[str] = []
    lines.append(f"# {phase_label.upper()} Acceptance Report: `{exp_id}`")
    lines.append("")
    lines.append("## Final Aggregate")
    lines.append("")
    lines.append("| JM | JR | ROS | TCF | BES |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| {aggregate.get('JM')} | {aggregate.get('JR')} | {aggregate.get('ROS')} | {aggregate.get('TCF')} | {aggregate.get('BES')} |"
    )
    lines.append("")

    lines.append("## Stage Best")
    lines.append("")
    lines.append("| Stage | Candidate | Source | use_sam3 | temporal_window |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for stage in SUPPORTED_STAGES:
        r = stage_best_map.get(stage)
        if r is None:
            continue
        s = r.spec
        lines.append(
            f"| {stage} | {s.name} | {s.source_stage} | {int(s.use_sam3)} | {s.temporal_window} |"
        )
    lines.append("")

    lines.append("## Final Best")
    lines.append("")
    s = final_best.spec
    lines.append(
        f"- `{s.name}`: source={s.source_stage}, use_sam3={s.use_sam3}, profile={s.e1_profile}, temporal_window={s.temporal_window}"
    )
    if s.stage.startswith("F"):
        if s.f_source_key.strip().lower() == "vggt4d":
            lines.append(
                f"- Route decision: final output forced to pure VGGT4D prior result -> `{s.stage}/{s.name}`"
            )
        else:
            lines.append(f"- Route decision: invalid F-final candidate without VGGT4D prior key -> `{s.stage}/{s.name}`")
    lines.append("")

    lines.append("## Per-Dataset Metrics")
    lines.append("")
    lines.append("| Dataset | JM | JR | ROS | TCF | BES |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for ds_name, ds_payload in per_dataset.items():
        metrics = ds_payload.get("metrics", {}) if isinstance(ds_payload, dict) else {}
        lines.append(
            f"| {ds_name} | {metrics.get('JM')} | {metrics.get('JR')} | {metrics.get('ROS')} | {metrics.get('TCF')} | {metrics.get('BES')} |"
        )
    lines.append("")

    lines.append("## B-best Comparison")
    lines.append("")
    lines.append(f"- Phase2 reference: `{phase2_ref_token}`")
    lines.append(f"- Comparison CSV: `{compare_csv}`")
    if priors_csv is not None:
        lines.append(f"- Mask priors CSV: `{priors_csv}`")
    lines.append("")

    lines.append("## Runtime Notes")
    lines.append("")
    lines.append(f"- Seed: `{seed}`")
    lines.append(f"- Selection datasets used for stage scoring: `{selection_datasets}`")
    lines.append(f"- SAM3 permission: `{e3_permission_meta}`")
    if e3_skipped_reason:
        lines.append(f"- E3 skipped reason: `{e3_skipped_reason}`")
    lines.append(f"- Failure explanations: `{failure_explained_csv}`")
    lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part3 exploration pipeline (E/F routes).")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--datasets", type=str, default="mandatory", help="mandatory | all | csv")
    parser.add_argument("--exp-id", type=str, default=None)
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--stages", type=str, default="E1,E2,E3,E4")

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto-install-missing", type=str, default=None)
    parser.add_argument("--phase2-exp-id", type=str, default=None)
    parser.add_argument("--selection-exclude-datasets", type=str, default=None)
    parser.add_argument("--sam3-env-name", type=str, default=None)
    parser.add_argument("--strict-sam3-permission", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--phase", type=str, default=None, help="phase3 | phase4 | auto (detect from stages)")
    parser.add_argument(
        "--phase4-bbest-backend-override",
        type=str,
        default=None,
        help="Experimental Route F backend override: sam2|trackanything. Default follows Phase2 B-best.",
    )
    return parser.parse_args()


def run_candidate(
    *,
    spec: CandidateSpec,
    datasets: list[str],
    dataset_payloads: dict[str, DatasetPayload],
    source_masks_map: dict[str, list[np.ndarray]],
    out_root: Path,
    part2_cfg: dict[str, Any],
    propainter_repo: Path | None,
    target_fps: float,
    sam3_repo: Path | None,
    sam3_env_name: str,
    sam3_cfg: dict[str, Any],
    output_policy: dict[str, Any],
    logger: logging.Logger,
    motion_maps: dict[str, list[np.ndarray]] | None = None,
    motion_scores: dict[str, list[float]] | None = None,
    flow_consistency_maps: dict[str, list[np.ndarray]] | None = None,
    external_masks_by_dataset: dict[str, list[np.ndarray]] | None = None,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[np.ndarray]]]:
    candidate_root = out_root / spec.stage / spec.candidate_id
    candidate_root.mkdir(parents=True, exist_ok=True)
    e3_skip_datasets = {
        str(x).strip().lower()
        for x in ((sam3_cfg.get("skip_datasets", []) or []) if isinstance(sam3_cfg, dict) else [])
        if str(x).strip()
    }
    e3_merge_mode = str((sam3_cfg.get("merge_with_input", "replace") if isinstance(sam3_cfg, dict) else "replace")).strip().lower()
    if e3_merge_mode in {"union", "max"}:
        e3_merge_mode = "or"
    elif e3_merge_mode in {"intersect", "intersection"}:
        e3_merge_mode = "and"
    elif e3_merge_mode in {"sam3", "sam3_only"}:
        e3_merge_mode = "replace"
    if e3_merge_mode not in {"replace", "or", "and"}:
        logger.warning("Unknown part3.e3.sam3.merge_with_input=%s, fallback to replace", e3_merge_mode)
        e3_merge_mode = "replace"

    mask_stats: dict[str, dict[str, Any]] = {}
    stage_mask_meta: dict[str, dict[str, Any]] = {}
    propainter_meta: dict[str, dict[str, Any]] = {}
    produced_masks: dict[str, list[np.ndarray]] = {}
    cleanup_enabled = bool(resolve_output_policy(output_policy).get("auto_cleanup_intermediates", True))

    for ds in datasets:
        payload = dataset_payloads[ds]
        base_masks = source_masks_map[ds]
        h, w = payload.frames[0].shape[:2]
        masks_u8 = [ensure_binary_mask(np.asarray(m), (h, w)) for m in base_masks]

        ds_meta: dict[str, Any] = {
            "source_stage": spec.source_stage,
            "use_sam3": spec.use_sam3,
        }

        if spec.stage == "E1":
            masks_u8 = apply_morph_profile(
                masks_u8=masks_u8,
                opening_kernel=int(spec.e1_profile.get("opening_kernel", 3)),
                closing_kernel=int(spec.e1_profile.get("closing_kernel", 5)),
                dilation_kernel=int(spec.e1_profile.get("dilation_kernel", 3)),
                smoothing_kernel=int(spec.e1_profile.get("smoothing_kernel", 0)),
            )
            ds_meta["e1_profile"] = dict(spec.e1_profile)

        elif spec.stage == "E2":
            masks_u8 = temporal_smooth_masks(masks_u8=masks_u8, temporal_window=spec.temporal_window)
            ds_meta["temporal_window"] = int(spec.temporal_window)

        elif spec.stage == "E3":
            if sam3_repo is None:
                raise CandidateExecutionError("SAM3 repo unavailable for E3 stage")
            if ds.lower() in e3_skip_datasets:
                ds_meta["sam3_skipped"] = True
                ds_meta["sam3_skip_reason"] = "dataset_in_skip_list"
            else:
                input_masks_for_merge = [m.copy() for m in masks_u8]
                in_mask_dir = candidate_root / "_sam3_input_masks" / ds
                out_mask_dir = candidate_root / "_sam3_output_masks" / ds
                in_mask_dir.mkdir(parents=True, exist_ok=True)
                out_mask_dir.mkdir(parents=True, exist_ok=True)
                for name, m in zip(payload.frame_names, masks_u8):
                    cv2.imwrite(str(in_mask_dir / name), m)

                sam3_meta = run_sam3_refine_subprocess(
                    repo_path=sam3_repo,
                    sam3_env_name=sam3_env_name,
                    dataset_name=ds,
                    frame_dir=payload.frame_dir,
                    input_mask_dir=in_mask_dir,
                    output_mask_dir=out_mask_dir,
                    sam3_cfg=sam3_cfg,
                    logger=logger,
                )
                ds_meta["sam3_subprocess"] = sam3_meta
                if sam3_meta.get("status") != "ok":
                    raise CandidateExecutionError(
                        f"E3 SAM3 refinement failed for dataset={ds}: {sam3_meta.get('stderr_tail', '')[-200:]}"
                    )

                masks_u8, load_meta = load_masks_by_frame_names(
                    mask_dir=out_mask_dir,
                    frame_names=payload.frame_names,
                    frame_shape=(h, w),
                )
                ds_meta["sam3_output_load_meta"] = load_meta
                ds_meta["sam3_merge_mode"] = e3_merge_mode
                if e3_merge_mode == "or":
                    masks_u8 = [
                        (((np.asarray(src) > 0) | (np.asarray(sam3m) > 0)).astype(np.uint8) * 255)
                        for src, sam3m in zip(input_masks_for_merge, masks_u8)
                    ]
                elif e3_merge_mode == "and":
                    masks_u8 = [
                        (((np.asarray(src) > 0) & (np.asarray(sam3m) > 0)).astype(np.uint8) * 255)
                        for src, sam3m in zip(input_masks_for_merge, masks_u8)
                    ]
                if cleanup_enabled:
                    removed = cleanup_named_subdirs(
                        candidate_root,
                        ["_sam3_input_masks", "_sam3_output_masks"],
                    )
                    if removed:
                        ds_meta["sam3_cleanup_removed_paths"] = removed

        elif spec.stage == "E4":
            # E4 uses anchor masks directly and focuses on final rendering/evaluation.
            pass

        elif spec.stage == "F1":
            # F1: final Route F path with postprocessed VGGT4D prior.
            ds_meta["f1_prior_key"] = spec.f_source_key or "vggt4d"

        elif spec.stage == "F2":
            # F2: unchanged B-best baseline.
            ds_meta["f2_prior_key"] = spec.f_source_key

        elif spec.stage == "F3":
            # F3: prior-fusion variants that inject VGGT4D cues on top of B-best baseline.
            ds_mm = (motion_maps or {}).get(ds)
            ds_ms = (motion_scores or {}).get(ds)
            if ds_mm is None or ds_ms is None:
                raise CandidateExecutionError(f"F3 requires motion priors for dataset={ds}")
            method = spec.f_fusion_method.strip().lower() or "weighted"
            if method not in SUPPORTED_FUSION_METHODS:
                raise CandidateExecutionError(f"F3 unsupported fusion method={method}")
            ext_masks = (external_masks_by_dataset or {}).get(ds)
            if method == "vggt4d_guided" and ext_masks is None:
                raise CandidateExecutionError(f"F3 method={method} requires external VGGT4D masks for dataset={ds}")
            fused, fusion_meta = apply_fusion(
                semantic_masks=masks_u8,
                motion_maps=ds_mm,
                motion_scores=ds_ms,
                method=method,
                fusion_cfg=spec.f_fusion_cfg,
                external_masks=ext_masks,
            )
            masks_u8 = fused
            ds_meta["fusion"] = fusion_meta

        elif spec.stage == "F4":
            ds_mm = (motion_maps or {}).get(ds)
            ds_ms = (motion_scores or {}).get(ds)
            ds_fc = (flow_consistency_maps or {}).get(ds)
            if ds_mm is None or ds_ms is None:
                raise CandidateExecutionError(f"F4 requires motion priors for dataset={ds}")

            refined_mm = ds_mm
            if spec.f_use_bidirectional and ds_fc is not None:
                max_err = float(spec.f_fusion_cfg.get("max_consistency_error", 3.0))
                refined_mm = [mask_flow_reliability(mm, fc, max_err) for mm, fc in zip(ds_mm, ds_fc)]
                ds_meta["bidirectional_applied"] = True
                ds_meta["max_consistency_error"] = max_err
            else:
                ds_meta["bidirectional_applied"] = False

            method = spec.f_fusion_method.strip().lower() or "weighted"
            ext_masks = (external_masks_by_dataset or {}).get(ds)
            if method == "vggt4d_guided" and ext_masks is None:
                method = "weighted"

            fused, fusion_meta = apply_fusion(
                semantic_masks=masks_u8,
                motion_maps=refined_mm,
                motion_scores=ds_ms,
                method=method,
                fusion_cfg=spec.f_fusion_cfg,
                external_masks=ext_masks,
            )
            masks_u8 = fused
            traj_masks, traj_meta = apply_trajectory_consistency(
                masks_u8=masks_u8,
                motion_scores=ds_ms,
                trajectory_cfg=spec.f_trajectory_cfg,
            )
            masks_u8 = traj_masks
            ds_meta["fusion"] = fusion_meta
            ds_meta["trajectory"] = traj_meta

        elif spec.stage == "F5":
            # F5 finalization / case-study pass.
            pass
        else:
            raise ValueError(f"Unsupported stage in run_candidate: {spec.stage}")

        masks_u8 = align_masks_to_frame_count(masks_u8, len(payload.frame_names), (h, w))
        ratio = compute_mean_mask_ratio(masks_u8)
        active_ratio = compute_active_frame_ratio(masks_u8)
        mask_stats[ds] = {
            "mean_mask_ratio": ratio,
            "active_frame_ratio": active_ratio,
            "frame_count": int(len(masks_u8)),
        }
        stage_mask_meta[ds] = ds_meta

        restored_frames, p_meta = run_propainter_with_fallback(
            payload=payload,
            dataset_name=ds,
            masks_u8=masks_u8,
            candidate_root=candidate_root,
            propainter_repo=propainter_repo,
            part2_cfg=part2_cfg,
            profile_override={"name": spec.name},
            target_fps=target_fps,
            output_policy=output_policy,
            logger=logger,
        )

        if not restored_frames:
            raise CandidateExecutionError(f"No restored frames for dataset={ds}, stage={spec.stage}")

        n = min(len(payload.frame_names), len(masks_u8), len(restored_frames))
        if n <= 0:
            raise CandidateExecutionError(f"Invalid lengths for dataset={ds}, stage={spec.stage}")

        write_dataset_outputs(
            out_root=candidate_root,
            dataset_name=ds,
            frame_names=payload.frame_names[:n],
            restored_frames=restored_frames[:n],
            masks_u8=masks_u8[:n],
            target_fps=target_fps,
            save_mp4=bool((part2_cfg or {}).get("save_mp4", False)),
            output_policy=output_policy,
        )

        propainter_meta[ds] = p_meta
        produced_masks[ds] = [m.copy() for m in masks_u8[:n]]

    write_json(
        candidate_root / "candidate_config.json",
        {
            "candidate": asdict(spec),
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "mask_stats": mask_stats,
            "stage_mask_meta": stage_mask_meta,
            "propainter_meta": propainter_meta,
        },
    )

    logger.info(
        "Candidate complete: stage=%s name=%s root=%s",
        spec.stage,
        spec.name,
        candidate_root,
    )

    return candidate_root, mask_stats, stage_mask_meta, propainter_meta, produced_masks


def main() -> None:
    args = parse_args()
    requested_stages = [s.strip().upper() for s in (args.stages or "").split(",") if s.strip()]
    has_f_requested = any(s.startswith("F") for s in requested_stages)
    default_phase_prefix = "phase4" if has_f_requested else "phase3"
    exp_id = args.exp_id or f"{default_phase_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger, log_path = setup_logger(exp_id)
    logger.info("%s start | exp_id=%s", default_phase_prefix.upper(), exp_id)

    config_path = resolve_repo_path(Path(args.config))
    config = read_yaml(config_path)
    part2_cfg = config.get("part2", {}) or {}
    part3_cfg = config.get("part3", {}) or {}
    output_policy = resolve_output_policy(config)

    ds_map, all_names, mandatory_names = collect_dataset_cfg(config)
    selected_datasets = resolve_dataset_names(args.datasets, all_names, mandatory_names)
    logger.info("Datasets: %s", selected_datasets)

    stages = parse_stages(args.stages)
    logger.info("Stages to run: %s", stages)

    runtime_cfg = part3_cfg.get("runtime", {}) or {}
    auto_install_missing = str2bool(
        args.auto_install_missing,
        default=bool(runtime_cfg.get("auto_install_missing", True)),
    )
    strict_sam3_permission = str2bool(
        args.strict_sam3_permission,
        default=bool(runtime_cfg.get("strict_sam3_permission", True)),
    )

    seed_value = int(args.seed) if args.seed is not None else int(config.get("project", {}).get("seed", 42))
    seed_meta = set_global_seed(seed_value, logger)

    device = resolve_device(runtime_cfg=runtime_cfg, logger=logger)
    logger.info("Device: %s", device)
    logger.info(
        "Output policy: video_only=%s write_h264_videos=%s auto_cleanup_intermediates=%s",
        bool(output_policy.get("video_only", True)),
        bool(output_policy.get("write_h264_videos", True)),
        bool(output_policy.get("auto_cleanup_intermediates", True)),
    )

    max_frames = int(args.max_frames) if args.max_frames is not None and int(args.max_frames) > 0 else None

    phase2_ref_token, phase2_pred_root, phase2_metrics_root = resolve_phase2_reference(
        phase2_exp_id_arg=args.phase2_exp_id,
        part3_cfg=part3_cfg,
    )
    if not phase2_pred_root.exists():
        raise FileNotFoundError(
            f"Phase2 reference predictions not found: {phase2_pred_root}. "
            "Run scripts/sync_best_aliases.sh or pass --phase2-exp-id."
        )
    logger.info("Phase2 reference token=%s pred_root=%s", phase2_ref_token, phase2_pred_root)
    phase2_run_meta_path = phase2_metrics_root / "phase2_run_meta.json"
    phase2_run_meta = read_json(phase2_run_meta_path) if phase2_run_meta_path.exists() else {}
    phase2_final_spec = (phase2_run_meta.get("final_best", {}) or {}) if isinstance(phase2_run_meta, dict) else {}
    bbest_mask_backend = str(phase2_final_spec.get("mask_backend", "")).strip().lower()
    bbest_mask_variant = str(phase2_final_spec.get("mask_variant", "")).strip().lower()
    bbest_mask_backend_original = bbest_mask_backend
    bbest_mask_variant_original = bbest_mask_variant
    bbest_backend_source = "phase2_final_best"
    backend_override = str(args.phase4_bbest_backend_override or "").strip().lower()
    if backend_override:
        if backend_override not in {"sam2", "trackanything"}:
            raise RuntimeError(
                "--phase4-bbest-backend-override must be sam2 or trackanything; "
                f"got {backend_override!r}"
            )
        bbest_mask_backend = backend_override
        bbest_mask_variant = f"override_from_{bbest_mask_backend_original or 'unknown'}"
        bbest_backend_source = "cli_override"
    if any(stage.startswith("F") for stage in stages):
        if bbest_mask_backend not in {"sam2", "trackanything"}:
            raise RuntimeError(
                f"Route F requires Phase2 final_best.mask_backend to be sam2 or trackanything; got {bbest_mask_backend!r}"
            )
        logger.info(
            "Route F postprocess backend: backend=%s variant=%s source=%s original_backend=%s",
            bbest_mask_backend,
            bbest_mask_variant or "unknown",
            bbest_backend_source,
            bbest_mask_backend_original or "unknown",
        )

    phase1_ref_token, phase1_metrics_root = resolve_phase1_reference(part3_cfg=part3_cfg)

    selection_cfg = part3_cfg.get("selection", {}) or {}
    part2_selection_cfg = part2_cfg.get("selection", {}) or {}
    selection_primary_metric_source = (
        "part3.selection.primary_metric"
        if "primary_metric" in selection_cfg
        else "part2.selection.primary_metric"
    )
    selection_primary_metric = normalize_selection_primary_metric(
        selection_cfg.get("primary_metric", part2_selection_cfg.get("primary_metric", "auto")),
        default="auto",
    )
    logger.info(
        "Selection primary metric=%s (source=%s)",
        selection_primary_metric,
        selection_primary_metric_source,
    )
    excludes = parse_selection_excludes(
        arg_value=args.selection_exclude_datasets,
        cfg_excludes=(selection_cfg.get("exclude_datasets", []) or []),
    )
    exclude_set = set(excludes)
    selection_datasets = [ds for ds in selected_datasets if ds not in exclude_set]
    if not selection_datasets:
        selection_datasets = list(selected_datasets)
    logger.info("Selection datasets=%s (excluded=%s)", selection_datasets, sorted(list(exclude_set)))

    external_root = REPO_ROOT / "outputs" / "external" / "part3"
    external_root.mkdir(parents=True, exist_ok=True)
    part2_external_root = REPO_ROOT / "outputs" / "external" / "part2"

    f_bbest_backend_env: dict[str, dict[str, Any]] = {}
    if any(stage.startswith("F") for stage in stages):
        f_bbest_backend_env = probe_backend_environment(
            backends=[bbest_mask_backend],
            part2_cfg=part2_cfg,
            external_root=part2_external_root,
            auto_install_missing=auto_install_missing,
            logger=logger,
        )
        if not bool((f_bbest_backend_env.get(bbest_mask_backend, {}) or {}).get("ready", False)):
            raise RuntimeError(
                f"Route F B-best backend is not ready: backend={bbest_mask_backend}, env={f_bbest_backend_env}"
            )

    propainter_repo, propainter_meta = ensure_propainter_ready(
        part2_cfg=part2_cfg,
        external_root=part2_external_root,
        auto_install_missing=auto_install_missing,
        logger=logger,
    )

    sam3_cfg = ((part3_cfg.get("e3", {}) or {}).get("sam3", {}) or {})
    sam3_env_name = args.sam3_env_name or str(sam3_cfg.get("env_name", "sam3"))

    sam3_auto_install = bool(auto_install_missing and bool(sam3_cfg.get("auto_install", True)))
    sam3_repo, sam3_repo_meta = ensure_sam3_repo(
        part3_cfg=part3_cfg,
        external_root=external_root,
        sam3_env_name=sam3_env_name,
        auto_install_missing=sam3_auto_install,
        logger=logger,
    )

    e3_permission_meta: dict[str, Any] = {"checked": False}
    if "E3" in stages:
        e3_permission_meta = check_sam3_permission(
            sam3_cfg=sam3_cfg,
            sam3_env_name=sam3_env_name,
            strict_permission=strict_sam3_permission,
            logger=logger,
        )
        e3_permission_meta["checked"] = True
        logger.info("E3 SAM3 permission: %s", e3_permission_meta)

    target_fps = float(config.get("preprocess", {}).get("target_fps", 24.0))
    gt_root = resolve_repo_path(Path(config.get("paths", {}).get("gt_data_dir", "data/gt")))

    eval_cfg = config.get("evaluation", {}) or {}
    eval_selection_cfg = eval_cfg.get("selection", {}) or {}
    selection_coverage_constraints = parse_selection_coverage_constraints(eval_selection_cfg)
    enforce_selection_coverage = bool(eval_selection_cfg.get("enforce_if_candidate_available", True))
    allow_missing_gt = bool(eval_cfg.get("allow_missing_gt", True))
    save_visualization = bool(eval_cfg.get("save_visualization", True))
    quality_weights_cfg = eval_cfg.get("quality_weights", {}) or {}
    quality_weights = {
        "ros": float(quality_weights_cfg.get("ros", 0.5)),
        "tcf": float(quality_weights_cfg.get("tcf", 0.3)),
        "bes": float(quality_weights_cfg.get("bes", 0.2)),
    }
    ros_cfg = eval_cfg.get("ros", {}) or {}
    tcf_cfg = eval_cfg.get("tcf", {}) or {}
    bes_cfg = eval_cfg.get("bes", {}) or {}
    if selection_coverage_constraints:
        logger.info(
            "Selection coverage constraints=%s enforce_if_candidate_available=%s",
            selection_coverage_constraints,
            str(enforce_selection_coverage).lower(),
        )
    seg_cfg = (config.get("part1", {}) or {}).get("segmentation", {}) or {}
    prompt_cfg = (config.get("part2", {}) or {}).get("prompt", {}) or {}
    dynamic_classes = set(
        str(x).strip().lower()
        for x in prompt_cfg.get(
            "dynamic_classes",
            [
                "person",
                "bicycle",
                "motorcycle",
                "car",
                "bus",
                "truck",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
            ],
        )
        if str(x).strip()
    )
    detector = DynamicObjectDetector(
        backend_priority=[str(x).strip().lower() for x in (ros_cfg.get("backend_priority", ["yolo", "maskrcnn"]) or []) if str(x).strip()],
        dynamic_classes=dynamic_classes,
        yolo_model=str(seg_cfg.get("yolo_model", "yolov8n-seg.pt")),
        yolo_conf=float(seg_cfg.get("yolo_conf_threshold", 0.25)),
        yolo_imgsz=int(seg_cfg.get("yolo_imgsz", 960)),
        maskrcnn_conf=float(seg_cfg.get("maskrcnn_conf_threshold", 0.5)),
        device=device,
    )

    pred_root_base = resolve_repo_path(Path(args.pred_root))
    exp_pred_root = pred_root_base / exp_id
    candidate_root = exp_pred_root / "_candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)

    dataset_payloads: dict[str, DatasetPayload] = {}
    b_masks_map: dict[str, list[np.ndarray]] = {}
    b_mask_load_meta: dict[str, dict[str, Any]] = {}

    for ds in selected_datasets:
        payload = load_dataset_payload(ds, ds_map[ds], max_frames=max_frames)
        dataset_payloads[ds] = payload

        phase2_dataset_root = phase2_pred_root / ds
        b_mask_dir = phase2_dataset_root / "masks"
        _, phase2_mask_video = dataset_video_paths(phase2_dataset_root, output_policy)
        mask_threshold = int(((output_policy.get("mask_h264", {}) or {}).get("threshold", 127)))
        masks, load_meta = load_masks_by_frame_names(
            mask_dir=b_mask_dir,
            frame_names=payload.frame_names,
            frame_shape=payload.frames[0].shape[:2],
            mask_video_path=phase2_mask_video,
            threshold=mask_threshold,
        )
        b_masks_map[ds] = align_masks_to_frame_count(
            masks_u8=masks,
            frame_count=len(payload.frame_names),
            frame_shape=payload.frames[0].shape[:2],
        )
        b_mask_load_meta[ds] = load_meta
        logger.info("Loaded source B masks for %s ratio=%.6f", ds, compute_mean_mask_ratio(b_masks_map[ds]))

    e1_cfg = part3_cfg.get("e1", {}) or {}
    e1_grid = e1_cfg.get("grid", []) or []
    if not isinstance(e1_grid, list) or not e1_grid:
        e1_grid = [
            {
                "name": "morph_a",
                "opening_kernel": 3,
                "closing_kernel": 5,
                "dilation_kernel": 3,
                "smoothing_kernel": 0,
            },
            {
                "name": "morph_b",
                "opening_kernel": 3,
                "closing_kernel": 7,
                "dilation_kernel": 5,
                "smoothing_kernel": 3,
            },
        ]

    e2_cfg = part3_cfg.get("e2", {}) or {}
    e2_windows = [int(x) for x in (e2_cfg.get("temporal_windows", [0, 1, 2]) or [])]
    if not e2_windows:
        e2_windows = [0, 1]

    all_results: list[CandidateResult] = []
    stage_best_map: dict[str, CandidateResult] = {}
    stage_best_masks: dict[str, dict[str, list[np.ndarray]]] = {}

    e3_skipped_reason: str | None = None
    f_route_cfg = part3_cfg.get("f_route", {}) or {}
    f_flow_cfg = f_route_cfg.get("flow", {}) or {}
    f_fusion_grid_cfg = (f_route_cfg.get("fusion", {}) or {}).get("grid", []) or []
    f_traj_grid_cfg = (f_route_cfg.get("trajectory", {}) or {}).get("grid", []) or []
    f4_cfg = f_route_cfg.get("f4", {}) or {}
    f_vggt4d_cfg = f_route_cfg.get("vggt4d", {}) or {}
    f_motion_maps: dict[str, list[np.ndarray]] = {}
    f_motion_scores: dict[str, list[float]] = {}
    f_flow_consistency: dict[str, list[np.ndarray]] = {}
    f1_prior_sources: dict[str, dict[str, list[np.ndarray]]] = {}
    f1_external_masks: dict[str, list[np.ndarray]] = {}
    f1_bbest_post_meta: dict[str, dict[str, Any]] = {}
    f1_prior_rows: list[dict[str, Any]] = []

    def execute_stage(stage: str, specs: list[CandidateSpec], source_masks: dict[str, list[np.ndarray]]) -> CandidateResult:
        stage_results: list[CandidateResult] = []
        stage_masks: dict[str, dict[str, list[np.ndarray]]] = {}

        for spec in specs:
            source_for_spec = source_masks
            if spec.f_source_key and spec.f_source_key in f1_prior_sources:
                source_for_spec = f1_prior_sources[spec.f_source_key]
            try:
                c_root, mask_stats, stage_mask_meta, p_meta, produced_masks = run_candidate(
                    spec=spec,
                    datasets=selected_datasets,
                    dataset_payloads=dataset_payloads,
                    source_masks_map=source_for_spec,
                    out_root=candidate_root,
                    part2_cfg=part2_cfg,
                    propainter_repo=propainter_repo,
                    target_fps=target_fps,
                    sam3_repo=sam3_repo,
                    sam3_env_name=sam3_env_name,
                    sam3_cfg=sam3_cfg,
                    output_policy=output_policy,
                    logger=logger,
                    motion_maps=f_motion_maps or None,
                    motion_scores=f_motion_scores or None,
                    flow_consistency_maps=f_flow_consistency or None,
                    external_masks_by_dataset=f1_external_masks or None,
                )
            except Exception as e:
                logger.warning("Candidate failed: stage=%s candidate=%s reason=%s", stage, spec.name, str(e))
                logger.debug(traceback.format_exc())
                if stage in {"E1", "E2", "E4"}:
                    raise
                continue

            eval_exp_id = f"{exp_id}__{stage}__{spec.candidate_id}"
            summary_path, summary = run_evaluation(
                config_path=config_path,
                datasets=selected_datasets,
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
                stage_mask_meta=stage_mask_meta,
                propainter_meta=p_meta,
            )
            stage_results.append(result)
            all_results.append(result)
            stage_masks[spec.name] = produced_masks

            logger.info(
                "[%s] candidate=%s -> JM=%.4f JR=%.4f ROS=%.4f TCF=%.4f BES=%.4f",
                stage,
                spec.name,
                metric_or_neg_inf(result.aggregate, "JM"),
                metric_or_neg_inf(result.aggregate, "JR"),
                metric_or_neg_inf(result.aggregate, "ROS"),
                metric_or_neg_inf(result.aggregate, "TCF"),
                metric_or_neg_inf(result.aggregate, "BES"),
            )

        if not stage_results:
            raise RuntimeError(f"No successful candidate in {stage}")

        best = select_best(
            stage,
            stage_results,
            score_datasets=selection_datasets,
            coverage_constraints=selection_coverage_constraints,
            enforce_if_candidate_available=enforce_selection_coverage,
            primary_metric=selection_primary_metric,
            logger=logger,
        )
        stage_best_map[stage] = best
        stage_best_masks[stage] = stage_masks[best.spec.name]
        logger.info("[%s] best candidate=%s", stage, best.spec.name)
        return best

    best_e1: CandidateResult | None = None
    best_e2: CandidateResult | None = None
    best_e3: CandidateResult | None = None
    best_e4: CandidateResult | None = None
    best_f1: CandidateResult | None = None
    best_f2: CandidateResult | None = None
    best_f3: CandidateResult | None = None
    best_f4: CandidateResult | None = None
    best_f5: CandidateResult | None = None

    if "E1" in stages:
        e1_specs: list[CandidateSpec] = []
        for item in e1_grid:
            if not isinstance(item, dict):
                continue
            e1_specs.append(
                CandidateSpec(
                    stage="E1",
                    name=sanitize_name(str(item.get("name", f"e1_{len(e1_specs)}"))),
                    source_stage="B",
                    e1_profile={
                        "opening_kernel": int(item.get("opening_kernel", 3)),
                        "closing_kernel": int(item.get("closing_kernel", 5)),
                        "dilation_kernel": int(item.get("dilation_kernel", 3)),
                        "smoothing_kernel": int(item.get("smoothing_kernel", 0)),
                    },
                    temporal_window=0,
                    use_sam3=False,
                )
            )
        if not e1_specs:
            raise RuntimeError("E1 grid resolved to empty")
        best_e1 = execute_stage("E1", e1_specs, b_masks_map)

    if "E2" in stages:
        anchor_masks = stage_best_masks.get("E1")
        if anchor_masks is None:
            raise RuntimeError("E2 requires E1 best masks")

        e2_specs = [
            CandidateSpec(
                stage="E2",
                name=f"temporal_w{tw}",
                source_stage="E1",
                e1_profile={},
                temporal_window=int(tw),
                use_sam3=False,
            )
            for tw in e2_windows
        ]
        best_e2 = execute_stage("E2", e2_specs, anchor_masks)

    if "E3" in stages:
        anchor_masks = stage_best_masks.get("E2")
        if anchor_masks is None:
            raise RuntimeError("E3 requires E2 best masks")

        e3_specs = [
            CandidateSpec(
                stage="E3",
                name="sam3_refine",
                source_stage="E2",
                e1_profile={},
                temporal_window=0,
                use_sam3=True,
            )
        ]

        try:
            best_e3 = execute_stage("E3", e3_specs, anchor_masks)
        except Exception as e:
            e3_skipped_reason = str(e)
            flow = decide_e3_flow(
                permission_passed=bool(e3_permission_meta.get("passed", False)),
                runtime_error=e3_skipped_reason,
            )
            if flow == "abort":
                raise
            logger.warning("E3 skipped due to runtime error after permission check: %s", e3_skipped_reason)

    if "E4" in stages:
        anchor_masks = stage_best_masks.get("E3") or stage_best_masks.get("E2")
        if anchor_masks is None:
            raise RuntimeError("E4 requires E3 or E2 best masks")

        e4_specs = [
            CandidateSpec(
                stage="E4",
                name="b_plus_e_finalize",
                source_stage="E3" if stage_best_masks.get("E3") else "E2",
                e1_profile={},
                temporal_window=0,
                use_sam3=bool(stage_best_masks.get("E3")),
            )
        ]
        best_e4 = execute_stage("E4", e4_specs, anchor_masks)

    has_f_stages = any(stage.startswith("F") for stage in stages)
    if has_f_stages:
        logger.info("=== Route F start ===")
        f1_external_root = REPO_ROOT / "outputs" / "external" / "vggt4d" / exp_id
        f1_external_root.mkdir(parents=True, exist_ok=True)

        yolo_detector = DynamicObjectDetector(
            backend_priority=["yolo"],
            dynamic_classes=dynamic_classes,
            yolo_model=str(seg_cfg.get("yolo_model", "yolov8n-seg.pt")),
            yolo_conf=float(seg_cfg.get("yolo_conf_threshold", 0.25)),
            yolo_imgsz=int(seg_cfg.get("yolo_imgsz", 960)),
            maskrcnn_conf=float(seg_cfg.get("maskrcnn_conf_threshold", 0.5)),
            device=device,
        )

        f1_prior_sources["yolo"] = {}
        f1_prior_sources["vggt4d"] = {}
        f1_prior_sources["vggt4d_yolo"] = {}

        vggt4d_masks_map, vggt4d_meta = generate_vggt4d_dynamic_priors(
            datasets_frames_bgr={ds: dataset_payloads[ds].frames for ds in selected_datasets},
            output_root=f1_external_root,
            cfg={**f_vggt4d_cfg, "device": device},
            logger_obj=logger,
        )
        bbest_post_cfg = f_route_cfg.get("bbest_postprocess", {}) or {}
        vggt_prior_max_prompts = int(
            bbest_post_cfg.get("max_prompts", (part2_cfg.get("prompt", {}) or {}).get("max_prompts", 3))
        )
        vggt_prior_min_area_ratio = float(bbest_post_cfg.get("min_prompt_area_ratio", 1e-6))
        f1_bbest_post_meta: dict[str, dict[str, Any]] = {}

        for ds in selected_datasets:
            payload = dataset_payloads[ds]
            ds_h, ds_w = payload.frames[0].shape[:2]

            yolo_masks: list[np.ndarray] = []
            for frame in payload.frames:
                m, _ = yolo_detector.detect_mask(frame)
                yolo_masks.append(((m > 0).astype(np.uint8) * 255))

            vggt4d_masks = vggt4d_masks_map.get(ds, [])

            yolo_masks = align_masks_to_frame_count(yolo_masks, len(payload.frame_names), (ds_h, ds_w))
            vggt4d_raw_masks = align_masks_to_frame_count(vggt4d_masks, len(payload.frame_names), (ds_h, ds_w))
            vggt4d_masks, bbest_post_meta = run_bbest_backend_postprocess_on_prior(
                dataset_name=ds,
                payload=payload,
                prior_masks_u8=vggt4d_raw_masks,
                backend=bbest_mask_backend,
                part2_cfg=part2_cfg,
                backend_env=f_bbest_backend_env,
                device=device,
                max_prompts=vggt_prior_max_prompts,
                min_area_ratio=vggt_prior_min_area_ratio,
            )
            f1_bbest_post_meta[ds] = bbest_post_meta
            vggt4d_masks = align_masks_to_frame_count(vggt4d_masks, len(payload.frame_names), (ds_h, ds_w))
            union_masks = [
                (((np.asarray(a) > 0) | (np.asarray(b) > 0)).astype(np.uint8) * 255)
                for a, b in zip(yolo_masks, vggt4d_masks)
            ]

            f1_prior_sources["yolo"][ds] = yolo_masks
            f1_prior_sources["vggt4d"][ds] = vggt4d_masks
            f1_prior_sources["vggt4d_yolo"][ds] = union_masks
            f1_external_masks[ds] = vggt4d_masks

            for name, masks, source_name in [
                ("yolo", yolo_masks, "detector_yolo"),
                ("vggt4d_raw", vggt4d_raw_masks, str(vggt4d_meta.get("backend", "vggt4d"))),
                ("vggt4d", vggt4d_masks, f"vggt4d_with_bbest_backend_{bbest_mask_backend}"),
                ("vggt4d_yolo", union_masks, "union"),
            ]:
                f1_prior_rows.append(
                    {
                        "dataset": ds,
                        "prior": name,
                        "mean_mask_ratio": compute_mean_mask_ratio(masks),
                        "active_frame_ratio": compute_active_frame_ratio(masks),
                        "frame_count": len(masks),
                        "source": source_name,
                    }
                )

            logger.info(
                "[F1] %s priors ready | yolo=%.6f vggt4d_raw=%.6f vggt4d_prior=%.6f union=%.6f | bbest_backend=%s prompt_frame=%s boxes=%s",
                ds,
                compute_mean_mask_ratio(yolo_masks),
                compute_mean_mask_ratio(vggt4d_raw_masks),
                compute_mean_mask_ratio(vggt4d_masks),
                compute_mean_mask_ratio(union_masks),
                bbest_mask_backend,
                bbest_post_meta.get("prompt_meta", {}).get("anchor_frames"),
                bbest_post_meta.get("prompt_meta", {}).get("anchor_count"),
            )
            logger.info("[F1] %s B-best backend postprocess meta: %s", ds, bbest_post_meta)

            compensate_global = bool(f_flow_cfg.get("compensate_global", True))
            f_motion_maps[ds] = compute_video_motion_maps(payload.frames, f_flow_cfg, compensate_global=compensate_global)
            f_motion_scores[ds] = compute_video_instance_motion_scores(
                payload.frames,
                b_masks_map[ds],
                f_flow_cfg,
                compensate_global=compensate_global,
            )
            if bool(f4_cfg.get("enable_bidirectional", True)):
                farneback_cfg = (f_flow_cfg.get("farneback", {}) or {})
                f_flow_consistency[ds] = compute_video_flow_consistency(payload.frames, farneback_cfg=farneback_cfg)

        if "F1" in stages:
            # F1: final Route F variant using postprocessed VGGT4D prior.
            f1_specs = [
                CandidateSpec(
                    stage="F1",
                    name="bbest_vggt4d_replace_yolo",
                    source_stage="B",
                    e1_profile={},
                    temporal_window=0,
                    use_sam3=False,
                    f_source_key="vggt4d",
                )
            ]
            best_f1 = execute_stage("F1", f1_specs, b_masks_map)

        if "F2" in stages:
            # F2: untouched B-best baseline for direct comparison.
            f2_specs = [
                CandidateSpec(
                    stage="F2",
                    name="bbest_baseline",
                    source_stage="B",
                    e1_profile={},
                    temporal_window=0,
                    use_sam3=False,
                )
            ]
            best_f2 = execute_stage("F2", f2_specs, b_masks_map)

        if "F3" in stages:
            # F3: prior-fusion ablations only; never exported as Route F final output.
            anchor_masks = stage_best_masks.get("F2") or b_masks_map
            if not f_fusion_grid_cfg:
                f_fusion_grid_cfg = [
                    {"name": "vggt4d_guided", "method": "vggt4d_guided", "vggt4d_alpha": 0.35, "vggt4d_beta": 0.35, "vggt4d_threshold": 0.5},
                    {"name": "weighted_a04", "method": "weighted", "weighted_alpha": 0.4, "weighted_threshold": 0.5},
                    {"name": "weighted_a06", "method": "weighted", "weighted_alpha": 0.6, "weighted_threshold": 0.5},
                    {"name": "intersection", "method": "intersection", "pixel_motion_threshold": 1.5},
                    {"name": "union", "method": "union", "pixel_motion_threshold": 1.5},
                ]
            else:
                has_vggt4d_guided = any(
                    str(item.get("method", item.get("name", ""))).strip().lower() == "vggt4d_guided"
                    for item in f_fusion_grid_cfg
                    if isinstance(item, dict)
                )
                if not has_vggt4d_guided:
                    f_fusion_grid_cfg = list(f_fusion_grid_cfg) + [
                        {
                            "name": "vggt4d_guided_auto",
                            "method": "vggt4d_guided",
                            "vggt4d_alpha": 0.35,
                            "vggt4d_beta": 0.35,
                            "vggt4d_threshold": 0.5,
                        }
                    ]
            f3_specs: list[CandidateSpec] = []
            for item in f_fusion_grid_cfg:
                method = str(item.get("method", item.get("name", ""))).strip().lower()
                if method not in SUPPORTED_FUSION_METHODS:
                    continue
                cfg_copy = {k: v for k, v in item.items() if k not in {"name", "method"}}
                f3_specs.append(
                    CandidateSpec(
                        stage="F3",
                        name=sanitize_name(str(item.get("name", method))),
                        source_stage="F2",
                        e1_profile={},
                        temporal_window=0,
                        use_sam3=False,
                        f_fusion_method=method,
                        f_fusion_cfg=cfg_copy,
                    )
                )
            if not f3_specs:
                raise RuntimeError("F3 fusion grid resolved to empty")
            best_f3 = execute_stage("F3", f3_specs, anchor_masks)

        if "F4" in stages:
            anchor_masks = stage_best_masks.get("F3") or stage_best_masks.get("F2") or b_masks_map
            if anchor_masks is None:
                raise RuntimeError("F4 requires F3/F2/B masks")

            f4_grid = f4_cfg.get("grid", []) or []
            if not f4_grid:
                f4_grid = [{"name": "bidir_e3.0", "max_consistency_error": float(f4_cfg.get("max_consistency_error", 3.0))}]
            if not f_traj_grid_cfg:
                f_traj_grid_cfg = [
                    {"name": "balanced", "min_track_length": 3, "motion_smooth_window": 3, "track_motion_threshold": 1.5}
                ]

            best_fusion_method = "weighted"
            best_fusion_cfg: dict[str, Any] = {}
            if best_f3 is not None:
                best_fusion_method = best_f3.spec.f_fusion_method or "weighted"
                best_fusion_cfg = dict(best_f3.spec.f_fusion_cfg)

            enable_bidir = bool(f4_cfg.get("enable_bidirectional", True))
            f4_specs: list[CandidateSpec] = []
            for grid_item in f4_grid:
                for traj_item in f_traj_grid_cfg:
                    merged_cfg = dict(best_fusion_cfg)
                    merged_cfg["max_consistency_error"] = float(
                        grid_item.get("max_consistency_error", f4_cfg.get("max_consistency_error", 3.0))
                    )
                    f4_specs.append(
                        CandidateSpec(
                            stage="F4",
                            name=sanitize_name(f"{grid_item.get('name', 'f4')}_{traj_item.get('name', 'traj')}"),
                            source_stage="F3" if stage_best_masks.get("F3") else ("F2" if stage_best_masks.get("F2") else "B"),
                            e1_profile={},
                            temporal_window=0,
                            use_sam3=False,
                            f_fusion_method=best_fusion_method,
                            f_fusion_cfg=merged_cfg,
                            f_trajectory_cfg={k: v for k, v in traj_item.items() if k != "name"},
                            f_use_bidirectional=enable_bidir,
                        )
                    )
            if not f4_specs:
                raise RuntimeError("F4 grid resolved to empty")
            best_f4 = execute_stage("F4", f4_specs, anchor_masks)

        if "F5" in stages:
            anchor_masks = (
                stage_best_masks.get("F4")
                or stage_best_masks.get("F3")
                or stage_best_masks.get("F2")
                or b_masks_map
            )
            if anchor_masks is None:
                raise RuntimeError("F5 requires earlier F stages")

            f5_specs = [
                CandidateSpec(
                    stage="F5",
                    name="b_plus_f_finalize",
                    source_stage="F4" if stage_best_masks.get("F4") else ("F3" if stage_best_masks.get("F3") else "F2"),
                    e1_profile={},
                    temporal_window=0,
                    use_sam3=False,
                )
            ]
            best_f5 = execute_stage("F5", f5_specs, anchor_masks)

    f_stage_bests = [x for x in [best_f1, best_f2, best_f3, best_f4, best_f5] if x is not None]
    f_final: CandidateResult | None = None
    if f_stage_bests:
        f_final = select_f_route_best(
            entries=f_stage_bests,
            score_datasets=selection_datasets,
            coverage_constraints=selection_coverage_constraints,
            enforce_if_candidate_available=enforce_selection_coverage,
            logger=logger,
        )
        logger.info(
            "[F-best] candidate=%s stage=%s source=%s",
            f_final.spec.name,
            f_final.spec.stage,
            f_final.spec.source_stage,
        )

    e_final = best_e4 or best_e3 or best_e2 or best_e1
    final_best = f_final or e_final
    if final_best is None:
        raise RuntimeError("Part3 finished without successful candidate")

    copy_candidate_outputs(best_candidate_root=final_best.candidate_root, final_root=exp_pred_root, datasets=selected_datasets)

    final_summary_path, final_summary = run_evaluation(
        config_path=config_path,
        datasets=selected_datasets,
        pred_root=exp_pred_root,
        gt_root=gt_root,
        eval_exp_id=exp_id,
        allow_missing_gt=allow_missing_gt,
        save_visualization=save_visualization,
        logger=logger,
    )

    phase_label = args.phase or ("phase4" if has_f_stages else "phase3")
    metrics_dir = REPO_ROOT / "outputs" / "metrics" / exp_id
    ablation_csv, selection_json = write_ablation_outputs(
        metrics_dir=metrics_dir,
        all_results=all_results,
        stage_best_map=stage_best_map,
        final_best=final_best,
        phase_label=phase_label,
    )

    phase2_summary_path = phase2_metrics_root / "summary.json"
    if not phase2_summary_path.exists():
        raise FileNotFoundError(
            f"Phase2 summary missing for reference '{phase2_ref_token}': {phase2_summary_path}"
        )
    phase2_summary = read_json(phase2_summary_path)

    b_vs_e_csv, b_vs_e_meta = write_b_vs_e_comparison(
        metrics_dir=metrics_dir,
        phase2_summary=phase2_summary,
        phase2_ref_token=phase2_ref_token,
        phase3_summary=final_summary,
    )

    b_vs_f_csv: Path | None = None
    b_vs_f_meta: dict[str, Any] = {}
    if has_f_stages:
        b_vs_f_csv, b_vs_f_meta = write_b_vs_f_comparison(
            metrics_dir=metrics_dir,
            phase2_summary=phase2_summary,
            phase2_ref_token=phase2_ref_token,
            phase4_summary=final_summary,
        )

    priors_csv: Path | None = None
    if has_f_stages and f1_prior_rows:
        priors_csv = write_phase4_mask_priors(metrics_dir=metrics_dir, rows=f1_prior_rows)

    figures_dir = REPO_ROOT / "outputs" / "figures" / exp_id
    failure_csv, failure_explained_csv = build_failure_case_index(
        pred_root=exp_pred_root,
        gt_root=gt_root,
        datasets=selected_datasets,
        out_dir=figures_dir / "failure_cases",
        detector=detector,
        quality_weights=quality_weights,
        tcf_dilate_kernel=int(tcf_cfg.get("dilate_kernel", 5)),
        bes_dilate_kernel=int(bes_cfg.get("dilate_kernel", 5)),
        bes_erode_kernel=int(bes_cfg.get("erode_kernel", 3)),
        bes_sobel_ksize=int(bes_cfg.get("sobel_ksize", 3)),
        backend_meta=final_best.stage_mask_meta,
        propainter_meta=final_best.propainter_meta,
        top_k=3,
    )

    report_path = metrics_dir / f"{phase_label}_acceptance_report.md"
    compare_csv = b_vs_f_csv if b_vs_f_csv is not None else b_vs_e_csv
    write_acceptance_report(
        report_path=report_path,
        exp_id=exp_id,
        phase_label=phase_label,
        aggregate=final_summary.get("aggregate", {}),
        stage_best_map=stage_best_map,
        final_best=final_best,
        per_dataset=final_summary.get("datasets", {}),
        phase2_ref_token=phase2_ref_token,
        compare_csv=compare_csv,
        failure_explained_csv=failure_explained_csv,
        seed=seed_value,
        selection_datasets=selection_datasets,
        e3_permission_meta=e3_permission_meta,
        e3_skipped_reason=e3_skipped_reason,
        priors_csv=priors_csv,
    )

    write_json(
        metrics_dir / f"{phase_label}_run_meta.json",
        {
            "exp_id": exp_id,
            "phase_label": phase_label,
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "config": str(config_path),
            "datasets": selected_datasets,
            "selection_datasets": selection_datasets,
            "excluded_datasets": sorted(list(exclude_set)),
            "selection_primary_metric": selection_primary_metric,
            "selection_primary_metric_source": selection_primary_metric_source,
            "stages": stages,
            "seed": seed_value,
            "seed_meta": seed_meta,
            "device": device,
            "auto_install_missing": auto_install_missing,
            "output_policy": output_policy,
            "sam3_auto_install": sam3_auto_install,
            "phase2_reference": phase2_ref_token,
            "phase2_summary": str(phase2_summary_path),
            "phase1_reference": phase1_ref_token,
            "phase1_metrics_root": str(phase1_metrics_root),
            "propainter_environment": propainter_meta,
            "sam3_repo": sam3_repo_meta,
            "sam3_permission": e3_permission_meta,
            "e3_skipped_reason": e3_skipped_reason,
            "b_mask_load_meta": b_mask_load_meta,
            "ablation_csv": str(ablation_csv),
            "selection_json": str(selection_json),
            "summary_json": str(final_summary_path),
            "phase3_b_vs_e_csv": str(b_vs_e_csv),
            "phase3_b_vs_e_meta": b_vs_e_meta,
            "phase4_b_vs_f_csv": str(b_vs_f_csv) if b_vs_f_csv is not None else None,
            "phase4_b_vs_f_meta": b_vs_f_meta,
            "phase4_mask_priors_csv": str(priors_csv) if priors_csv is not None else None,
            "phase4_bbest_backend": bbest_mask_backend if has_f_stages else None,
            "phase4_bbest_backend_variant": bbest_mask_variant if has_f_stages else None,
            "phase4_bbest_backend_source": bbest_backend_source if has_f_stages else None,
            "phase4_bbest_backend_original": bbest_mask_backend_original if has_f_stages else None,
            "phase4_bbest_backend_variant_original": bbest_mask_variant_original if has_f_stages else None,
            "phase4_bbest_backend_environment": f_bbest_backend_env if has_f_stages else None,
            "phase4_mask_propagation": normalize_mask_propagation_cfg(part2_cfg) if has_f_stages else None,
            "phase4_bbest_postprocess_meta": f1_bbest_post_meta if has_f_stages else None,
            "failure_csv": str(failure_csv),
            "failure_explained_csv": str(failure_explained_csv),
            "acceptance_report": str(report_path),
            "log_path": str(log_path),
            "final_best": asdict(final_best.spec),
            "f_best": asdict(f_final.spec) if f_final is not None else None,
            "phase4_final_policy": "force_vggt4d_prior" if has_f_stages else None,
            "f_stage_best_candidates": [asdict(x.spec) for x in f_stage_bests],
            "has_f_stages": has_f_stages,
            "f1_prior_sources": sorted(list(f1_prior_sources.keys())),
        },
    )

    cleanup_stats = cleanup_video_only_outputs(
        exp_pred_root=exp_pred_root,
        datasets=selected_datasets,
        output_policy=output_policy,
    )
    logger.info("Video-only cleanup stats: %s", cleanup_stats)

    logger.info("%s summary: %s", phase_label.upper(), final_summary_path)
    logger.info("%s ablation csv: %s", phase_label.upper(), ablation_csv)
    logger.info("%s selection json: %s", phase_label.upper(), selection_json)
    logger.info("Phase3 B-vs-E csv: %s", b_vs_e_csv)
    if b_vs_f_csv is not None:
        logger.info("Phase4 B-vs-F csv: %s", b_vs_f_csv)
    if priors_csv is not None:
        logger.info("Phase4 mask priors csv: %s", priors_csv)
    logger.info("%s acceptance report: %s", phase_label.upper(), report_path)


if __name__ == "__main__":
    main()
