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
from dataclasses import asdict, dataclass
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
    build_failure_case_index,
    collect_dataset_cfg,
    ensure_propainter_ready,
    load_dataset_payload,
    metric_or_neg_inf,
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

SUPPORTED_STAGES = ["E1", "E2", "E3", "E4"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass
class CandidateSpec:
    stage: str
    name: str
    source_stage: str
    e1_profile: dict[str, Any]
    temporal_window: int
    use_sam3: bool

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


def dataset_metric_aggregate(
    per_dataset: dict[str, Any],
    datasets_for_scoring: list[str] | None,
) -> dict[str, float]:
    keys = ["JM", "JR", "ROS", "TCF", "BES", "Q_REMOVE"]
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


def stage_score(stage: str, agg: dict[str, float], mean_mask_ratio: float) -> tuple[float, float, float, float]:
    jm = float(agg.get("JM", float("-inf")))
    jr = float(agg.get("JR", float("-inf")))
    q_remove = float(agg.get("Q_REMOVE", float("-inf")))
    if stage in {"E1", "E2", "E3"}:
        return (jm, jr, q_remove, -abs(mean_mask_ratio - 0.1))
    return (q_remove, jm, jr, -abs(mean_mask_ratio - 0.1))


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
        return stage_score(stage, agg, mean_ratio)

    return max(pool, key=score)


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
            "JM": r.aggregate.get("JM"),
            "JR": r.aggregate.get("JR"),
            "ROS": r.aggregate.get("ROS"),
            "TCF": r.aggregate.get("TCF"),
            "BES": r.aggregate.get("BES"),
            "Q_REMOVE": r.aggregate.get("Q_REMOVE"),
            "mean_mask_ratio": mean_ratio,
            "active_frame_ratio": active_ratio,
            "pred_root": str(r.candidate_root),
            "eval_exp_id": r.eval_exp_id,
            "is_stage_best": int(stage_best_map.get(r.spec.stage) is r),
            "is_final_best": int(final_best is r),
        }
        rows.append(row)

    csv_path = metrics_dir / "phase3_ablation.csv"
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
                "JM",
                "JR",
                "ROS",
                "TCF",
                "BES",
                "Q_REMOVE",
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

    selection_path = metrics_dir / "phase3_selection.json"
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
            "B_Q_REMOVE": bm.get("Q_REMOVE"),
            "E_Q_REMOVE": em.get("Q_REMOVE"),
            "delta_Q_REMOVE": None if bm.get("Q_REMOVE") is None or em.get("Q_REMOVE") is None else float(em.get("Q_REMOVE")) - float(bm.get("Q_REMOVE")),
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
            "B_Q_REMOVE": b_agg.get("Q_REMOVE"),
            "E_Q_REMOVE": e_agg.get("Q_REMOVE"),
            "delta_Q_REMOVE": None if b_agg.get("Q_REMOVE") is None or e_agg.get("Q_REMOVE") is None else float(e_agg.get("Q_REMOVE")) - float(b_agg.get("Q_REMOVE")),
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
                "B_Q_REMOVE",
                "E_Q_REMOVE",
                "delta_Q_REMOVE",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "status": "ok",
        "phase2_reference": phase2_ref_token,
    }
    return out_path, meta


def write_acceptance_report(
    report_path: Path,
    exp_id: str,
    aggregate: dict[str, Any],
    stage_best_map: dict[str, CandidateResult],
    final_best: CandidateResult,
    per_dataset: dict[str, Any],
    phase2_ref_token: str,
    b_vs_e_csv: Path,
    failure_explained_csv: Path,
    seed: int,
    selection_datasets: list[str],
    e3_permission_meta: dict[str, Any],
    e3_skipped_reason: str | None,
) -> None:
    lines: list[str] = []
    lines.append(f"# Phase 3 Acceptance Report: `{exp_id}`")
    lines.append("")
    lines.append("## Final Aggregate")
    lines.append("")
    lines.append("| JM | JR | ROS | TCF | BES | Q_REMOVE |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| {aggregate.get('JM')} | {aggregate.get('JR')} | {aggregate.get('ROS')} | {aggregate.get('TCF')} | {aggregate.get('BES')} | {aggregate.get('Q_REMOVE')} |"
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

    lines.append("## Final E-best")
    lines.append("")
    s = final_best.spec
    lines.append(
        f"- `{s.name}`: source={s.source_stage}, use_sam3={s.use_sam3}, profile={s.e1_profile}, temporal_window={s.temporal_window}"
    )
    lines.append("")

    lines.append("## Per-Dataset Metrics")
    lines.append("")
    lines.append("| Dataset | JM | JR | ROS | TCF | BES | Q_REMOVE |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for ds_name, ds_payload in per_dataset.items():
        metrics = ds_payload.get("metrics", {}) if isinstance(ds_payload, dict) else {}
        lines.append(
            f"| {ds_name} | {metrics.get('JM')} | {metrics.get('JR')} | {metrics.get('ROS')} | {metrics.get('TCF')} | {metrics.get('BES')} | {metrics.get('Q_REMOVE')} |"
        )
    lines.append("")

    lines.append("## B-best vs B+E")
    lines.append("")
    lines.append(f"- Phase2 reference: `{phase2_ref_token}`")
    lines.append(f"- Comparison CSV: `{b_vs_e_csv}`")
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
    parser = argparse.ArgumentParser(description="Run Part3 exploration pipeline (E1-E4).")
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
    exp_id = args.exp_id or f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger, log_path = setup_logger(exp_id)
    logger.info("Phase3 start | exp_id=%s", exp_id)

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

    phase1_ref_token, phase1_metrics_root = resolve_phase1_reference(part3_cfg=part3_cfg)

    selection_cfg = part3_cfg.get("selection", {}) or {}
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

    propainter_repo, propainter_meta = ensure_propainter_ready(
        part2_cfg=part2_cfg,
        external_root=REPO_ROOT / "outputs" / "external" / "part2",
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

    def execute_stage(stage: str, specs: list[CandidateSpec], source_masks: dict[str, list[np.ndarray]]) -> CandidateResult:
        stage_results: list[CandidateResult] = []
        stage_masks: dict[str, dict[str, list[np.ndarray]]] = {}

        for spec in specs:
            try:
                c_root, mask_stats, stage_mask_meta, p_meta, produced_masks = run_candidate(
                    spec=spec,
                    datasets=selected_datasets,
                    dataset_payloads=dataset_payloads,
                    source_masks_map=source_masks,
                    out_root=candidate_root,
                    part2_cfg=part2_cfg,
                    propainter_repo=propainter_repo,
                    target_fps=target_fps,
                    sam3_repo=sam3_repo,
                    sam3_env_name=sam3_env_name,
                    sam3_cfg=sam3_cfg,
                    output_policy=output_policy,
                    logger=logger,
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
                "[%s] candidate=%s -> JM=%.4f JR=%.4f ROS=%.4f TCF=%.4f BES=%.4f Q_REMOVE=%.4f",
                stage,
                spec.name,
                metric_or_neg_inf(result.aggregate, "JM"),
                metric_or_neg_inf(result.aggregate, "JR"),
                metric_or_neg_inf(result.aggregate, "ROS"),
                metric_or_neg_inf(result.aggregate, "TCF"),
                metric_or_neg_inf(result.aggregate, "BES"),
                metric_or_neg_inf(result.aggregate, "Q_REMOVE"),
            )

        if not stage_results:
            raise RuntimeError(f"No successful candidate in {stage}")

        best = select_best(
            stage,
            stage_results,
            score_datasets=selection_datasets,
            coverage_constraints=selection_coverage_constraints,
            enforce_if_candidate_available=enforce_selection_coverage,
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

    final_best = best_e4 or best_e3 or best_e2 or best_e1
    if final_best is None:
        raise RuntimeError("Phase3 finished without successful candidate")

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

    metrics_dir = REPO_ROOT / "outputs" / "metrics" / exp_id
    ablation_csv, selection_json = write_ablation_outputs(
        metrics_dir=metrics_dir,
        all_results=all_results,
        stage_best_map=stage_best_map,
        final_best=final_best,
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

    report_path = metrics_dir / "phase3_acceptance_report.md"
    write_acceptance_report(
        report_path=report_path,
        exp_id=exp_id,
        aggregate=final_summary.get("aggregate", {}),
        stage_best_map=stage_best_map,
        final_best=final_best,
        per_dataset=final_summary.get("datasets", {}),
        phase2_ref_token=phase2_ref_token,
        b_vs_e_csv=b_vs_e_csv,
        failure_explained_csv=failure_explained_csv,
        seed=seed_value,
        selection_datasets=selection_datasets,
        e3_permission_meta=e3_permission_meta,
        e3_skipped_reason=e3_skipped_reason,
    )

    write_json(
        metrics_dir / "phase3_run_meta.json",
        {
            "exp_id": exp_id,
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "config": str(config_path),
            "datasets": selected_datasets,
            "selection_datasets": selection_datasets,
            "excluded_datasets": sorted(list(exclude_set)),
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
            "failure_csv": str(failure_csv),
            "failure_explained_csv": str(failure_explained_csv),
            "acceptance_report": str(report_path),
            "log_path": str(log_path),
            "final_best": asdict(final_best.spec),
        },
    )

    cleanup_stats = cleanup_video_only_outputs(
        exp_pred_root=exp_pred_root,
        datasets=selected_datasets,
        output_policy=output_policy,
    )
    logger.info("Video-only cleanup stats: %s", cleanup_stats)

    logger.info("Phase3 summary: %s", final_summary_path)
    logger.info("Phase3 ablation csv: %s", ablation_csv)
    logger.info("Phase3 selection json: %s", selection_json)
    logger.info("Phase3 B-vs-E csv: %s", b_vs_e_csv)
    logger.info("Phase3 acceptance report: %s", report_path)


if __name__ == "__main__":
    main()
