#!/usr/bin/env python3
"""Phase 5 Route G: GPU diffusion inpainting pipeline."""
from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.part1.run_baseline import write_dataset_outputs
from src.part2.run_sota import (
    collect_dataset_cfg,
    read_json,
    resolve_dataset_names,
    run_evaluation,
    write_json,
)
from src.common.video_io import (
    cleanup_video_only_outputs,
    decode_video_frames,
    resolve_output_policy,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

SD_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
POSITIVE_PROMPT = "high quality, realistic texture, natural lighting, clean empty background, temporally consistent video frame"
NEGATIVE_PROMPT = "person, human, cyclist, bicycle, tennis player, racket, car, vehicle, text, logo, watermark, cartoon, painting, blurry, distorted geometry, inconsistent lighting"

# G variants per PLAN.md
VARIANTS = [
    {"name": "G-low",    "mask_dilation": 2,  "denoise_strength": 0.35, "keyframe_interval": 1},
    {"name": "G-mid",    "mask_dilation": 10, "denoise_strength": 0.55, "keyframe_interval": 4},
    {"name": "G-high",   "mask_dilation": 20, "denoise_strength": 0.75, "keyframe_interval": 1},
    {"name": "G-hybrid", "mask_dilation": 10, "denoise_strength": 0.55, "keyframe_interval": 8},
]

PHASE2_REF = "phase2_20260430_175248"


def setup_logger(exp_id: str) -> logging.Logger:
    log_dir = REPO_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_dir / f"phase5_{exp_id}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def load_pipeline(device: str) -> Any:
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def dilate_mask(mask_u8: np.ndarray, dilation: int) -> np.ndarray:
    if dilation <= 0:
        return mask_u8
    k = dilation * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask_u8, kernel)


def diffusion_inpaint_frame(
    pipe: Any,
    frame_bgr: np.ndarray,
    mask_u8: np.ndarray,
    denoise_strength: float,
    seed: int,
    steps: int = 20,
) -> np.ndarray:
    import torch
    from PIL import Image

    h, w = frame_bgr.shape[:2]
    # SD requires multiples of 8
    ph = ((h + 7) // 8) * 8
    pw = ((w + 7) // 8) * 8

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb).resize((pw, ph), Image.LANCZOS)
    pil_mask = Image.fromarray(mask_u8).resize((pw, ph), Image.NEAREST)

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    result = pipe(
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=pil_image,
        mask_image=pil_mask,
        strength=denoise_strength,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    result_np = np.array(result.resize((w, h), Image.LANCZOS))
    return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)


def propagate_keyframes(
    frames: list[np.ndarray],
    masks: list[np.ndarray],
    keyframe_results: dict[int, np.ndarray],
) -> list[np.ndarray]:
    """Fill non-keyframes by blending nearest keyframe results with original."""
    n = len(frames)
    out = []
    for i in range(n):
        mask = masks[i] > 0
        if not mask.any():
            out.append(frames[i])
            continue
        # Find nearest keyframe
        nearest = min(keyframe_results.keys(), key=lambda k: abs(k - i))
        kf = keyframe_results[nearest]
        # Blend: use diffusion result in mask region, original outside
        blended = frames[i].copy()
        blended[mask] = kf[mask]
        out.append(blended)
    return out


def run_variant(
    variant: dict[str, Any],
    datasets: list[str],
    ds_cfgs: dict[str, dict],
    source_exp_id: str,
    exp_id: str,
    seed: int,
    max_frames: int | None,
    output_policy: dict[str, Any],
    device: str,
    logger: logging.Logger,
) -> Path:
    vname = variant["name"]
    mask_dilation = int(variant["mask_dilation"])
    denoise_strength = float(variant["denoise_strength"])
    keyframe_interval = int(variant.get("keyframe_interval", 1))
    variant_exp_id = f"{exp_id}__{vname}"
    pred_root = REPO_ROOT / "outputs" / "videos" / variant_exp_id
    pred_root.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Loading diffusion pipeline on %s", vname, device)
    pipe = load_pipeline(device)

    cfg_yaml = read_yaml(REPO_ROOT / "configs" / "base.yaml")
    fps = float((cfg_yaml.get("preprocess", {}) or {}).get("target_fps", 24))

    for ds in datasets:
        logger.info("[%s] dataset=%s", vname, ds)
        ds_cfg = ds_cfgs[ds]

        # Use B-best output from Phase 2 as diffusion input base.
        source_ds_root = REPO_ROOT / "outputs" / "videos" / source_exp_id / ds
        source_frame_dir = source_ds_root / "frames"
        if source_frame_dir.exists() and list_images(source_frame_dir):
            frame_paths = list_images(source_frame_dir)
            if max_frames:
                frame_paths = frame_paths[:max_frames]
            frame_names = [p.name for p in frame_paths]
            frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in frame_paths]
        else:
            # Fallback: decode from restored video
            restored_video = source_ds_root / "restored_h264.mp4"
            if not restored_video.exists():
                raise RuntimeError(f"Phase 2 restored video not found: {restored_video}")
            frames = decode_video_frames(restored_video, as_gray=False)
            if not frames:
                raise RuntimeError(f"No frames decoded from {restored_video}")
            if max_frames:
                frames = frames[:max_frames]
            # Generate synthetic frame names matching original count
            raw_frame_dir = REPO_ROOT / ds_cfg.get("processed_frames_dir", "")
            raw_paths = list_images(raw_frame_dir)
            frame_names = [p.name for p in raw_paths[:len(frames)]]
            if len(frame_names) < len(frames):
                frame_names += [f"{i:05d}.png" for i in range(len(frame_names), len(frames))]

        mask_video = source_ds_root / "mask_h264.mp4"
        raw_masks = decode_video_frames(mask_video, as_gray=True)
        if not raw_masks:
            raise RuntimeError(f"No masks from {mask_video}")
        if max_frames:
            raw_masks = raw_masks[:max_frames]
        h, w = frames[0].shape[:2]
        masks_u8 = []
        for m in raw_masks:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            masks_u8.append(((m > 127).astype(np.uint8) * 255))
        while len(masks_u8) < len(frames):
            masks_u8.append(masks_u8[-1].copy())
        masks_u8 = masks_u8[: len(frames)]

        n = len(frames)
        if keyframe_interval <= 1:
            # Process every frame
            restored = []
            for i, (f, m) in enumerate(zip(frames, masks_u8)):
                dilated = dilate_mask(m, mask_dilation)
                if (dilated > 0).any():
                    r = diffusion_inpaint_frame(pipe, f, dilated, denoise_strength, seed + i)
                else:
                    r = f.copy()
                restored.append(r)
                if (i + 1) % 10 == 0:
                    logger.info("[%s] %s %d/%d frames", vname, ds, i + 1, n)
        else:
            # Keyframe strategy: run diffusion on keyframes, propagate to others
            keyframe_indices = list(range(0, n, keyframe_interval))
            keyframe_results: dict[int, np.ndarray] = {}
            for idx in keyframe_indices:
                f, m = frames[idx], masks_u8[idx]
                dilated = dilate_mask(m, mask_dilation)
                if (dilated > 0).any():
                    keyframe_results[idx] = diffusion_inpaint_frame(pipe, f, dilated, denoise_strength, seed + idx)
                else:
                    keyframe_results[idx] = f.copy()
            logger.info("[%s] %s keyframes done (%d)", vname, ds, len(keyframe_indices))
            restored = propagate_keyframes(frames, masks_u8, keyframe_results)

        write_dataset_outputs(
            out_root=pred_root,
            dataset_name=ds,
            frame_names=frame_names,
            restored_frames=restored,
            masks_u8=masks_u8,
            target_fps=fps,
            save_mp4=False,
            output_policy=output_policy,
        )
        logger.info("[%s] %s done", vname, ds)

    del pipe
    import torch
    if device == "cuda":
        torch.cuda.empty_cache()

    return pred_root


def metric_or_pos_inf(agg: dict[str, Any], key: str) -> float:
    v = agg.get(key)
    if v is None:
        return float("inf")
    fv = float(v)
    if math.isnan(fv):
        return float("inf")
    return fv


def resolve_comparison_reference(
    metrics_root: Path,
    phase2_exp_id: str,
) -> tuple[str, str, dict[str, Any]]:
    phase2_summary = metrics_root / phase2_exp_id / "summary.json"
    if not phase2_summary.exists():
        raise RuntimeError(
            f"Phase 2 summary is missing for --phase2-exp-id={phase2_exp_id}: {phase2_summary}"
        )
    phase2_agg = read_json(phase2_summary).get("aggregate", {}) or {}
    return "B-best", phase2_exp_id, phase2_agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 Route G: GPU diffusion inpainting.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--exp-id", default=f"phase5_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--datasets", default="mandatory")
    parser.add_argument("--phase2-exp-id", default=PHASE2_REF)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--keep-variant-videos",
        action="store_true",
        help="Keep intermediate per-variant videos (<exp_id>__G-*) after evaluation.",
    )
    args = parser.parse_args()

    config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    cfg = read_yaml(config_path)
    logger = setup_logger(args.exp_id)
    logger.info("Phase 5 exp_id=%s device=%s", args.exp_id, args.device)

    ds_cfgs, all_names, mandatory_names = collect_dataset_cfg(cfg)
    datasets = resolve_dataset_names(args.datasets, all_names, mandatory_names)
    output_policy = resolve_output_policy(cfg)
    gt_root = REPO_ROOT / cfg.get("paths", {}).get("gt_data_dir", "data/gt")

    variant_results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        vname = variant["name"]
        variant_exp_id = f"{args.exp_id}__{vname}"
        variant_pred_root = REPO_ROOT / "outputs" / "videos" / variant_exp_id
        logger.info("=== Variant %s ===", vname)
        try:
            pred_root = run_variant(
                variant=variant,
                datasets=datasets,
                ds_cfgs=ds_cfgs,
                source_exp_id=args.phase2_exp_id,
                exp_id=args.exp_id,
                seed=args.seed,
                max_frames=args.max_frames,
                output_policy=output_policy,
                device=args.device,
                logger=logger,
            )
            _, summary = run_evaluation(
                config_path=config_path,
                datasets=datasets,
                pred_root=pred_root,
                gt_root=gt_root,
                eval_exp_id=variant_exp_id,
                allow_missing_gt=True,
                save_visualization=False,
                logger=logger,
            )
            cleanup_video_only_outputs(
                exp_pred_root=pred_root,
                datasets=datasets,
                output_policy=output_policy,
            )
            agg = summary.get("aggregate", {}) or {}
            tcf = metric_or_pos_inf(agg, "TCF")
            variant_results.append({"name": vname, "tcf": tcf, "aggregate": agg, "pred_root": str(pred_root)})
            logger.info("Variant %s TCF=%.4f", vname, tcf)
            if not args.keep_variant_videos and pred_root.exists():
                shutil.rmtree(pred_root)
                logger.info("[%s] cleaned intermediate variant videos: %s", vname, pred_root)
        except Exception as e:
            logger.error("Variant %s failed: %s", vname, e, exc_info=True)
            variant_results.append({"name": vname, "tcf": float("inf"), "aggregate": {}, "pred_root": "", "error": str(e)})
            if not args.keep_variant_videos and variant_pred_root.exists():
                shutil.rmtree(variant_pred_root)
                logger.info("[%s] cleaned failed intermediate variant videos: %s", vname, variant_pred_root)

    successful_variants = [vr for vr in variant_results if math.isfinite(float(vr.get("tcf", float("inf"))))]
    if not successful_variants:
        error_lines = []
        for vr in variant_results:
            error_lines.append(f"{vr.get('name')}: {vr.get('error', 'unknown error')}")
        raise RuntimeError("All Phase 5 variants failed:\n" + "\n".join(error_lines))

    # Select best by TCF (lower is better)
    best = min(successful_variants, key=lambda x: float(x["tcf"]))
    best_name = best["name"]
    logger.info("Best variant: %s (TCF=%.4f)", best_name, best["tcf"])

    # Re-run best variant as the canonical exp_id output
    best_variant = next(v for v in VARIANTS if v["name"] == best_name)
    final_pred_root = run_variant(
        variant=best_variant,
        datasets=datasets,
        ds_cfgs=ds_cfgs,
        source_exp_id=args.phase2_exp_id,
        exp_id=args.exp_id,
        seed=args.seed,
        max_frames=args.max_frames,
        output_policy=output_policy,
        device=args.device,
        logger=logger,
    )
    final_root = REPO_ROOT / "outputs" / "videos" / args.exp_id
    if final_root.exists() and final_root != final_pred_root:
        shutil.rmtree(final_root)
    if final_pred_root != final_root:
        final_pred_root.rename(final_root)

    _, final_summary = run_evaluation(
        config_path=config_path,
        datasets=datasets,
        pred_root=final_root,
        gt_root=gt_root,
        eval_exp_id=args.exp_id,
        allow_missing_gt=True,
        save_visualization=False,
        logger=logger,
    )
    cleanup_video_only_outputs(
        exp_pred_root=final_root,
        datasets=datasets,
        output_policy=output_policy,
    )

    metrics_dir = REPO_ROOT / "outputs" / "metrics" / args.exp_id
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # phase5_ablation.csv
    with (metrics_dir / "phase5_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "mask_dilation", "denoise_strength", "keyframe_interval", "TCF", "JM", "JR", "ROS", "BES"])
        w.writeheader()
        for vr in variant_results:
            vspec = next((v for v in VARIANTS if v["name"] == vr["name"]), {})
            agg = vr.get("aggregate", {}) or {}
            w.writerow({
                "variant": vr["name"],
                "mask_dilation": vspec.get("mask_dilation", ""),
                "denoise_strength": vspec.get("denoise_strength", ""),
                "keyframe_interval": vspec.get("keyframe_interval", ""),
                "TCF": agg.get("TCF", ""),
                "JM": agg.get("JM", ""),
                "JR": agg.get("JR", ""),
                "ROS": agg.get("ROS", ""),
                "BES": agg.get("BES", ""),
            })

    write_json(metrics_dir / "phase5_selection.json", {
        "selected_variant": best_name,
        "tcf": best["tcf"],
        "all_variants": [{"name": vr["name"], "tcf": vr["tcf"]} for vr in variant_results],
    })

    # phase5_b_vs_g.csv
    metrics_root = REPO_ROOT / "outputs" / "metrics"
    ref_label, ref_exp_id, ref_agg = resolve_comparison_reference(
        metrics_root=metrics_root,
        phase2_exp_id=args.phase2_exp_id,
    )
    g_agg = final_summary.get("aggregate", {}) or {}
    with (metrics_dir / "phase5_b_vs_g.csv").open("w", newline="", encoding="utf-8") as f:
        w2 = csv.DictWriter(f, fieldnames=["method", "JM", "JR", "ROS", "TCF", "BES"])
        w2.writeheader()
        w2.writerow({"method": ref_label, **{k: ref_agg.get(k, "") for k in ["JM", "JR", "ROS", "TCF", "BES"]}})
        w2.writerow({"method": f"G-best ({best_name})", **{k: g_agg.get(k, "") for k in ["JM", "JR", "ROS", "TCF", "BES"]}})

    # per_dataset.csv
    eval_per_ds = metrics_dir / "per_dataset.csv"
    if not eval_per_ds.exists():
        ds_data = final_summary.get("datasets", {}) or {}
        with eval_per_ds.open("w", newline="", encoding="utf-8") as f:
            w3 = csv.DictWriter(f, fieldnames=["dataset", "JM", "JR", "ROS", "TCF", "BES"])
            w3.writeheader()
            for ds_name, ds_info in ds_data.items():
                m = (ds_info.get("metrics", {}) or {})
                w3.writerow({"dataset": ds_name, **{k: m.get(k, "") for k in ["JM", "JR", "ROS", "TCF", "BES"]}})

    write_json(metrics_dir / "phase5_run_meta.json", {
        "exp_id": args.exp_id,
        "phase2_ref": args.phase2_exp_id,
        "comparison_ref": {
            "label": ref_label,
            "exp_id": ref_exp_id,
        },
        "g_final_variant": best_name,
        "has_g_variants": True,
        "seed": args.seed,
        "device": args.device,
        "model_id": SD_MODEL_ID,
        "variants": [v["name"] for v in VARIANTS],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    # phase5_acceptance_report.md
    lines = [
        "# Phase 5 (Route G) Acceptance Report",
        "",
        f"**exp_id**: {args.exp_id}",
        f"**phase2_ref**: {args.phase2_exp_id}",
        f"**selected_variant**: {best_name}",
        f"**model**: {SD_MODEL_ID}",
        f"**device**: {args.device}",
        f"**generated_at**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Method",
        "",
        f"GPU-based Stable Diffusion Inpainting (SD 1.5). Input base/masks from {args.phase2_exp_id}.",
        "Four variants (G-low/mid/high/hybrid) ablate mask_dilation, denoise_strength, and keyframe_interval.",
        "Best variant selected by TCF (lower is better).",
        "",
        "## Ablation",
        "",
        "| Variant | mask_dilation | denoise_strength | keyframe_interval | TCF |",
        "|---------|--------------|-----------------|-------------------|-----|",
    ]
    for vr in variant_results:
        vspec = next((v for v in VARIANTS if v["name"] == vr["name"]), {})
        agg = vr.get("aggregate", {}) or {}
        lines.append(f"| {vr['name']} | {vspec.get('mask_dilation','')} | {vspec.get('denoise_strength','')} | {vspec.get('keyframe_interval','')} | {agg.get('TCF', 'N/A')} |")

    lines += [
        "",
        f"## {ref_label} vs G-best",
        "",
        "| Method | JM | JR | ROS | TCF | BES |",
        "|--------|----|----|-----|-----|-----|",
        f"| {ref_label} | {ref_agg.get('JM','')} | {ref_agg.get('JR','')} | {ref_agg.get('ROS','')} | {ref_agg.get('TCF','')} | {ref_agg.get('BES','')} |",
        f"| G-best ({best_name}) | {g_agg.get('JM','')} | {g_agg.get('JR','')} | {g_agg.get('ROS','')} | {g_agg.get('TCF','')} | {g_agg.get('BES','')} |",
        "",
        "## Failure Analysis",
        "",
        "- Style drift: diffusion may generate textures inconsistent with surrounding background.",
        "- Temporal flicker: per-frame diffusion without optical flow warping causes frame-to-frame inconsistency.",
        "- Structural hallucination: large mask regions may produce plausible but incorrect background structures.",
        "- Boundary seam: mask boundary artifacts where diffusion output meets original frame.",
    ]
    (metrics_dir / "phase5_acceptance_report.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("Phase 5 complete. exp_id=%s", args.exp_id)
    logger.info("G-best=%s TCF=%.4f", best_name, metric_or_pos_inf(g_agg, "TCF"))


if __name__ == "__main__":
    main()
