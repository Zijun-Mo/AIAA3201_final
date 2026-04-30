#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def list_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def find_mask_for_name(mask_dir: Path, frame_name: str) -> Path | None:
    direct = mask_dir / frame_name
    if direct.exists():
        return direct
    stem = Path(frame_name).stem
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def ensure_binary_mask(mask: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((mask > 0).astype(np.uint8) * 255)


def load_masks_by_frame_names(mask_dir: Path, frame_names: list[str], shape_hw: tuple[int, int]) -> list[np.ndarray]:
    h, w = shape_hw
    out: list[np.ndarray] = []
    for name in frame_names:
        mp = find_mask_for_name(mask_dir=mask_dir, frame_name=name)
        if mp is None:
            out.append(np.zeros((h, w), dtype=np.uint8))
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            out.append(np.zeros((h, w), dtype=np.uint8))
            continue
        out.append(ensure_binary_mask(m, (h, w)))
    return out


def build_prompt_anchors(
    masks_u8: list[np.ndarray],
    shape_hw: tuple[int, int],
    max_anchors: int = 3,
    max_prompts_per_anchor: int = 3,
    min_anchor_gap_ratio: float = 0.16,
    min_area_ratio: float = 1e-6,
) -> list[dict]:
    """Mirror of Phase 2 build_prior_prompt_anchors_from_masks: multi-anchor, multi-box, gap-spaced."""
    h, w = shape_hw
    frame_count = len(masks_u8)
    min_area_px = max(1, int(min_area_ratio * h * w))
    min_gap = int(round(min_anchor_gap_ratio * max(1, frame_count - 1)))

    candidates = []
    for idx, mask in enumerate(masks_u8):
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        comps = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area_px:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            bw = int(stats[label, cv2.CC_STAT_WIDTH])
            bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            comps.append((area, (x, y, x + bw - 1, y + bh - 1)))
        if comps:
            candidates.append({
                "frame_idx": idx,
                "total_area": float(binary.sum()),
                "components": sorted(comps, key=lambda c: c[0], reverse=True),
            })

    selected = []
    for cand in sorted(candidates, key=lambda c: c["total_area"], reverse=True):
        if min_gap > 0 and any(abs(cand["frame_idx"] - s["frame_idx"]) < min_gap for s in selected):
            continue
        boxes = [box for _, box in cand["components"][:max_prompts_per_anchor]]
        if boxes:
            selected.append({"frame_idx": cand["frame_idx"], "boxes": boxes})
        if len(selected) >= max_anchors:
            break

    if not selected:
        raise RuntimeError("Input masks are all zero; cannot build SAM3 box prompts")
    return selected


def materialize_video_dir(frames_dir: Path) -> tuple[Path, list[str]]:
    frame_paths = list_images(frames_dir)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found under: {frames_dir}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="phase3_sam3_frames_"))
    for idx, src in enumerate(frame_paths):
        dst = tmp_dir / f"{idx:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
    frame_names = [p.name for p in frame_paths]
    return tmp_dir, frame_names


def resolve_sam3_builder() -> tuple[Any, str]:
    candidates = [
        ("sam3.model_builder", "build_sam3_video_predictor"),
        ("sam3.build_sam", "build_sam3_video_predictor"),
        ("sam3.build_sam3", "build_sam3_video_predictor"),
        ("sam3.build_sam", "build_sam_video_predictor"),
        ("sam3.build_sam3", "build_sam_video_predictor"),
    ]
    errors: list[str] = []
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, attr, None)
            if fn is None:
                errors.append(f"{mod_name}.{attr}:missing_attr")
                continue
            return fn, f"{mod_name}.{attr}"
        except Exception as e:
            errors.append(f"{mod_name}.{attr}:{type(e).__name__}:{e}")
            continue
    raise RuntimeError("Unable to import SAM3 video predictor builder: " + " | ".join(errors))


def build_predictor(
    builder: Any,
    model_cfg: str,
    checkpoint: str | None,
    device: str,
) -> Any:
    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    # Official SAM3 predictor takes checkpoint_path and builds GPU model internally.
    if checkpoint:
        attempts.extend(
            [
                ((), {"checkpoint_path": checkpoint}),
                ((), {"checkpoint": checkpoint}),
                ((checkpoint,), {}),
            ]
        )
    attempts.extend(
        [
            ((), {}),
            ((), {"load_from_HF": True}),
        ]
    )

    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            return builder(*args, **kwargs)
        except Exception as e:
            last_error = e
            continue

    if last_error is None:
        raise RuntimeError("No valid predictor build attempt")
    raise last_error


def to_numpy_mask_logits(mask_logits: Any, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    arr = mask_logits
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr_np = np.asarray(arr)

    frame_mask = np.zeros((h, w), dtype=np.uint8)
    if arr_np.ndim == 2:
        return (arr_np > 0).astype(np.uint8) * 255

    for m in arr_np:
        m2 = m[0] if getattr(m, "ndim", 0) == 3 else m
        frame_mask = np.maximum(frame_mask, (np.asarray(m2) > 0).astype(np.uint8) * 255)
    return frame_mask


def run_sam3_refine(
    repo_path: Path,
    input_frames: Path,
    input_masks: Path,
    output_masks: Path,
    checkpoint: str | None,
    model_cfg: str,
    prompt_text: str,
    device: str,
) -> dict[str, Any]:
    if not repo_path.exists():
        raise FileNotFoundError(f"SAM3 repo path not found: {repo_path}")

    sys.path.insert(0, str(repo_path))

    video_dir, frame_names = materialize_video_dir(input_frames)
    try:
        first = cv2.imread(str((input_frames / frame_names[0]).resolve()), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Unable to read first frame: {input_frames / frame_names[0]}")
        h, w = first.shape[:2]
        masks_u8 = load_masks_by_frame_names(mask_dir=input_masks, frame_names=frame_names, shape_hw=(h, w))
        anchors = build_prompt_anchors(masks_u8=masks_u8, shape_hw=(h, w))

        builder, builder_name = resolve_sam3_builder()
        predictor = build_predictor(builder=builder, model_cfg=model_cfg, checkpoint=checkpoint, device=device)

        pred_masks = [np.zeros((h, w), dtype=np.uint8) for _ in frame_names]

        def absorb_stream(stream: Any) -> None:
            for response in stream:
                if not isinstance(response, dict):
                    continue
                frame_idx = int(response.get("frame_index", -1))
                if frame_idx < 0 or frame_idx >= len(pred_masks):
                    continue
                out = response.get("outputs", {}) or {}
                masks_raw = out.get("out_binary_masks", None)
                if masks_raw is None:
                    continue
                arr = np.asarray(masks_raw)
                if arr.ndim == 2:
                    arr = arr[None, ...]
                if arr.ndim == 4:
                    arr = arr[:, 0, :, :]
                frame_mask = np.zeros((h, w), dtype=np.uint8)
                for item in arr:
                    m = np.asarray(item)
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    frame_mask = np.maximum(frame_mask, ((m > 0).astype(np.uint8) * 255))
                pred_masks[frame_idx] = np.maximum(pred_masks[frame_idx], frame_mask)

        # bidirectional_no_wrap: each anchor propagates forward then backward independently
        for anchor in anchors:
            anchor_frame_idx = int(anchor["frame_idx"])
            boxes_xywh_rel = []
            for x1, y1, x2, y2 in anchor["boxes"]:
                bw = max(1, x2 - x1 + 1)
                bh = max(1, y2 - y1 + 1)
                boxes_xywh_rel.append([
                    float(x1) / float(max(1, w)),
                    float(y1) / float(max(1, h)),
                    float(bw) / float(max(1, w)),
                    float(bh) / float(max(1, h)),
                ])

            start = predictor.handle_request({"type": "start_session", "resource_path": str(video_dir)})
            session_id = str(start.get("session_id", ""))
            if not session_id:
                raise RuntimeError("SAM3 start_session did not return session_id")
            try:
                for box_rel in boxes_xywh_rel:
                    add_req: dict[str, Any] = {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": anchor_frame_idx,
                        "bounding_boxes": [box_rel],
                        "bounding_box_labels": [1],
                    }
                    if prompt_text.strip():
                        add_req["text"] = prompt_text.strip()
                    predictor.handle_request(request=add_req)

                # forward pass
                absorb_stream(predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": anchor_frame_idx,
                }))
                # backward pass (no_wrap: only if anchor is not first frame)
                if anchor_frame_idx > 0:
                    absorb_stream(predictor.handle_stream_request({
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "backward",
                        "start_frame_index": anchor_frame_idx,
                    }))
            finally:
                try:
                    predictor.handle_request({"type": "close_session", "session_id": session_id})
                except Exception:
                    pass

        mean_ratio = float(np.mean(np.asarray([(m > 0).mean() for m in pred_masks], dtype=np.float32)))
        if mean_ratio <= 0.0:
            raise RuntimeError("SAM3 output masks are all zero")

        output_masks.mkdir(parents=True, exist_ok=True)
        for name, mask in zip(frame_names, pred_masks):
            cv2.imwrite(str(output_masks / name), ensure_binary_mask(mask, (h, w)))

        return {
            "status": "ok",
            "builder": builder_name,
            "num_anchors": len(anchors),
            "anchor_frame_indices": [a["frame_idx"] for a in anchors],
            "mean_mask_ratio": mean_ratio,
        }
    finally:
        shutil.rmtree(video_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 refinement worker (local official repo).")
    parser.add_argument("--repo-path", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True, type=str)
    parser.add_argument("--input-frames", required=True, type=Path)
    parser.add_argument("--input-masks", required=True, type=Path)
    parser.add_argument("--output-masks", required=True, type=Path)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--model-cfg", default="", type=str)
    parser.add_argument("--prompt-text", default="", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.strip() or None
    model_cfg = args.model_cfg.strip()

    meta = run_sam3_refine(
        repo_path=args.repo_path,
        input_frames=args.input_frames,
        input_masks=args.input_masks,
        output_masks=args.output_masks,
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        prompt_text=args.prompt_text,
        device=args.device,
    )
    print(json.dumps({"dataset": args.dataset_name, **meta}, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[sam3_refine_worker] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
