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


def pick_anchor_boxes(masks_u8: list[np.ndarray], max_boxes: int = 1) -> tuple[int, list[tuple[int, int, int, int]]]:
    if not masks_u8:
        raise RuntimeError("No masks provided for SAM3 prompt selection")

    areas = [int((m > 0).sum()) for m in masks_u8]
    frame_idx = int(np.argmax(np.asarray(areas, dtype=np.int64)))
    if areas[frame_idx] <= 0:
        raise RuntimeError("Input masks are all zero; cannot build SAM3 box prompt")

    mask = (masks_u8[frame_idx] > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError("Failed to derive prompt box from input masks")

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    boxes: list[tuple[int, int, int, int]] = [(x1, y1, x2, y2)]

    if not boxes:
        raise RuntimeError("Failed to derive prompt boxes from input masks")
    return frame_idx, boxes


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
        prompt_frame_idx, prompt_boxes = pick_anchor_boxes(masks_u8=masks_u8, max_boxes=1)
        boxes_xywh_rel: list[list[float]] = []
        for x1, y1, x2, y2 in prompt_boxes:
            bw = max(1, x2 - x1 + 1)
            bh = max(1, y2 - y1 + 1)
            boxes_xywh_rel.append(
                [
                    float(x1) / float(max(1, w)),
                    float(y1) / float(max(1, h)),
                    float(bw) / float(max(1, w)),
                    float(bh) / float(max(1, h)),
                ]
            )

        builder, builder_name = resolve_sam3_builder()
        predictor = build_predictor(builder=builder, model_cfg=model_cfg, checkpoint=checkpoint, device=device)

        start = predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": str(video_dir),
            }
        )
        session_id = str(start.get("session_id", ""))
        if not session_id:
            raise RuntimeError("SAM3 start_session did not return session_id")

        try:
            add_request: dict[str, Any] = {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": int(prompt_frame_idx),
                # SAM3 video API currently expects exactly one visual prompt box per initial add_prompt.
                "bounding_boxes": [boxes_xywh_rel[0]],
                "bounding_box_labels": [1],
            }
            if prompt_text.strip():
                add_request["text"] = prompt_text.strip()
            add_resp = predictor.handle_request(request=add_request)
            _ = add_resp

            pred_masks = [np.zeros((h, w), dtype=np.uint8) for _ in frame_names]
            for response in predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "both",
                    "start_frame_index": int(prompt_frame_idx),
                }
            ):
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
                pred_masks[frame_idx] = frame_mask
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
            "prompt_frame_idx": int(prompt_frame_idx),
            "num_prompt_boxes": int(len(prompt_boxes)),
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
