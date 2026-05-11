#!/usr/bin/env python3
"""Build final paper-ready result tables and figures from accepted runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GT_COVERAGE_KEY = "GT_Coverage"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    exp_id: str
    mask_source: str
    inpainting: str
    note: str = ""


DEFAULT_METHODS = [
    MethodSpec("A", "A-best", "phase1_maskscore_fastvqa_20260510_023457_pl220", "Mask R-CNN + Flow", "OpenCV Telea + temporal"),
    MethodSpec("B", "B-best", "phase2_maskscore_fastvqa_20260510_023457_pl220", "Track Anything coarse", "ProPainter"),
    MethodSpec("E", "B+E", "phase3_maskscore_fastvqa_20260510_023457_pl220", "SAM3 refined", "ProPainter"),
    MethodSpec("F", "B+F", "phase4_maskscore_fastvqa_altbackend_20260510_pl220", "VGGT4D prior + SAM2", "ProPainter"),
    MethodSpec("P6", "Phase6-best", "phase6_core_maskscore_fastvqa_20260511_235610_pl220", "F-best prior + SAM3", "ProPainter"),
]
G_SPEC = MethodSpec("G", "Phase5/G-hybrid", "phase5_maskscore_fastvqa_20260510_023457_pl220", "B mask", "ProPainter + diffusion", "qualitative only")

DATASETS = ["wild", "bmx-trees", "tennis"]
FRAME_PICK = {"wild": 120, "bmx-trees": 40, "tennis": 35}
BOUNDARY_FRAME_PICK = {"bmx-trees": 40, "tennis": 35}
FAILURE_FRAME_PICK = {"wild": [118, 120, 122], "bmx-trees": [38, 40, 42], "tennis": [33, 35, 37]}

COLORS = {
    "A-best": (87, 132, 193),
    "B-best": (91, 160, 93),
    "B+E": (237, 125, 49),
    "B+F": (166, 118, 29),
    "Phase6-best": (192, 80, 77),
    "Phase5/G-hybrid": (128, 100, 162),
}


def build_method_specs(args: argparse.Namespace) -> tuple[list[MethodSpec], MethodSpec]:
    methods = [
        MethodSpec("A", "A-best", args.phase1_exp_id, "Mask R-CNN + Flow", "OpenCV Telea + temporal"),
        MethodSpec("B", "B-best", args.phase2_exp_id, "Track Anything coarse", "ProPainter"),
        MethodSpec("E", "B+E", args.phase3_exp_id, "SAM3 refined", "ProPainter"),
        MethodSpec("F", "B+F", args.phase4_exp_id, "VGGT4D prior + alternate backend", "ProPainter"),
        MethodSpec("P6", "Phase6-best", args.phase6_exp_id, "F-best prior + SAM3", "ProPainter"),
    ]
    g_spec = MethodSpec("G", "Phase5/G-hybrid", args.phase5_exp_id, "B mask", "ProPainter + diffusion", "qualitative only")
    return methods, g_spec


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def markdown_table(fields: list[str], rows: list[dict[str, Any]]) -> str:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out) + "\n"


def latex_escape(text: Any) -> str:
    token = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        token = token.replace(a, b)
    return token


def latex_table(fields: list[str], rows: list[dict[str, Any]]) -> str:
    spec = "l" + "r" * max(0, len(fields) - 1)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(f) for f in fields) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(f, "")) for f in fields) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_table_bundle(out_dir: Path, stem: str, fields: list[str], rows: list[dict[str, Any]], title: str) -> dict[str, str]:
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    tex_path = out_dir / f"{stem}.tex"
    write_csv(csv_path, rows, fields)
    md_path.write_text(f"# {title}\n\n" + markdown_table(fields, rows), encoding="utf-8")
    tex_path.write_text(latex_table(fields, rows), encoding="utf-8")
    return {"csv": rel(csv_path), "md": rel(md_path), "tex": rel(tex_path)}


def metric_from_summary(exp_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = REPO_ROOT / "outputs" / "metrics" / exp_id / "summary.json"
    summary = read_json(summary_path)
    return summary, summary.get("aggregate", {}) or {}


def mask_score(row: dict[str, Any]) -> float | None:
    gt_cov = row.get(GT_COVERAGE_KEY)
    jm = row.get("JM")
    jr = row.get("JR")
    if jm is None or jr is None:
        return None
    try:
        if gt_cov is None or gt_cov == "":
            return (float(jm) + float(jr)) / 2.0
        return 0.5 * float(gt_cov) + 0.25 * float(jm) + 0.25 * float(jr)
    except (TypeError, ValueError):
        return None


def delta_fmt(current: Any, base: Any) -> str:
    try:
        cur = float(current)
        ref = float(base)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(cur) or not math.isfinite(ref):
        return ""
    return fmt(cur - ref)


def build_main_table(out_dir: Path, methods: list[MethodSpec]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    base_agg: dict[str, Any] | None = None
    for spec in methods:
        _, agg = metric_from_summary(spec.exp_id)
        if spec.key == "B":
            base_agg = agg
        score = mask_score(agg)
        rows.append(
            {
                "Method": spec.label,
                "Mask Source": spec.mask_source,
                "Inpainting": spec.inpainting,
                GT_COVERAGE_KEY: fmt(agg.get(GT_COVERAGE_KEY)),
                "JM": fmt(agg.get("JM")),
                "JR": fmt(agg.get("JR")),
                "MaskScore": fmt(score),
                "TCF": fmt(agg.get("TCF")),
                "FAST_VQA": fmt(agg.get("FAST_VQA")),
                "Exp ID": spec.exp_id,
                "Note": spec.note,
            }
        )
    if base_agg:
        for row, spec in zip(rows, methods):
            _, agg = metric_from_summary(spec.exp_id)
            row["Delta GT_Coverage vs B"] = delta_fmt(agg.get(GT_COVERAGE_KEY), base_agg.get(GT_COVERAGE_KEY))
            row["Delta JM vs B"] = delta_fmt(agg.get("JM"), base_agg.get("JM"))
            row["Delta JR vs B"] = delta_fmt(agg.get("JR"), base_agg.get("JR"))
            row["Delta TCF vs B"] = delta_fmt(agg.get("TCF"), base_agg.get("TCF"))

    fields = [
        "Method",
        "Mask Source",
        "Inpainting",
        GT_COVERAGE_KEY,
        "JM",
        "JR",
        "MaskScore",
        "TCF",
        "FAST_VQA",
        "Delta GT_Coverage vs B",
        "Delta JM vs B",
        "Delta JR vs B",
        "Delta TCF vs B",
        "Exp ID",
        "Note",
    ]
    paths = write_table_bundle(out_dir, "table1_main_performance", fields, rows, "Table 1: Main Performance")
    return rows, paths


def build_a_ablation(out_dir: Path, phase1_exp_id: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    path = REPO_ROOT / "outputs" / "metrics" / phase1_exp_id / "phase1_ablation.csv"
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        rows.append(
            {
                "Setting": f"{row.get('stage')}/{row.get('candidate')}",
                "Seg Model": row.get("seg_model", ""),
                "Flow Thr.": row.get("flow_threshold", ""),
                "Dilation": row.get("dilation_kernel", ""),
                "Inpaint": row.get("inpaint_method", ""),
                "Temporal W": row.get("temporal_window", ""),
                GT_COVERAGE_KEY: fmt(row.get(GT_COVERAGE_KEY)),
                "JM": fmt(row.get("JM")),
                "JR": fmt(row.get("JR")),
                "TCF": fmt(row.get("TCF")),
                "FAST_VQA": fmt(row.get("FAST_VQA")),
                "Final": row.get("is_final_best", ""),
            }
        )
    fields = [
        "Setting",
        "Seg Model",
        "Flow Thr.",
        "Dilation",
        "Inpaint",
        "Temporal W",
        GT_COVERAGE_KEY,
        "JM",
        "JR",
        "TCF",
        "FAST_VQA",
        "Final",
    ]
    paths = write_table_bundle(out_dir, "table2_a_ablation", fields, rows, "Table 2: Route A Ablation")
    return rows, paths


def load_ablation_rows(exp_id: str, filename: str, keep: Iterable[tuple[str, str]] | None = None) -> list[dict[str, str]]:
    path = REPO_ROOT / "outputs" / "metrics" / exp_id / filename
    rows = read_csv(path)
    if keep is None:
        return rows
    keep_set = set(keep)
    return [row for row in rows if (row.get("stage", ""), row.get("candidate", "")) in keep_set]


def build_extension_ablation(
    out_dir: Path,
    *,
    phase2_exp_id: str,
    phase3_exp_id: str,
    phase4_exp_id: str,
    phase6_exp_id: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    specs: list[tuple[str, str, dict[str, str]]] = []
    specs.extend(
        [
            ("B-best", phase2_exp_id, row)
            for row in load_ablation_rows(
                phase2_exp_id,
                "phase2_ablation.csv",
                keep=[("B5", "b_best_finalize"), ("B3", "refined_mask"), ("B4", "balanced_b")],
            )
        ]
    )
    specs.extend(
        [
            ("E-route", phase3_exp_id, row)
            for row in load_ablation_rows(
                phase3_exp_id,
                "phase3_ablation.csv",
                keep=[("E1", "morph_identity"), ("E2", "temporal_w2"), ("E3", "sam3_refine"), ("E4", "b_plus_e_finalize")],
            )
        ]
    )
    specs.extend(
        [
            ("F-route", phase4_exp_id, row)
            for row in load_ablation_rows(
                phase4_exp_id,
                "phase4_ablation.csv",
                keep=[
                    ("F1", "bbest_vggt4d_replace_yolo"),
                    ("F3", "intersection"),
                    ("F3", "union"),
                    ("F3", "vggt4d_guided_auto"),
                    ("F4", "bidir_e8.0_strict"),
                ],
            )
        ]
    )
    specs.extend(
        [
            ("Phase6", phase6_exp_id, row)
            for row in load_ablation_rows(phase6_exp_id, "phase6_ablation.csv")
        ]
    )

    rows: list[dict[str, Any]] = []
    for group, exp_id, row in specs:
        score = mask_score(row)
        motion = row.get("f_source_key") or row.get("f_fusion_method") or row.get("mask_backend") or row.get("source_stage") or "none"
        refinement = "SAM3" if str(row.get("use_sam3", "0")) in {"1", "true", "True"} else row.get("mask_variant", "")
        if row.get("f_fusion_method"):
            refinement = (refinement + "+" if refinement else "") + row.get("f_fusion_method", "")
        rows.append(
            {
                "Group": group,
                "Setting": f"{row.get('stage')}/{row.get('candidate')}",
                "Motion/Prior": motion,
                "Refinement": refinement or "none",
                GT_COVERAGE_KEY: fmt(row.get(GT_COVERAGE_KEY)),
                "JM": fmt(row.get("JM")),
                "JR": fmt(row.get("JR")),
                "MaskScore": fmt(score),
                "TCF": fmt(row.get("TCF")),
                "FAST_VQA": fmt(row.get("FAST_VQA")),
                "Final": row.get("is_final_best", ""),
                "Exp ID": exp_id,
            }
        )
    fields = [
        "Group",
        "Setting",
        "Motion/Prior",
        "Refinement",
        GT_COVERAGE_KEY,
        "JM",
        "JR",
        "MaskScore",
        "TCF",
        "FAST_VQA",
        "Final",
        "Exp ID",
    ]
    paths = write_table_bundle(out_dir, "table3_bef_phase6_ablation", fields, rows, "Table 3: B/E/F/Phase6 Ablation")
    return rows, paths


def build_phase5_failure_table(out_dir: Path, phase5_exp_id: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    path = REPO_ROOT / "outputs" / "metrics" / phase5_exp_id / "phase5_ablation.csv"
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        rows.append(
            {
                "Variant": row.get("variant", ""),
                "Dilation": row.get("mask_dilation", ""),
                "Denoise": row.get("denoise_strength", ""),
                "Keyframe": row.get("keyframe_interval", ""),
                GT_COVERAGE_KEY: fmt(row.get(GT_COVERAGE_KEY)),
                "JM": fmt(row.get("JM")),
                "JR": fmt(row.get("JR")),
                "TCF": fmt(row.get("TCF")),
                "FAST_VQA": fmt(row.get("FAST_VQA")),
                "Role": "qualitative failure analysis only",
            }
        )
    fields = [
        "Variant",
        "Dilation",
        "Denoise",
        "Keyframe",
        GT_COVERAGE_KEY,
        "JM",
        "JR",
        "TCF",
        "FAST_VQA",
        "Role",
    ]
    paths = write_table_bundle(out_dir, "table4_phase5_qualitative", fields, rows, "Table 4: Phase 5 Qualitative Failure Analysis")
    return rows, paths


def read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = max(0, min(int(frame_idx), max(0, total - 1))) if total > 0 else int(frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
    finally:
        cap.release()
    raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")


def read_input_frame(dataset: str, frame_idx: int) -> np.ndarray:
    frames = sorted((REPO_ROOT / "data" / "processed" / dataset / "frames").glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames for dataset={dataset}")
    idx = max(0, min(int(frame_idx), len(frames) - 1))
    img = cv2.imread(str(frames[idx]), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read input frame: {frames[idx]}")
    return img


def restored_video(spec: MethodSpec, dataset: str) -> Path:
    return REPO_ROOT / "outputs" / "videos" / spec.exp_id / dataset / "restored_h264.mp4"


def mask_video(spec: MethodSpec, dataset: str) -> Path:
    return REPO_ROOT / "outputs" / "videos" / spec.exp_id / dataset / "mask_h264.mp4"


def ensure_size(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def put_label(img: np.ndarray, label: str, sublabel: str | None = None) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    band_h = 30 if sublabel is None else 48
    cv2.rectangle(out, (0, 0), (w, band_h), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    if sublabel:
        cv2.putText(out, sublabel, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def hstack_tiles(tiles: list[np.ndarray], pad: int = 8) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    bg = np.full((h, pad, 3), 245, dtype=np.uint8)
    padded = []
    for tile in tiles:
        if tile.shape[0] < h:
            bottom = np.full((h - tile.shape[0], tile.shape[1], 3), 245, dtype=np.uint8)
            tile = np.vstack([tile, bottom])
        padded.append(tile)
    out = padded[0]
    for tile in padded[1:]:
        out = np.hstack([out, bg, tile])
    return out


def vstack_rows(rows: list[np.ndarray], pad: int = 10) -> np.ndarray:
    w = max(r.shape[1] for r in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < w:
            right = np.full((row.shape[0], w - row.shape[1], 3), 245, dtype=np.uint8)
            row = np.hstack([row, right])
        padded_rows.append(row)
    sep = np.full((pad, w, 3), 245, dtype=np.uint8)
    out = padded_rows[0]
    for row in padded_rows[1:]:
        out = np.vstack([out, sep, row])
    return out


def overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.42, color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = frame.copy()
    colored = np.zeros_like(out)
    colored[:, :] = color
    idx = mask_resized > 127
    if np.any(idx):
        blended = out[idx].astype(np.float32) * (1.0 - alpha) + colored[idx].astype(np.float32) * alpha
        out[idx] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def mask_as_binary(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = target_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 127


def bbox_from_binary(mask: np.ndarray, expand: float = 0.22) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    h, w = mask.shape[:2]
    if ys.size == 0 or xs.size == 0:
        cx, cy = w // 2, h // 2
        bw, bh = w // 3, h // 3
        return max(0, cx - bw // 2), max(0, cy - bh // 2), min(w, cx + bw // 2), min(h, cy + bh // 2)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    pad_x = int(max(24, bw * expand))
    pad_y = int(max(24, bh * expand))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(w, x1 + pad_x), min(h, y1 + pad_y)


def crop_box(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1].copy()


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def draw_metric_bar_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    w, h = 1500, 920
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Final Method Comparison", (42, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(img, "MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR; TCF lower is better", (42, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 80, 80), 1, cv2.LINE_AA)

    metrics = [GT_COVERAGE_KEY, "JM", "JR", "MaskScore"]
    x0, y0, plot_w, plot_h = 90, 145, 1320, 420
    cv2.rectangle(img, (x0, y0), (x0 + plot_w, y0 + plot_h), (230, 230, 230), 1)
    for i in range(6):
        y = y0 + plot_h - int(plot_h * i / 5)
        cv2.line(img, (x0, y), (x0 + plot_w, y), (238, 238, 238), 1)
        cv2.putText(img, f"{i/5:.1f}", (38, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)

    group_w = plot_w / len(rows)
    bar_w = int(group_w * 0.14)
    metric_colors = [(72, 145, 170), (91, 160, 93), (87, 132, 193), (192, 80, 77)]
    for i, row in enumerate(rows):
        cx = int(x0 + group_w * i + group_w * 0.18)
        for j, metric in enumerate(metrics):
            try:
                value = float(row[metric])
            except (TypeError, ValueError):
                value = 0.0
            bh = int(plot_h * max(0.0, min(1.0, value)))
            x = cx + j * (bar_w + 7)
            cv2.rectangle(img, (x, y0 + plot_h - bh), (x + bar_w, y0 + plot_h), metric_colors[j], -1)
            cv2.putText(img, fmt(value, 3), (x - 5, y0 + plot_h - bh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, row["Method"], (int(x0 + group_w * i + 8), y0 + plot_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1, cv2.LINE_AA)

    legend_x = x0 + plot_w - 365
    for j, metric in enumerate(metrics):
        cv2.rectangle(img, (legend_x, y0 + 20 + 32 * j), (legend_x + 22, y0 + 42 + 32 * j), metric_colors[j], -1)
        cv2.putText(img, metric, (legend_x + 34, y0 + 39 + 32 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    y1 = 660
    cv2.putText(img, "TCF", (42, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (30, 30, 30), 2, cv2.LINE_AA)
    max_tcf = max(float(row["TCF"]) for row in rows if row.get("TCF"))
    max_tcf = max(0.08, max_tcf * 1.15)
    bar_area_w = 1120
    for i, row in enumerate(rows):
        value = float(row["TCF"])
        bw = int(bar_area_w * value / max_tcf)
        y = y1 + i * 45
        color = COLORS.get(row["Method"], (120, 120, 120))
        cv2.rectangle(img, (210, y), (210 + bw, y + 24), color, -1)
        cv2.putText(img, row["Method"], (42, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
        cv2.putText(img, fmt(value), (220 + bw, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    save_png(path, img)


def draw_scatter(path: Path, table_rows: list[dict[str, Any]], title: str) -> None:
    w, h = 1200, 900
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(img, title, (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2, cv2.LINE_AA)
    x0, y0, pw, ph = 110, 110, 930, 650
    xmin, xmax = 0.45, 0.95
    ymin, ymax = 0.45, 1.02
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (210, 210, 210), 1)
    for i in range(6):
        x = x0 + int(pw * i / 5)
        y = y0 + ph - int(ph * i / 5)
        cv2.line(img, (x, y0), (x, y0 + ph), (238, 238, 238), 1)
        cv2.line(img, (x0, y), (x0 + pw, y), (238, 238, 238), 1)
        cv2.putText(img, f"{xmin + (xmax-xmin)*i/5:.2f}", (x - 25, y0 + ph + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(img, f"{ymin + (ymax-ymin)*i/5:.2f}", (35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, "JM", (x0 + pw // 2, y0 + ph + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "JR", (30, y0 + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 1, cv2.LINE_AA)

    for row in table_rows:
        try:
            jm = float(row["JM"])
            jr = float(row["JR"])
        except (KeyError, TypeError, ValueError):
            continue
        x = int(x0 + (jm - xmin) / (xmax - xmin) * pw)
        y = int(y0 + ph - (jr - ymin) / (ymax - ymin) * ph)
        label = str(row.get("Method") or row.get("Setting") or row.get("candidate") or "")
        color = COLORS.get(label, (80, 80, 80))
        radius = 12 if "Phase6" in label or row.get("Final") == "1" else 8
        cv2.circle(img, (x, y), radius, color, -1)
        cv2.putText(img, label[:30], (x + 14, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (35, 35, 35), 1, cv2.LINE_AA)
    save_png(path, img)


def draw_gtcoverage_scatter(path: Path, table_rows: list[dict[str, Any]], title: str) -> None:
    w, h = 1200, 900
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(img, title, (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.08, (30, 30, 30), 2, cv2.LINE_AA)
    x0, y0, pw, ph = 130, 110, 900, 650
    xmin, xmax = 0.60, 1.02
    ymin, ymax = 0.45, 0.95
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (210, 210, 210), 1)
    for i in range(6):
        x = x0 + int(pw * i / 5)
        y = y0 + ph - int(ph * i / 5)
        cv2.line(img, (x, y0), (x, y0 + ph), (238, 238, 238), 1)
        cv2.line(img, (x0, y), (x0 + pw, y), (238, 238, 238), 1)
        cv2.putText(img, f"{xmin + (xmax-xmin)*i/5:.2f}", (x - 25, y0 + ph + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(img, f"{ymin + (ymax-ymin)*i/5:.2f}", (45, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.putText(img, GT_COVERAGE_KEY, (x0 + pw // 2 - 85, y0 + ph + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "MaskScore", (18, y0 + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 1, cv2.LINE_AA)

    for row in table_rows:
        try:
            gt_cov = float(row[GT_COVERAGE_KEY])
            score = float(row["MaskScore"])
        except (KeyError, TypeError, ValueError):
            continue
        x = int(x0 + (gt_cov - xmin) / (xmax - xmin) * pw)
        y = int(y0 + ph - (score - ymin) / (ymax - ymin) * ph)
        x = max(x0, min(x0 + pw, x))
        y = max(y0, min(y0 + ph, y))
        label = str(row.get("Method") or row.get("Setting") or "")
        color = COLORS.get(label, (80, 80, 80))
        radius = 12 if "Phase6" in label or row.get("Final") == "1" else 8
        cv2.circle(img, (x, y), radius, color, -1)
        cv2.putText(img, label[:30], (x + 14, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (35, 35, 35), 1, cv2.LINE_AA)
    save_png(path, img)


def make_a_vs_b_figure(out_path: Path, methods: list[MethodSpec]) -> None:
    method_map = {m.key: m for m in methods}
    rows: list[np.ndarray] = []
    for ds in DATASETS:
        idx = FRAME_PICK[ds]
        tiles = [put_label(ensure_size(read_input_frame(ds, idx), (280, 158)), f"{ds} input", f"frame {idx}")]
        for key in ["A", "B"]:
            spec = method_map[key]
            frame = read_video_frame(restored_video(spec, ds), idx)
            mask = read_video_frame(mask_video(spec, ds), idx)
            over = overlay_mask(frame, mask)
            tiles.append(put_label(ensure_size(over, (280, 158)), spec.label, "restored + mask"))
        rows.append(hstack_tiles(tiles))
    save_png(out_path, vstack_rows(rows))


def make_visual_comparison(out_path: Path, methods: list[MethodSpec]) -> None:
    rows: list[np.ndarray] = []
    for ds in DATASETS:
        idx = FRAME_PICK[ds]
        tiles = [put_label(ensure_size(read_input_frame(ds, idx), (240, 135)), f"{ds}", f"input {idx}")]
        for spec in methods[1:]:
            frame = read_video_frame(restored_video(spec, ds), idx)
            tiles.append(put_label(ensure_size(frame, (240, 135)), spec.label))
        rows.append(hstack_tiles(tiles, pad=6))
    save_png(out_path, vstack_rows(rows, pad=8))


def make_boundary_figure(out_path: Path, methods: list[MethodSpec]) -> None:
    method_map = {m.key: m for m in methods}
    rows: list[np.ndarray] = []
    for ds, idx in BOUNDARY_FRAME_PICK.items():
        original = read_input_frame(ds, idx)
        h, w = original.shape[:2]
        masks = []
        for key in ["B", "E", "F", "P6"]:
            mask = read_video_frame(mask_video(method_map[key], ds), idx)
            masks.append(mask_as_binary(mask, (h, w)))
        union = np.logical_or.reduce(masks)
        box = bbox_from_binary(union)
        tiles = [put_label(ensure_size(crop_box(original, box), (260, 180)), f"{ds}", f"crop {idx}")]
        for key, binary in zip(["B", "E", "F", "P6"], masks):
            over = overlay_mask(original, (binary.astype(np.uint8) * 255))
            tiles.append(put_label(ensure_size(crop_box(over, box), (260, 180)), method_map[key].label))
        rows.append(hstack_tiles(tiles, pad=6))
    save_png(out_path, vstack_rows(rows, pad=8))


def make_b_vs_f_figure(out_path: Path, methods: list[MethodSpec]) -> None:
    method_map = {m.key: m for m in methods}
    rows: list[np.ndarray] = []
    for ds in ["bmx-trees", "tennis", "wild"]:
        idx = FRAME_PICK[ds]
        tiles = [put_label(ensure_size(read_input_frame(ds, idx), (280, 158)), f"{ds} input", f"frame {idx}")]
        for key in ["B", "F", "P6"]:
            spec = method_map[key]
            frame = read_video_frame(restored_video(spec, ds), idx)
            mask = read_video_frame(mask_video(spec, ds), idx)
            tiles.append(put_label(ensure_size(overlay_mask(frame, mask), (280, 158)), spec.label))
        rows.append(hstack_tiles(tiles, pad=6))
    save_png(out_path, vstack_rows(rows, pad=8))


def make_g_failure_strip(out_path: Path, methods: list[MethodSpec], g_spec: MethodSpec) -> None:
    b_spec = next(m for m in methods if m.key == "B")
    rows: list[np.ndarray] = []
    for ds in ["wild", "bmx-trees", "tennis"]:
        for spec in [b_spec, g_spec]:
            tiles = []
            for idx in FAILURE_FRAME_PICK[ds]:
                frame = read_video_frame(restored_video(spec, ds), idx)
                tiles.append(put_label(ensure_size(frame, (260, 146)), f"{ds} {spec.label}", f"frame {idx}"))
            rows.append(hstack_tiles(tiles, pad=6))
    save_png(out_path, vstack_rows(rows, pad=8))


def make_phase6_candidate_chart(out_path: Path, phase6_exp_id: str) -> None:
    rows = read_csv(REPO_ROOT / "outputs" / "metrics" / phase6_exp_id / "phase6_pareto.csv")
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "Setting": f"{row.get('stage')}/{row.get('candidate')}",
                GT_COVERAGE_KEY: row.get(GT_COVERAGE_KEY),
                "JM": row.get("JM"),
                "JR": row.get("JR"),
                "MaskScore": row.get("MaskScore"),
                "Final": row.get("is_final_best"),
            }
        )
    draw_gtcoverage_scatter(out_path, table_rows, "Phase 6 Candidate Pareto: GT_Coverage vs MaskScore")


def build_figures(
    out_dir: Path,
    main_rows: list[dict[str, Any]],
    phase6_ablation_rows: list[dict[str, Any]],
    methods: list[MethodSpec],
    *,
    phase6_exp_id: str,
    g_spec: MethodSpec,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "overall_metrics": out_dir / "figure1_overall_metrics.png",
        "a_vs_b": out_dir / "figure2_a_vs_b_global.png",
        "visual_comparison": out_dir / "figure3_b_e_f_phase6_visual.png",
        "boundary": out_dir / "figure4_b_e_f_phase6_boundary.png",
        "b_f_hard": out_dir / "figure5_b_f_phase6_hard_scene.png",
        "jmjr": out_dir / "figure6_jmjr_main_methods.png",
        "phase6_pareto": out_dir / "figure7_phase6_pareto.png",
        "phase5_failure": out_dir / "figure8_phase5_failure_strip.png",
    }
    draw_metric_bar_chart(paths["overall_metrics"], main_rows)
    make_a_vs_b_figure(paths["a_vs_b"], methods)
    make_visual_comparison(paths["visual_comparison"], methods)
    make_boundary_figure(paths["boundary"], methods)
    make_b_vs_f_figure(paths["b_f_hard"], methods)
    draw_scatter(paths["jmjr"], main_rows, "Main Methods: JM vs JR")
    make_phase6_candidate_chart(paths["phase6_pareto"], phase6_exp_id)
    make_g_failure_strip(paths["phase5_failure"], methods, g_spec)
    return {key: rel(value) for key, value in paths.items()}


def build_summary_markdown(
    path: Path,
    table_paths: dict[str, dict[str, str]],
    figure_paths: dict[str, str],
    main_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Final Results Artifacts",
        "",
        "Generated from accepted experiment outputs under `outputs/metrics` and `outputs/videos`.",
        "GT_Coverage/JM/JR and MaskScore are computed on datasets with GT masks (`bmx-trees`, `tennis`); MaskScore = 0.5*GT_Coverage + 0.25*JM + 0.25*JR. TCF is color-frame temporal consistency; FAST_VQA is reported from the configured official FAST-VQA wrapper.",
        "TCF is only a temporal smoothness proxy: blurry, low-frequency, or wrong-but-stable fills can score lower than sharper realistic fills. FAST_VQA better matches our human visual inspection and is used with qualitative figures for generated-video quality.",
        "Phase 5/G is kept as qualitative failure analysis and is not ranked in the main table.",
        "",
        "## Main Table",
        "",
        markdown_table(
            ["Method", GT_COVERAGE_KEY, "JM", "JR", "MaskScore", "TCF", "FAST_VQA", "Delta GT_Coverage vs B", "Delta JM vs B", "Delta JR vs B"],
            main_rows,
        ),
        "## Table Files",
        "",
    ]
    for key, paths in table_paths.items():
        lines.append(f"- {key}: `{paths['csv']}`, `{paths['md']}`, `{paths['tex']}`")
    lines.extend(["", "## Figure Files", ""])
    for key, fig_path in figure_paths.items():
        lines.append(f"- {key}: `{fig_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final result tables and figures from accepted AIAA3201 runs.")
    parser.add_argument("--metrics-out", type=Path, default=Path("outputs/metrics/final_results"))
    parser.add_argument("--figures-out", type=Path, default=Path("outputs/figures/final_results"))
    parser.add_argument("--phase1-exp-id", default="phase1_maskscore_fastvqa_20260510_023457_pl220")
    parser.add_argument("--phase2-exp-id", default="phase2_maskscore_fastvqa_20260510_023457_pl220")
    parser.add_argument("--phase3-exp-id", default="phase3_maskscore_fastvqa_20260510_023457_pl220")
    parser.add_argument("--phase4-exp-id", default="phase4_maskscore_fastvqa_altbackend_20260510_pl220")
    parser.add_argument("--phase5-exp-id", default="phase5_maskscore_fastvqa_20260510_023457_pl220")
    parser.add_argument("--phase6-exp-id", default="phase6_core_maskscore_fastvqa_20260511_235610_pl220")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_out = args.metrics_out if args.metrics_out.is_absolute() else REPO_ROOT / args.metrics_out
    figures_out = args.figures_out if args.figures_out.is_absolute() else REPO_ROOT / args.figures_out
    metrics_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)
    methods, g_spec = build_method_specs(args)

    table_paths: dict[str, dict[str, str]] = {}
    main_rows, table_paths["table1_main_performance"] = build_main_table(metrics_out, methods)
    _, table_paths["table2_a_ablation"] = build_a_ablation(metrics_out, args.phase1_exp_id)
    phase6_ablation_rows, table_paths["table3_bef_phase6_ablation"] = build_extension_ablation(
        metrics_out,
        phase2_exp_id=args.phase2_exp_id,
        phase3_exp_id=args.phase3_exp_id,
        phase4_exp_id=args.phase4_exp_id,
        phase6_exp_id=args.phase6_exp_id,
    )
    _, table_paths["table4_phase5_qualitative"] = build_phase5_failure_table(metrics_out, args.phase5_exp_id)
    figure_paths = build_figures(
        figures_out,
        main_rows,
        phase6_ablation_rows,
        methods,
        phase6_exp_id=args.phase6_exp_id,
        g_spec=g_spec,
    )

    manifest = {
        "source_experiments": {spec.label: spec.exp_id for spec in [*methods, g_spec]},
        "tables": table_paths,
        "figures": figure_paths,
        "notes": [
            "JM/JR aggregate excludes GT-missing wild through missing metric values.",
            "Phase 5/G is qualitative failure analysis only.",
        ],
    }
    manifest_path = metrics_out / "final_results_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    build_summary_markdown(metrics_out / "final_results_summary.md", table_paths, figure_paths, main_rows)
    print(f"[OK] tables: {rel(metrics_out)}")
    print(f"[OK] figures: {rel(figures_out)}")
    print(f"[OK] manifest: {rel(manifest_path)}")


if __name__ == "__main__":
    main()
