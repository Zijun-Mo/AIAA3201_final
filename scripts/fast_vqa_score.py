#!/usr/bin/env python3
"""Run official FAST-VQA/FasterVQA inference and write a JSON score."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


MODEL_OPTIONS = {
    "FasterVQA": "./options/fast/f3dvqa-b.yml",
    "FasterVQA-MS": "./options/fast/fastervqa-ms.yml",
    "FasterVQA-MT": "./options/fast/fastervqa-mt.yml",
    "FAST-VQA": "./options/fast/fast-b.yml",
    "FAST-VQA-M": "./options/fast/fast-m.yml",
}

MEAN_STDS = {
    "FasterVQA": (0.14759505, 0.03613452),
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA": (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official FAST-VQA wrapper for a single video.")
    parser.add_argument("--repo-path", required=True, help="Path to FAST-VQA-and-FasterVQA checkout")
    parser.add_argument("--video", "--video-path", dest="video_path", required=True, help="Input MP4")
    parser.add_argument("--output-json", required=True, help="Where to write JSON with FAST_VQA score")
    parser.add_argument("--model", default="FAST-VQA-M", choices=sorted(MODEL_OPTIONS), help="FAST-VQA model variant")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    return parser.parse_args()


def sigmoid_rescale(score: float, model: str) -> float:
    mean, std = MEAN_STDS[model]
    x = (score - mean) / std
    return float(1.0 / (1.0 + np.exp(-x)))


def choose_data_args(opt: dict) -> dict:
    data = opt.get("data", {}) or {}
    preferred = data.get("val-kv1k")
    if isinstance(preferred, dict) and isinstance(preferred.get("args"), dict):
        return preferred["args"]
    for key in sorted(data.keys()):
        block = data.get(key, {}) or {}
        args = block.get("args", {}) or {}
        if str(key).startswith("val") and isinstance(args.get("sample_types"), dict):
            return args
    for block in data.values():
        args = (block or {}).get("args", {}) or {}
        if isinstance(args.get("sample_types"), dict):
            return args
    raise KeyError("No FAST-VQA validation data args with sample_types found")


def main() -> int:
    args = parse_args()
    repo_path = Path(args.repo_path).resolve()
    video_path = Path(args.video_path).resolve()
    output_json = Path(args.output_json).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"FAST-VQA repo not found: {repo_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    os.chdir(repo_path)
    sys.path.insert(0, str(repo_path))

    import decord
    from fastvqa.datasets import FragmentSampleFrames, SampleFrames, get_spatial_fragments
    from fastvqa.models import DiViDeAddEvaluator

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    opt_path = repo_path / MODEL_OPTIONS[args.model]
    with opt_path.open("r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)

    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(device)
    checkpoint = torch.load(opt["test_load_path"], map_location=device)
    evaluator.load_state_dict(checkpoint["state_dict"])
    evaluator.eval()

    video_reader = decord.VideoReader(str(video_path))
    t_data_opt = choose_data_args(opt)
    sample_types = t_data_opt["sample_types"]
    vsamples = {}

    for sample_type, sample_args in sample_types.items():
        if t_data_opt.get("t_frag", 1) > 1:
            sampler = FragmentSampleFrames(
                fsize_t=sample_args["clip_len"] // sample_args.get("t_frag", 1),
                fragments_t=sample_args.get("t_frag", 1),
                num_clips=sample_args.get("num_clips", 1),
            )
        else:
            sampler = SampleFrames(
                clip_len=sample_args["clip_len"],
                num_clips=sample_args.get("num_clips", 1),
            )

        num_clips = sample_args.get("num_clips", 1)
        frames = sampler(len(video_reader))
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0).permute(3, 0, 1, 2)

        sampled_video = get_spatial_fragments(video, **sample_args)
        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        sampled_video = sampled_video.reshape(
            sampled_video.shape[0],
            num_clips,
            -1,
            *sampled_video.shape[2:],
        ).transpose(0, 1)
        vsamples[sample_type] = sampled_video.to(device)

    with torch.no_grad():
        result = evaluator(vsamples)
    score = sigmoid_rescale(float(result.mean().item()), args.model)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "FAST_VQA": score,
        "score": score,
        "model": args.model,
        "device": device,
        "video": str(video_path),
        "repo_path": str(repo_path),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"FAST_VQA {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
