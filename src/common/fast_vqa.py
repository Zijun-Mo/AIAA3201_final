#!/usr/bin/env python3
"""Optional FAST-VQA subprocess integration."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _format_token(token: Any, video_path: Path, repo_root: Path, output_json: Path) -> str:
    return str(token).format(
        video=str(video_path),
        video_path=str(video_path),
        repo_root=str(repo_root),
        output_json=str(output_json),
    )


def _coerce_command(raw: Any, video_path: Path, repo_root: Path, output_json: Path) -> list[str]:
    if isinstance(raw, list):
        return [_format_token(x, video_path, repo_root, output_json) for x in raw]
    if isinstance(raw, str) and raw.strip():
        return [_format_token(x, video_path, repo_root, output_json) for x in raw.strip().split()]
    return []


def _extract_score_from_json(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(payload, dict):
        for key in ["FAST_VQA", "fast_vqa", "score", "quality", "mean_score"]:
            if key in payload:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    continue
    if isinstance(payload, list) and payload:
        vals: list[float] = []
        for item in payload:
            if isinstance(item, dict):
                for key in ["FAST_VQA", "fast_vqa", "score", "quality", "mean_score"]:
                    if key in item:
                        try:
                            vals.append(float(item[key]))
                        except (TypeError, ValueError):
                            pass
        if vals:
            return float(sum(vals) / len(vals))
    return None


def _extract_score_from_text(text: str, pattern: str) -> float | None:
    try:
        regex = re.compile(pattern, flags=re.IGNORECASE)
    except re.error:
        regex = re.compile(r"(?:FAST[_-]?VQA|score|quality)[^0-9+-]*([+-]?\d+(?:\.\d+)?)", flags=re.IGNORECASE)
    matches = regex.findall(text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = next((x for x in value if x), "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_fast_vqa_for_video(
    video_path: Path,
    cfg: dict[str, Any],
    *,
    repo_root: Path,
) -> tuple[float | None, dict[str, Any]]:
    """Run FAST-VQA if a real command is configured.

    The repo does not vendor FAST-VQA. Configure `evaluation.fast_vqa.command`
    with placeholders such as `{video}` and `{output_json}` to point at a local
    FAST-VQA checkout or wrapper.
    """
    fast_cfg = cfg or {}
    if not bool(fast_cfg.get("enabled", True)):
        return None, {"status": "disabled", "reason": "evaluation.fast_vqa.enabled=false"}
    if not video_path.exists():
        return None, {"status": "skipped", "reason": f"video_missing:{video_path}"}

    output_json = video_path.parent / "fast_vqa_score.json"
    if output_json.exists() and not bool(fast_cfg.get("force_recompute", False)):
        cached = _extract_score_from_json(output_json)
        if cached is not None:
            return cached, {"status": "cached", "output_json": str(output_json)}

    command = _coerce_command(fast_cfg.get("command", []), video_path, repo_root, output_json)
    if not command:
        executable = shutil.which(str(fast_cfg.get("executable", "fastvqa_score")))
        if executable:
            command = [executable, "--video", str(video_path), "--output-json", str(output_json)]

    if not command:
        return None, {
            "status": "not_configured",
            "reason": "no evaluation.fast_vqa.command and no fastvqa_score executable on PATH",
        }

    env_name = str(fast_cfg.get("env_name", "")).strip()
    if env_name:
        command = ["conda", "run", "--no-capture-output", "-n", env_name, *command]

    timeout_sec = int(fast_cfg.get("timeout_sec", 900))
    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, {"status": "failed", "reason": f"timeout_after_{timeout_sec}s", "command": command}
    except Exception as exc:
        return None, {"status": "failed", "reason": str(exc), "command": command}

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    score = _extract_score_from_json(output_json)
    if score is None:
        pattern = str(
            fast_cfg.get(
                "output_regex",
                r"(?:FAST[_-]?VQA|score|quality)[^0-9+-]*([+-]?\d+(?:\.\d+)?)",
            )
        )
        score = _extract_score_from_text(stdout + "\n" + stderr, pattern)

    meta = {
        "status": "ok" if completed.returncode == 0 and score is not None else "failed",
        "returncode": int(completed.returncode),
        "command": command,
        "output_json": str(output_json),
        "stdout_tail": stdout[-1000:],
        "stderr_tail": stderr[-1000:],
    }
    if score is None:
        meta["reason"] = "score_not_found"
    return score, meta
