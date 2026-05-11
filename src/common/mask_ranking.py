#!/usr/bin/env python3
"""Shared mask ranking utilities."""
from __future__ import annotations

import math
from typing import Any, Callable, Iterable, TypeVar

GT_COVERAGE_KEY = "GT_Coverage"
MASK_SCORE_WEIGHTS = {
    GT_COVERAGE_KEY: 0.5,
    "JM": 0.25,
    "JR": 0.25,
}

T = TypeVar("T")


def metric_float(metrics: dict[str, Any], key: str, default: float) -> float:
    value = metrics.get(key)
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def mask_score(metrics: dict[str, Any]) -> float:
    gt_coverage = metric_float(metrics, GT_COVERAGE_KEY, float("-inf"))
    jm = metric_float(metrics, "JM", float("-inf"))
    jr = metric_float(metrics, "JR", float("-inf"))
    if not math.isfinite(jm) or not math.isfinite(jr):
        return float("-inf")
    if not math.isfinite(gt_coverage):
        return (jm + jr) / 2.0
    return (
        MASK_SCORE_WEIGHTS[GT_COVERAGE_KEY] * gt_coverage
        + MASK_SCORE_WEIGHTS["JM"] * jm
        + MASK_SCORE_WEIGHTS["JR"] * jr
    )


def mask_tiebreak_key(metrics: dict[str, Any], mean_mask_ratio: float = 0.0) -> tuple[float, ...]:
    gt_coverage = metric_float(metrics, GT_COVERAGE_KEY, float("-inf"))
    jm = metric_float(metrics, "JM", float("-inf"))
    jr = metric_float(metrics, "JR", float("-inf"))
    tcf = metric_float(metrics, "TCF", float("inf"))
    if not math.isfinite(tcf):
        tcf = float("inf")
    return (
        mask_score(metrics),
        gt_coverage,
        jm,
        jr,
        -tcf,
        -abs(float(mean_mask_ratio) - 0.1),
    )


def select_maskscore_best(
    items: Iterable[T],
    metrics_fn: Callable[[T], dict[str, Any]],
    *,
    tiebreak_fn: Callable[[T], tuple[Any, ...]] | None = None,
) -> T:
    """Select by 0.5*GT_Coverage + 0.25*JM + 0.25*JR."""
    pool = list(items)
    if not pool:
        raise ValueError("select_maskscore_best requires at least one item")

    if tiebreak_fn is None:
        return max(pool, key=lambda item: mask_tiebreak_key(metrics_fn(item)))
    return max(pool, key=tiebreak_fn)


def select_gt_coverage_first(
    items: Iterable[T],
    metrics_fn: Callable[[T], dict[str, Any]],
    *,
    tie_eps: float | None = None,
    tiebreak_fn: Callable[[T], tuple[Any, ...]] | None = None,
) -> T:
    """Backward-compatible alias for the current composite mask ranking."""
    return select_maskscore_best(items, metrics_fn, tiebreak_fn=tiebreak_fn)
