"""Evaluation metrics. Convention: y = 1 means error; higher score means more likely error."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

Metric = Callable[[np.ndarray, np.ndarray], float]


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s))


def auprc(y, s) -> float:
    return float(average_precision_score(y, s))


def reliability_bins(y, p, n_bins: int) -> list[dict]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else None,
                     "frac_pos": float(y[m].mean()) if m.any() else None})
    return bins


def ece(y, p, n_bins: int = 15) -> float:
    n = len(y)
    return float(sum(b["count"] / n * abs(b["mean_pred"] - b["frac_pos"])
                     for b in reliability_bins(y, p, n_bins) if b["count"]))


def risk_coverage(y, s) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk at coverage k/N (k = 1..N), answering the lowest-score examples first.
    Tied scores are resolved by their expected risk under uniform random tie-breaking."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(y)
    order = np.argsort(s, kind="stable")
    ys, ss = y[order], s[order]
    risk = np.empty(n)
    errors_before, start = 0.0, 0
    while start < n:
        end = start
        while end + 1 < n and ss[end + 1] == ss[start]:
            end += 1
        group = ys[start : end + 1]
        rate = group.mean()
        for k in range(start + 1, end + 2):
            risk[k - 1] = (errors_before + (k - start) * rate) / k
        errors_before += group.sum()
        start = end + 1
    return np.arange(1, n + 1) / n, risk


def aurc(y, s) -> float:
    return float(risk_coverage(y, s)[1].mean())


def risk_at_coverage(y, s, coverage: float) -> float:
    risk = risk_coverage(y, s)[1]
    k = max(1, math.ceil(coverage * len(risk) - 1e-9))
    return float(risk[k - 1])


def bootstrap_ci(metric: Metric, y, s, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI; resamples containing a single class are skipped."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], s[idx]))
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_delta(metric: Metric, y, s_a, s_b, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """metric(b) - metric(a) with a paired percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(metric(y[idx], s_b[idx]) - metric(y[idx], s_a[idx]))
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2]) if deltas else (float("nan"), float("nan"))
    return {"delta": metric(y, s_b) - metric(y, s_a), "lo": float(lo), "hi": float(hi)}


def summarize(y, score, prob=None, *, n_boot: int, seed: int, ece_bins: int) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not np.all(np.isfinite(score)):
        raise ValueError("scores must be finite; fill NaN scores before evaluation")
    out = {
        "n": int(len(y)), "base_rate": float(y.mean()),
        "auroc": auroc(y, score), "auroc_ci": list(bootstrap_ci(auroc, y, score, n_boot, seed)),
        "auprc": auprc(y, score), "auprc_ci": list(bootstrap_ci(auprc, y, score, n_boot, seed)),
        "aurc": aurc(y, score),
        "risk_at_50": risk_at_coverage(y, score, 0.5), "risk_at_80": risk_at_coverage(y, score, 0.8),
    }
    if prob is not None:
        out["ece"] = ece(y, prob, ece_bins)
        out["reliability"] = reliability_bins(y, prob, ece_bins)
    return out
