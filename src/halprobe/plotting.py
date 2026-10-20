"""Run-report figures (headless matplotlib)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical palette in fixed slot order, validated for colour-vision deficiency on a light surface
# (adjacent-pair CVD dE >= 9.1, normal-vision dE >= 19.6). Slots 3-5 are below 3:1 contrast, so every
# figure carries a legend and the numbers are also in tables/summary.md.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
NEUTRAL = "#898781"
GRID = "#e1e0d9"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layer_sweep(layers: Sequence[int], val_auroc: Sequence[float], test_auroc: Sequence[float],
                     test_ci: Sequence[Sequence[float]], baselines: Mapping[str, float], selected_layer: int,
                     path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lo = np.array([c[0] for c in test_ci], dtype=float)
    hi = np.array([c[1] for c in test_ci], dtype=float)
    ax.fill_between(layers, lo, hi, color=PALETTE[0], alpha=0.15, linewidth=0, label="probe test 95% CI")
    ax.plot(layers, test_auroc, color=PALETTE[0], marker="o", markersize=3, label="probe (test)")
    ax.plot(layers, val_auroc, color=PALETTE[0], linestyle="--", linewidth=1, label="probe (val)")
    for i, (name, value) in enumerate(baselines.items(), start=1):
        ax.axhline(value, color=PALETTE[i % len(PALETTE)], linestyle=":", linewidth=1.4, label=name)
    ax.axhline(0.5, color="#cccccc", linewidth=0.8)
    finite = [v for v in [*val_auroc, *test_auroc, *lo, *baselines.values()] if np.isfinite(v)]
    ax.set_ylim(min(0.45, min(finite) - 0.02), 1.0)
    ax.axvline(selected_layer, color=NEUTRAL, linewidth=0.8)
    ax.text(selected_layer, 0.99, " val-selected", va="top", fontsize=8, color="#555555")
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("AUROC (error = positive)")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style(ax)
    _save(fig, path)


def plot_reliability(bins_by_method: Mapping[str, Sequence[dict]], path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=0.8, linestyle="--", label="perfect calibration")
    for i, (name, bins) in enumerate(bins_by_method.items()):
        points = [(b["mean_pred"], b["frac_pos"]) for b in bins if b["count"]]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, marker="o", markersize=3, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted P(error)", ylabel="observed error rate")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    _save(fig, path)


def plot_risk_coverage(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], base_rate: float,
                       path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (name, (coverage, risk)) in enumerate(curves.items()):
        ax.plot(coverage, risk, color=PALETTE[i % len(PALETTE)], linewidth=1.4, label=name)
    ax.axhline(base_rate, color=NEUTRAL, linewidth=0.8, linestyle="--", label="no abstention")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="coverage (fraction answered)", ylabel="error rate among answered")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    _save(fig, path)


def plot_relation_auroc(per_relation: Mapping[str, Mapping[str, float]], path: Path | str) -> None:
    relations = sorted({r for d in per_relation.values() for r in d})
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.3 * len(relations) * max(1, len(per_relation)))))
    if not relations:
        ax.text(0.5, 0.5, "no relation has enough test examples of both classes", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return
    height = 0.8 / len(per_relation)
    ys = np.arange(len(relations))
    for i, (name, values) in enumerate(per_relation.items()):
        ax.barh(ys + i * height, [values.get(r, np.nan) for r in relations], height=height,
                color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=1.0, label=name)
    ax.set_yticks(ys + height * (len(per_relation) - 1) / 2, relations)
    ax.axvline(0.5, color=NEUTRAL, linewidth=0.8)
    ax.set(xlim=(0, 1), xlabel="AUROC within relation (test)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)
