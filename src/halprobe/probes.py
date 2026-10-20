"""Per-layer logistic-regression probes; C and the reported layer are selected on validation AUROC."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from halprobe.metrics import auroc


@dataclass
class LayerFit:
    layer: int
    C: float
    val_auroc: float
    val_scores: np.ndarray  # P(error) on the labelled validation rows
    eval_scores: np.ndarray  # P(error) on all evaluation rows (e.g. every test row, incl. ambiguous)
    coef: np.ndarray  # in standardized feature space
    intercept: float


def fit_probe(X_train, y_train, X_val, y_val, X_eval, C_grid: Sequence[float], max_iter: int, seed: int):
    best = None
    for C in C_grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        score = auroc(y_val, val_scores)
        if best is None or score > best[1]:
            best = (C, score, val_scores, clf)
    C, score, val_scores, clf = best
    lr = clf[-1]
    return C, score, val_scores, clf.predict_proba(X_eval)[:, 1], lr.coef_[0].copy(), float(lr.intercept_[0])


def sweep_layers(states, train_idx, y_train, val_idx, y_val, eval_idx, C_grid: Sequence[float], max_iter: int,
                 seed: int, log: Callable[[str], None] = print) -> list[LayerFit]:
    """states: array-like (n_states, N, d), e.g. the read-only extraction memmap."""
    fits = []
    for layer in range(states.shape[0]):
        view = states[layer]

        def rows(idx, view=view):
            return np.asarray(view[np.asarray(idx)], dtype=np.float32)

        C, val_auc, val_scores, eval_scores, coef, intercept = fit_probe(
            rows(train_idx), y_train, rows(val_idx), y_val, rows(eval_idx), C_grid, max_iter, seed)
        fits.append(LayerFit(layer, C, val_auc, val_scores, eval_scores, coef, intercept))
        log(f"[probe] layer {layer:2d}: C={C:g} val AUROC={val_auc:.3f}")
    return fits


def select_layer(fits: Sequence[LayerFit]) -> int:
    """Best validation AUROC; ties go to the lower layer."""
    return max(fits, key=lambda f: (f.val_auroc, -f.layer)).layer
