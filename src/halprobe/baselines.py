"""Uncertainty and shortcut baselines. All scores: higher = more likely an error."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def neg_mean_logprob(records: Sequence[dict]) -> np.ndarray:
    """-mean log-prob over answer tokens (stop token excluded); NaN when the answer is empty."""
    out = []
    for r in records:
        n = r["n_answer_tokens"]
        out.append(-float(np.mean(r["token_logprobs"][:n])) if n > 0 else np.nan)
    return np.array(out, dtype=float)


def first_token_entropy(records: Sequence[dict]) -> np.ndarray:
    return np.array([r["first_token_entropy"] for r in records], dtype=float)


def fill_nan_as_max(scores) -> np.ndarray:
    """Treat undefined scores (empty answers) as maximally uncertain."""
    s = np.array(scores, dtype=float)
    finite = s[np.isfinite(s)]
    s[~np.isfinite(s)] = finite.max() + 1.0 if finite.size else 0.0
    return s


def popularity_features(s_pop) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(s_pop, dtype=float), 1.0))[:, None]


def metadata_features(s_pop, relations: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    index = {r: i for i, r in enumerate(vocabulary)}
    onehot = np.zeros((len(relations), len(vocabulary)))
    for row, rel in enumerate(relations):
        if rel in index:
            onehot[row, index[rel]] = 1.0
    return np.hstack([popularity_features(s_pop), onehot])


def fit_logreg(X_train, y_train, X_eval_list: Sequence[np.ndarray], seed: int) -> list[np.ndarray]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=seed))
    clf.fit(X_train, y_train)
    return [clf.predict_proba(X)[:, 1] for X in X_eval_list]


def calibrate(score_train, y_train, score_eval, seed: int) -> np.ndarray:
    """Platt-style 1-D logistic calibration of a raw score into P(error)."""
    return fit_logreg(np.asarray(score_train)[:, None], y_train, [np.asarray(score_eval)[:, None]], seed)[0]
