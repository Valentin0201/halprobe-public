"""Shortcut analyses (relation, popularity, incremental value) and label-sensitivity analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from halprobe.baselines import fit_logreg
from halprobe.metrics import auroc, paired_bootstrap_delta


def within_relation_auroc(y, s, relations, min_per_class: int) -> dict:
    """AUROC inside each relation with at least `min_per_class` examples of both classes; macro mean."""
    y, s, relations = np.asarray(y), np.asarray(s), np.asarray(relations)
    per = {}
    for rel in sorted(set(relations.tolist())):
        m = relations == rel
        n_pos = int(y[m].sum())
        if n_pos >= min_per_class and int(m.sum()) - n_pos >= min_per_class:
            per[rel] = auroc(y[m], s[m])
    return {"macro": float(np.mean(list(per.values()))) if per else None, "per_relation": per, "n_relations": len(per)}


def popularity_quintile_auroc(y, s, s_pop, n_bins: int = 5) -> list[dict]:
    y, s, s_pop = np.asarray(y), np.asarray(s), np.asarray(s_pop, dtype=float)
    edges = np.quantile(s_pop, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], s_pop, side="right"), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        two_classes = m.any() and 0 < int(y[m].sum()) < int(m.sum())
        out.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                    "base_rate": float(y[m].mean()) if m.any() else None,
                    "auroc": auroc(y[m], s[m]) if two_classes else None})
    return out


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def incremental_value(meta_val, probe_val, y_val, meta_test, probe_test, y_test, *, n_boot: int, seed: int) -> dict:
    """Does the probe add information beyond metadata? The 2-feature combiner is fit on validation,
    where both scores are out-of-sample (train-set probe scores are in-sample and over-confident)."""
    X_val = np.column_stack([_logit(meta_val), _logit(probe_val)])
    X_test = np.column_stack([_logit(meta_test), _logit(probe_test)])
    (combined,) = fit_logreg(X_val, y_val, [X_test], seed)
    d = paired_bootstrap_delta(auroc, y_test, meta_test, combined, n_boot, seed)
    return {"auroc_metadata": auroc(y_test, meta_test), "auroc_combined": auroc(y_test, combined),
            "delta": d["delta"], "delta_ci": [d["lo"], d["hi"]]}


def label_sensitivity(labels: Sequence[str], scores: Mapping[str, np.ndarray]) -> dict:
    labels = np.asarray(labels)
    out: dict = {"n_ambiguous": int((labels == "ambiguous").sum())}
    for name, ambiguous_y in (("ambiguous_as_correct", 0), ("ambiguous_as_incorrect", 1)):
        y = np.where(labels == "incorrect", 1, np.where(labels == "correct", 0, ambiguous_y))
        out[name] = {method: auroc(y, s) for method, s in scores.items()}
    return out
