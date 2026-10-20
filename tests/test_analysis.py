import numpy as np
import pytest

from halprobe.analysis import incremental_value, label_sensitivity, popularity_quintile_auroc, within_relation_auroc
from halprobe.metrics import auroc


def test_within_relation_exposes_relation_shortcut():
    rng = np.random.default_rng(0)
    relations = np.array(["capital"] * 100 + ["director"] * 100)
    y = np.concatenate([rng.random(100) < 0.2, rng.random(100) < 0.8]).astype(int)
    s = (relations == "director").astype(float) + rng.normal(0, 0.01, 200)  # knows only the relation
    assert auroc(y, s) > 0.75
    res = within_relation_auroc(y, s, relations, min_per_class=5)
    assert res["n_relations"] == 2
    assert abs(res["macro"] - 0.5) < 0.15


def test_within_relation_skips_small_relations():
    y = np.array([0, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.2, 0.3])
    rel = np.array(["a", "a", "b", "b"])
    res = within_relation_auroc(y, s, rel, min_per_class=1)
    assert list(res["per_relation"]) == ["a"]
    none = within_relation_auroc(y, s, rel, min_per_class=5)
    assert none["macro"] is None and none["n_relations"] == 0


def test_popularity_quintiles_cover_all_examples():
    rng = np.random.default_rng(1)
    s_pop = rng.integers(1, 10**6, 500)
    y = rng.integers(0, 2, 500)
    bins = popularity_quintile_auroc(y, rng.random(500), s_pop)
    assert len(bins) == 5 and sum(b["n"] for b in bins) == 500
    assert all(b["auroc"] is None or 0 <= b["auroc"] <= 1 for b in bins)


def test_incremental_value_positive_when_probe_adds_information():
    rng = np.random.default_rng(2)
    n = 1200
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (a + b + rng.normal(0, 0.5, n) > 0).astype(int)
    meta = 1 / (1 + np.exp(-a))
    probe = 1 / (1 + np.exp(-b))
    res = incremental_value(meta[:600], probe[:600], y[:600], meta[600:], probe[600:], y[600:], n_boot=100, seed=0)
    assert res["auroc_combined"] > res["auroc_metadata"]
    assert res["delta"] > 0 and res["delta_ci"][0] > 0


def test_label_sensitivity_hand_computed():
    labels = ["correct", "incorrect", "ambiguous", "ambiguous"]
    res = label_sensitivity(labels, {"m": np.array([0.1, 0.9, 0.95, 0.05])})
    assert res["n_ambiguous"] == 2
    assert res["ambiguous_as_correct"]["m"] == pytest.approx(2 / 3)
    assert res["ambiguous_as_incorrect"]["m"] == pytest.approx(2 / 3)
