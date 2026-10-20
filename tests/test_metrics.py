import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from halprobe.metrics import (
    aurc,
    auprc,
    auroc,
    bootstrap_ci,
    ece,
    paired_bootstrap_delta,
    reliability_bins,
    risk_at_coverage,
    risk_coverage,
    summarize,
)


def test_auroc_auprc_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(0, 1, 200)
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert auprc(y, s) == pytest.approx(average_precision_score(y, s))


def test_ece_hand_computed():
    y = [0, 1, 1, 0]
    p = [0.1, 0.9, 0.8, 0.3]
    assert ece(y, p, n_bins=2) == pytest.approx(0.175)
    bins = reliability_bins(y, p, n_bins=2)
    assert [b["count"] for b in bins] == [2, 2]
    assert bins[0]["mean_pred"] == pytest.approx(0.2) and bins[1]["frac_pos"] == pytest.approx(1.0)


def test_reliability_bins_handle_empty_bins_and_p_equal_one():
    bins = reliability_bins([1], [1.0], n_bins=4)
    assert [b["count"] for b in bins] == [0, 0, 0, 1]
    assert bins[0]["mean_pred"] is None


def test_risk_coverage_hand_computed():
    cov, risk = risk_coverage([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
    assert aurc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)


def test_risk_coverage_ties_use_expected_risk_and_ignore_input_order():
    y, s = [0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]
    _, risk = risk_coverage(y, s)
    assert risk.tolist() == pytest.approx([0.0, 0.25, 1 / 3, 0.5])
    _, risk_permuted = risk_coverage([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1])
    assert risk_permuted.tolist() == pytest.approx(risk.tolist())


def test_risk_at_coverage():
    y, s = [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]
    assert risk_at_coverage(y, s, 0.5) == pytest.approx(0.0)
    assert risk_at_coverage(y, s, 0.8) == pytest.approx(0.5)


def test_bootstrap_ci_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1, 300)
    a = bootstrap_ci(auroc, y, s, n_boot=200, seed=0)
    assert a == bootstrap_ci(auroc, y, s, n_boot=200, seed=0)
    assert a[0] < auroc(y, s) < a[1]


def test_paired_delta_zero_for_identical_scores():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 100)
    s = rng.normal(size=100)
    d = paired_bootstrap_delta(auroc, y, s, s, n_boot=50, seed=0)
    assert d["delta"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0


def test_summarize_keys():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 100)
    s = rng.random(100)
    out = summarize(y, s, s, n_boot=20, seed=0, ece_bins=10)
    assert {"n", "base_rate", "auroc", "auroc_ci", "auprc", "auprc_ci", "aurc", "risk_at_50", "risk_at_80",
            "ece", "reliability"} <= set(out)
    assert out["n"] == 100 and len(out["reliability"]) == 10


def test_summarize_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        summarize([0, 1], [0.1, float("nan")], n_boot=5, seed=0, ece_bins=5)
