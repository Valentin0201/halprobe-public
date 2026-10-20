import numpy as np
import pytest

from halprobe.baselines import (
    calibrate,
    fill_nan_as_max,
    first_token_entropy,
    fit_logreg,
    metadata_features,
    neg_mean_logprob,
    popularity_features,
)
from halprobe.metrics import auroc


def test_neg_mean_logprob_excludes_stop_token():
    recs = [{"token_logprobs": [-1.0, -3.0, -0.5], "n_answer_tokens": 2}]
    assert neg_mean_logprob(recs).tolist() == [2.0]


def test_neg_mean_logprob_nan_for_empty_answer():
    assert np.isnan(neg_mean_logprob([{"token_logprobs": [-0.1], "n_answer_tokens": 0}])[0])


def test_fill_nan_as_max():
    assert fill_nan_as_max([1.0, np.nan, 3.0]).tolist() == [1.0, 4.0, 3.0]
    assert fill_nan_as_max([np.nan]).tolist() == [0.0]


def test_first_token_entropy():
    assert first_token_entropy([{"first_token_entropy": 1.5}, {"first_token_entropy": 0.2}]).tolist() == [1.5, 0.2]


def test_metadata_features_one_hot_and_unseen_relation():
    X = metadata_features([10, 1000, 0], ["genre", "capital", "sport"], ["capital", "genre"])
    assert X.tolist() == [[1.0, 0.0, 1.0], [3.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    assert popularity_features([100]).tolist() == [[2.0]]


def test_fit_logreg_learns_popularity_shortcut():
    rng = np.random.default_rng(0)
    s_pop = 10 ** rng.uniform(0, 6, 400)
    y = (np.log10(s_pop) + rng.normal(0, 0.5, 400) < 3).astype(int)  # rare entities -> errors
    X = popularity_features(s_pop)
    (p,) = fit_logreg(X[:300], y[:300], [X[300:]], seed=0)
    assert auroc(y[300:], p) > 0.9


def test_calibrate_returns_monotone_probabilities():
    rng = np.random.default_rng(1)
    score = rng.normal(size=300)
    y = (score + rng.normal(0, 1, 300) > 0).astype(int)
    p = calibrate(score[:200], y[:200], score[200:], seed=0)
    assert np.all((p > 0) & (p < 1))
    order = np.argsort(score[200:])
    assert np.all(np.diff(p[order]) >= -1e-12)
