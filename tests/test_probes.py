import numpy as np

from halprobe.probes import LayerFit, select_layer, sweep_layers


def _synthetic(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    states = rng.normal(size=(3, n, d)).astype(np.float16)
    states[1, :, 0] += (2.0 * y).astype(np.float16)  # layer 1 is informative
    states[2, :, 0] += (0.5 * y).astype(np.float16)  # layer 2 is weakly informative
    return states, y


def _sweep(states, y):
    idx = np.arange(len(y))
    tr, va, te = idx[:200], idx[200:250], idx[250:]
    return sweep_layers(states, tr, y[tr], va, y[va], te, C_grid=(0.01, 1.0), max_iter=1000, seed=0, log=lambda m: None)


def test_sweep_selects_informative_layer():
    states, y = _synthetic()
    fits = _sweep(states, y)
    assert [f.layer for f in fits] == [0, 1, 2]
    assert select_layer(fits) == 1
    assert fits[1].val_auroc > 0.85
    assert abs(fits[0].val_auroc - 0.5) < 0.25
    assert fits[1].val_scores.shape == (50,) and fits[1].eval_scores.shape == (50,)
    assert fits[1].coef.shape == (6,) and fits[1].C in (0.01, 1.0)
    assert np.all((fits[1].eval_scores >= 0) & (fits[1].eval_scores <= 1))


def test_sweep_is_deterministic():
    states, y = _synthetic()
    a, b = _sweep(states, y), _sweep(states, y)
    assert all(np.array_equal(x.eval_scores, z.eval_scores) for x, z in zip(a, b))


def test_select_layer_prefers_lower_layer_on_tie():
    def fit(layer, auc):
        return LayerFit(layer, 1.0, auc, np.zeros(1), np.zeros(1), np.zeros(1), 0.0)

    assert select_layer([fit(0, 0.6), fit(1, 0.8), fit(2, 0.8)]) == 1
