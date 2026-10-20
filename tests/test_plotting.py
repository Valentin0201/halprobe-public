import numpy as np

from halprobe.metrics import reliability_bins, risk_coverage
from halprobe.plotting import plot_layer_sweep, plot_relation_auroc, plot_reliability, plot_risk_coverage


def _png(path):
    return path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_all_figures_render(tmp_path):
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 100), rng.random(100)
    plot_layer_sweep([0, 1, 2], [0.5, 0.7, 0.65], [0.52, 0.68, 0.66], [(0.45, 0.6), (0.6, 0.75), (0.6, 0.72)],
                     {"metadata": 0.62, "neg_mean_logprob": 0.7}, 1, tmp_path / "sweep.png")
    plot_reliability({"probe": reliability_bins(y, p, 10)}, tmp_path / "rel.png")
    plot_risk_coverage({"probe": risk_coverage(y, p)}, float(y.mean()), tmp_path / "rc.png")
    plot_relation_auroc({"probe": {"capital": 0.7, "genre": 0.6}, "metadata": {"capital": 0.5}}, tmp_path / "relations.png")
    assert all(_png(tmp_path / f) for f in ("sweep.png", "rel.png", "rc.png", "relations.png"))


def test_relation_plot_with_no_relations(tmp_path):
    plot_relation_auroc({"probe": {}}, tmp_path / "empty.png")
    assert _png(tmp_path / "empty.png")
