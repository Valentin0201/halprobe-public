"""Run report: evaluation of all methods, shortcut/sensitivity analyses, figures, tables, provenance."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from halprobe.analysis import incremental_value, label_sensitivity, popularity_quintile_auroc, within_relation_auroc
from halprobe.cache import atomic_write_json, atomic_write_text
from halprobe.config import Config, to_plain
from halprobe.data import Example, split_views
from halprobe.metrics import auroc, bootstrap_ci, risk_coverage, summarize
from halprobe.plotting import plot_layer_sweep, plot_relation_auroc, plot_reliability, plot_risk_coverage

METHOD_TIMING = {
    "probe": "pre-generation",
    "probe_layer0": "pre-generation",
    "first_token_entropy": "pre-generation",
    "popularity": "pre-generation",
    "metadata": "pre-generation",
    "neg_mean_logprob": "post-generation",
}
BASELINE_NAMES = ("first_token_entropy", "popularity", "metadata", "neg_mean_logprob")


def check_consistency(examples: Sequence[Example], gens_by_id: dict, argmax: np.ndarray, threshold: float) -> dict:
    """The extraction pass's argmax at the last prompt token must equal the first generated token."""
    first = np.array([gens_by_id[ex.id]["token_ids"][0] for ex in examples])
    match = first == np.asarray(argmax)
    rate = float(match.mean())
    result = {"match_rate": rate, "n": int(len(match)),
              "mismatched_ids": [ex.id for ex, ok in zip(examples, match) if not ok][:50]}
    if rate < threshold:
        raise RuntimeError(f"consistency check failed: argmax matches first generated token for {rate:.4f} "
                           f"< {threshold} of examples; padding/position bug? first mismatches: {result['mismatched_ids'][:5]}")
    return result


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info(cfg: Config) -> dict:
    import sklearn
    import torch
    import transformers

    from halprobe.device import resolve_device

    try:
        device = resolve_device(cfg.device, cfg.gpu_name_hint)
        device_name = "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device)
    except (RuntimeError, ValueError):
        device_name = None
    status = _git("status", "--porcelain")
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "hip": torch.version.hip, "cuda": torch.version.cuda, "device": device_name,
            "transformers": transformers.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def _fmt(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_ci(value, ci) -> str:
    return f"{_fmt(value)} [{_fmt(ci[0])}, {_fmt(ci[1])}]"


def format_summary_table(metrics: dict) -> str:
    lines = ["| Method | Timing | AUROC [95% CI] | AUPRC [95% CI] | ECE | AURC | Risk@80% |",
             "|---|---|---|---|---|---|---|"]
    for name, m in metrics["methods"].items():
        lines.append(f"| {name} | {m['timing']} | {_fmt_ci(m['auroc'], m['auroc_ci'])} | "
                     f"{_fmt_ci(m['auprc'], m['auprc_ci'])} | {_fmt(m.get('ece'))} | {_fmt(m['aurc'])} | "
                     f"{_fmt(m['risk_at_80'])} |")
    first = next(iter(metrics["methods"].values()))
    lines += ["", f"Test set: n = {first['n']}, error base rate = {_fmt(first['base_rate'])}. "
                  f"Probe uses layer {metrics['selected_layer']} (selected on validation AUROC)."]
    return "\n".join(lines)


def format_shortcut_table(metrics: dict) -> str:
    shortcut = metrics["shortcut"]
    lines = ["| Method | Within-relation AUROC (macro) | Relations used |", "|---|---|---|"]
    for name, w in shortcut["within_relation"].items():
        lines.append(f"| {name} | {_fmt(w['macro'])} | {w['n_relations']} |")
    inc = shortcut["incremental"]
    lines += ["", f"Metadata alone: AUROC {_fmt(inc['auroc_metadata'])}; metadata + probe: "
                  f"{_fmt(inc['auroc_combined'])}; ΔAUROC {_fmt_ci(inc['delta'], inc['delta_ci'])}."]
    return "\n".join(lines)


def format_sensitivity_table(metrics: dict) -> str:
    sens = metrics["sensitivity"]
    lines = [f"Ambiguous test examples: {sens['n_ambiguous']}.", "",
             "| Method | Headline AUROC | Ambiguous = correct | Ambiguous = incorrect |", "|---|---|---|---|"]
    for name, m in metrics["methods"].items():
        lines.append(f"| {name} | {_fmt(m['auroc'])} | {_fmt(sens['ambiguous_as_correct'][name])} | "
                     f"{_fmt(sens['ambiguous_as_incorrect'][name])} |")
    return "\n".join(lines)


def replace_block(text: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        raise ValueError(f"markers for block {name!r} not found")
    return text[: i + len(start)] + "\n" + content.strip() + "\n" + text[j:]


def build_report(cfg: Config, examples: Sequence[Example], labels_by_id: dict, probe_dir: Path, baselines_dir: Path,
                 consistency: dict, data_report: dict, label_report: dict, keys: dict, run_dir: Path) -> dict:
    ev, seed = cfg.evaluation, cfg.seed
    views = split_views(examples, labels_by_id)
    test, val = views["test"], views["val"]
    y = test.y
    probe_report = json.loads((Path(probe_dir) / "probe_report.json").read_text(encoding="utf-8"))
    selected = probe_report["selected_layer"]
    test_scores = np.load(Path(probe_dir) / "test_scores.npy")
    val_scores = np.load(Path(probe_dir) / "val_scores.npy")
    baseline_scores = np.load(Path(baselines_dir) / "scores.npz")

    score_all = {"probe": test_scores[selected], "probe_layer0": test_scores[0]}  # over test.idx_all
    prob_all = dict(score_all)
    for name in BASELINE_NAMES:
        score_all[name] = baseline_scores[f"score__{name}"][test.idx_all]
        prob_all[name] = baseline_scores[f"prob__{name}"][test.idx_all]
    labelled = {name: score_all[name][test.pos_lab] for name in METHOD_TIMING}

    methods = {name: {"timing": timing, **summarize(y, labelled[name], prob_all[name][test.pos_lab],
                                                    n_boot=ev.n_bootstrap, seed=seed, ece_bins=ev.ece_bins)}
               for name, timing in METHOD_TIMING.items()}
    sweep = []
    for info in probe_report["layers"]:
        s = test_scores[info["layer"]][test.pos_lab]
        sweep.append({**info, "test_auroc": auroc(y, s),
                      "test_auroc_ci": list(bootstrap_ci(auroc, y, s, ev.n_bootstrap, seed))})

    relations = np.array([examples[i].relation for i in test.idx_lab])
    s_pop = np.array([examples[i].s_pop for i in test.idx_lab])
    shortcut = {
        "within_relation": {n: within_relation_auroc(y, s, relations, ev.min_per_class_relation)
                            for n, s in labelled.items()},
        "popularity_quintiles": {n: popularity_quintile_auroc(y, s, s_pop) for n, s in labelled.items()},
        "incremental": incremental_value(baseline_scores["score__metadata"][val.idx_lab], val_scores[selected], val.y,
                                         labelled["metadata"], labelled["probe"], y, n_boot=ev.n_bootstrap, seed=seed),
    }
    test_labels = [labels_by_id[examples[i].id]["label"] for i in test.idx_all]
    metrics = {
        "experiment": cfg.experiment, "run_id": Path(run_dir).name, "label_mode": cfg.labeling.mode,
        "data": data_report, "labels": label_report, "consistency": consistency, "selected_layer": selected,
        "layer_sweep": sweep, "methods": methods, "shortcut": shortcut,
        "sensitivity": label_sensitivity(test_labels, score_all),
    }

    figures = Path(run_dir) / "figures"
    plot_layer_sweep([r["layer"] for r in sweep], [r["val_auroc"] for r in sweep], [r["test_auroc"] for r in sweep],
                     [r["test_auroc_ci"] for r in sweep], {n: methods[n]["auroc"] for n in BASELINE_NAMES},
                     selected, figures / "layer_sweep.png")
    plot_reliability({n: m["reliability"] for n, m in methods.items()}, figures / "reliability.png")
    plot_risk_coverage({n: risk_coverage(y, s) for n, s in labelled.items()}, float(y.mean()),
                       figures / "risk_coverage.png")
    plot_relation_auroc({n: shortcut["within_relation"][n]["per_relation"]
                         for n in ("probe", "metadata", "first_token_entropy")}, figures / "relation_auroc.png")

    atomic_write_text(Path(run_dir) / "tables" / "summary.md", "\n\n".join(
        [format_summary_table(metrics), format_shortcut_table(metrics), format_sensitivity_table(metrics)]) + "\n")
    atomic_write_json(Path(run_dir) / "metrics.json", metrics)
    atomic_write_text(Path(run_dir) / "config.yaml", yaml.safe_dump(to_plain(cfg), sort_keys=False, allow_unicode=True))
    atomic_write_json(Path(run_dir) / "env.json", environment_info(cfg))
    atomic_write_json(Path(run_dir) / "stage_keys.json", keys)
    return metrics


def render_results(run_dir: Path, readme: Path, results_md: Path) -> None:
    """Write result tables from run_dir/metrics.json into README.md and docs/results.md marker blocks."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("label_mode") != "alias":
        raise ValueError(f"{run_dir}: label_mode={metrics.get('label_mode')!r} (pseudo labels); "
                         "only alias-labelled runs may be reported")
    rel = run_dir.as_posix()
    head = f"Run `{metrics['run_id']}` · [metrics]({rel}/metrics.json) · [figures]({rel}/figures/)\n\n"
    readme = Path(readme)
    readme.write_text(replace_block(readme.read_text(encoding="utf-8"), "results", head + format_summary_table(metrics)),
                      encoding="utf-8")
    results_md = Path(results_md)
    text = results_md.read_text(encoding="utf-8")
    blocks = {
        "run": f"All numbers below come from run `{metrics['run_id']}` ([metrics.json](../{rel}/metrics.json)).",
        "summary": format_summary_table(metrics),
        "shortcut": format_shortcut_table(metrics),
        "sensitivity": format_sensitivity_table(metrics),
    }
    for name, content in blocks.items():
        text = replace_block(text, name, content)
    results_md.write_text(text, encoding="utf-8")
