import json

import numpy as np
import pytest

from halprobe.data import Example
from halprobe.report import (
    check_consistency,
    format_sensitivity_table,
    format_shortcut_table,
    format_summary_table,
    render_results,
    replace_block,
)


def _metrics(label_mode="alias"):
    method = {"timing": "pre-generation", "n": 100, "base_rate": 0.7, "auroc": 0.75, "auroc_ci": [0.7, 0.8],
              "auprc": 0.85, "auprc_ci": [0.8, 0.9], "ece": 0.05, "aurc": 0.5, "risk_at_50": 0.4, "risk_at_80": 0.6}
    return {
        "run_id": "20261012-120000", "label_mode": label_mode, "selected_layer": 14,
        "methods": {"probe": method, "metadata": {**method, "auroc": 0.66}},
        "shortcut": {"within_relation": {"probe": {"macro": 0.7, "n_relations": 12},
                                         "metadata": {"macro": None, "n_relations": 0}},
                     "incremental": {"auroc_metadata": 0.66, "auroc_combined": 0.77, "delta": 0.11,
                                     "delta_ci": [0.08, 0.14]}},
        "sensitivity": {"n_ambiguous": 30, "ambiguous_as_correct": {"probe": 0.73, "metadata": 0.65},
                        "ambiguous_as_incorrect": {"probe": 0.74, "metadata": 0.64}},
    }


def test_check_consistency_pass_and_fail():
    examples = [Example(str(i), "q", f"s{i}", "r", ("a",), 1, "test") for i in range(4)]
    gens = {str(i): {"token_ids": [i + 10, 0]} for i in range(4)}
    ok = check_consistency(examples, gens, np.array([10, 11, 12, 99]), threshold=0.75)
    assert ok["match_rate"] == 0.75 and ok["mismatched_ids"] == ["3"]
    with pytest.raises(RuntimeError, match="consistency"):
        check_consistency(examples, gens, np.array([10, 11, 98, 99]), threshold=0.75)


def test_format_tables():
    m = _metrics()
    summary = format_summary_table(m)
    assert "| probe | pre-generation | 0.750 [0.700, 0.800] | 0.850 [0.800, 0.900] | 0.050 | 0.500 | 0.600 |" in summary
    assert "layer 14" in summary and "n = 100" in summary
    assert "| metadata | n/a | 0 |" in format_shortcut_table(m)
    assert "0.110 [0.080, 0.140]" in format_shortcut_table(m)
    assert "| probe | 0.750 | 0.730 | 0.740 |" in format_sensitivity_table(m)


def test_replace_block():
    text = "a\n<!-- results:start -->\nold\n<!-- results:end -->\nb\n"
    assert replace_block(text, "results", "new") == "a\n<!-- results:start -->\nnew\n<!-- results:end -->\nb\n"
    with pytest.raises(ValueError, match="other"):
        replace_block(text, "other", "x")


def _write_run(tmp_path, metrics):
    run_dir = tmp_path / "results" / "exp" / metrics["run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- results:start -->\n<!-- results:end -->\n", encoding="utf-8")
    results_md = tmp_path / "results.md"
    results_md.write_text("".join(f"<!-- {n}:start -->\n<!-- {n}:end -->\n"
                                  for n in ("run", "summary", "shortcut", "sensitivity")), encoding="utf-8")
    return run_dir, readme, results_md


def test_render_results_writes_blocks(tmp_path):
    run_dir, readme, results_md = _write_run(tmp_path, _metrics())
    render_results(run_dir, readme, results_md)
    assert "| probe | pre-generation | 0.750" in readme.read_text(encoding="utf-8")
    text = results_md.read_text(encoding="utf-8")
    assert "20261012-120000" in text and "0.110 [0.080, 0.140]" in text


def test_render_results_refuses_pseudo_labels(tmp_path):
    run_dir, readme, results_md = _write_run(tmp_path, _metrics(label_mode="pseudo"))
    with pytest.raises(ValueError, match="pseudo"):
        render_results(run_dir, readme, results_md)
