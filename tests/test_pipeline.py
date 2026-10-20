import json
from dataclasses import replace

import yaml

from halprobe.__main__ import main
from halprobe.config import to_plain
from halprobe.data import load_examples
from halprobe.extraction import open_states
from halprobe.pipeline import STAGES, run, stage_dirs, stage_keys


def _quiet(_):
    pass


def test_stage_keys_cascade_only_where_relevant(tiny_config):
    base = stage_keys(tiny_config)
    probe_only = stage_keys(replace(tiny_config, probe=replace(tiny_config.probe, C_grid=(0.1,))))
    assert {k for k in base if base[k] != probe_only[k]} == {"probe"}
    batch = stage_keys(replace(tiny_config, generation=replace(tiny_config.generation, batch_size=4)))
    assert batch == base
    prompt = stage_keys(replace(tiny_config, prompt=replace(tiny_config.prompt, system="Other.")))
    assert {k for k in base if base[k] != prompt[k]} == {"generate", "label", "extract", "probe", "baselines"}
    seed = stage_keys(replace(tiny_config, seed=1))
    assert all(seed[k] != base[k] for k in base)


def test_end_to_end_tiny(tmp_path, tiny_config):
    artifacts, results = tmp_path / "artifacts", tmp_path / "results"
    first = run(tiny_config, artifacts, results, log=_quiet)
    assert first.cache_hits == {s: False for s in STAGES if s != "report"}
    metrics = json.loads((first.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["label_mode"] == "pseudo"
    assert set(metrics["methods"]) == {"probe", "probe_layer0", "first_token_entropy", "popularity", "metadata",
                                       "neg_mean_logprob"}
    assert all(0 <= m["auroc"] <= 1 for m in metrics["methods"].values())
    assert len(metrics["layer_sweep"]) == 3
    assert metrics["consistency"]["match_rate"] >= 0.9
    for name in ("layer_sweep.png", "reliability.png", "risk_coverage.png", "relation_auroc.png"):
        assert (first.run_dir / "figures" / name).stat().st_size > 0
    for name in ("config.yaml", "env.json", "stage_keys.json", "tables/summary.md"):
        assert (first.run_dir / name).exists()
    assert "| probe |" in (first.run_dir / "tables" / "summary.md").read_text(encoding="utf-8")

    dirs = stage_dirs(tiny_config, artifacts)
    n_examples = len(load_examples(dirs["data"].final / "examples.jsonl"))
    states, _ = open_states(dirs["extract"].final)
    assert states.shape == (3, n_examples, 64)
    del states

    second = run(tiny_config, artifacts, results, log=_quiet)
    assert all(second.cache_hits.values())
    assert second.run_dir != first.run_dir


def test_cli_runs_single_stage(tmp_path, tiny_config):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(to_plain(tiny_config)), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    assert main(["data", str(cfg_path), "--artifacts", str(artifacts), "--results", str(tmp_path / "r")]) == 0
    dirs = stage_dirs(tiny_config, artifacts)
    assert dirs["data"].is_complete() and not dirs["generate"].is_complete()


def test_cli_make_tiny(tmp_path):
    assert main(["make-tiny", str(tmp_path / "tiny")]) == 0
    assert (tmp_path / "tiny" / "popqa_tiny.tsv").exists() and (tmp_path / "tiny" / "model").is_dir()
