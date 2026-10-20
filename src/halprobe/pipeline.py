"""Stage orchestration with content-addressed caching and lazy model loading."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from halprobe import baselines as bl
from halprobe.cache import StageDir, atomic_write_json, read_jsonl, stage_key, write_jsonl
from halprobe.config import Config, to_plain
from halprobe.data import (
    SPLITS,
    Example,
    SplitView,
    build_examples,
    fetch_popqa,
    load_examples,
    load_popqa_rows,
    save_examples,
    split_views,
)
from halprobe.device import resolve_device, seed_everything
from halprobe.extraction import ARGMAX_FILE, open_states, run_extraction
from halprobe.generation import run_generation
from halprobe.labeling import RULES_VERSION, label_answer, pseudo_label
from halprobe.models import load_model, load_tokenizer, stop_token_ids
from halprobe.probes import select_layer, sweep_layers
from halprobe.prompts import render_prompt
from halprobe.report import build_report, check_consistency

STAGES = ("data", "generate", "label", "extract", "probe", "baselines", "report")
# Bump a stage's version when its code changes in a way that alters its outputs.
STAGE_VERSIONS = {"data": 1, "generate": 1, "label": RULES_VERSION, "extract": 1, "probe": 1, "baselines": 1}
POSITION = "last_prompt"
Log = Callable[[str], None]


def stage_keys(cfg: Config) -> dict[str, str]:
    plain = to_plain(cfg)
    model_prompt = {"model": plain["model"], "prompt": plain["prompt"]}
    v = STAGE_VERSIONS
    keys = {"data": stage_key({"stage": "data", "v": v["data"], "data": plain["data"], "seed": cfg.seed,
                               "shot_subjects": [s.subject for s in cfg.prompt.shots]})}
    keys["generate"] = stage_key({"stage": "generate", "v": v["generate"], "up": keys["data"], **model_prompt,
                                  "max_new_tokens": cfg.generation.max_new_tokens})
    keys["label"] = stage_key({"stage": "label", "v": v["label"], "up": keys["generate"], "labeling": plain["labeling"]})
    keys["extract"] = stage_key({"stage": "extract", "v": v["extract"], "up": keys["data"], **model_prompt,
                                 "position": POSITION})
    keys["probe"] = stage_key({"stage": "probe", "v": v["probe"], "up": [keys["extract"], keys["label"]],
                               "probe": plain["probe"], "seed": cfg.seed})
    keys["baselines"] = stage_key({"stage": "baselines", "v": v["baselines"], "up": [keys["generate"], keys["label"]],
                                   "seed": cfg.seed})
    return keys


def stage_dirs(cfg: Config, artifacts_root: Path | str) -> dict[str, StageDir]:
    return {name: StageDir(artifacts_root, name, key) for name, key in stage_keys(cfg).items()}


class _ModelHandle:
    """Loads the tokenizer/model only when a GPU stage actually has work to do."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.cfg.model)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.cfg.model, resolve_device(self.cfg.device, self.cfg.gpu_name_hint))
        return self._model


def _run_stage(sd: StageDir, build: Callable[[Path], None], log: Log) -> tuple[Path, bool]:
    if sd.is_complete():
        log(f"[{sd.stage}] cache hit {sd.key}")
        return sd.final, True
    log(f"[{sd.stage}] building {sd.key}")
    build(sd.begin())
    return sd.commit(), False


def _build_data(cfg: Config, out: Path) -> None:
    d = cfg.data
    path = Path(d.local_path) if d.local_path else fetch_popqa(d.repo_id, d.filename, d.revision)
    examples, report = build_examples(load_popqa_rows(path), d.split_ratios, cfg.seed,
                                      [s.subject for s in cfg.prompt.shots])
    save_examples(out / "examples.jsonl", examples)
    atomic_write_json(out / "data_report.json", report)


def _build_generate(cfg: Config, handle: _ModelHandle, examples: Sequence[Example], out: Path, log: Log) -> None:
    tok, model = handle.tokenizer, handle.model
    items = [(ex.id, render_prompt(tok, ex.question, cfg.prompt)) for ex in examples]
    backoffs: list[int] = []
    run_generation(model, tok, items, out / "generations.jsonl", cfg.generation.batch_size,
                   cfg.generation.max_new_tokens, stop_token_ids(model, tok), log=log, on_backoff=backoffs.append)
    atomic_write_json(out / "meta.json", {"batch_size": cfg.generation.batch_size, "oom_backoffs": backoffs,
                                          "device": str(model.device), "n": len(items)})


def _build_label(cfg: Config, examples: Sequence[Example], gens_by_id: dict, out: Path) -> None:
    rows = []
    for ex in examples:
        gen = gens_by_id[ex.id]
        result = (pseudo_label(ex.id) if cfg.labeling.mode == "pseudo"
                  else label_answer(gen["answer"], ex.answers, gen["finish_reason"]))
        rows.append({"id": ex.id, "label": result.label, "reason": result.reason, "y": result.y})
    write_jsonl(out / "labels.jsonl", rows)
    split_of = {ex.id: ex.split for ex in examples}
    counts = Counter(r["label"] for r in rows)
    decided = counts["correct"] + counts["incorrect"]
    atomic_write_json(out / "label_report.json", {
        "mode": cfg.labeling.mode, "counts": dict(counts),
        "ambiguous_reasons": dict(Counter(r["reason"] for r in rows if r["reason"])),
        "accuracy_unambiguous": counts["correct"] / decided if decided else None,
        "by_split": {s: dict(Counter(r["label"] for r in rows if split_of[r["id"]] == s)) for s in SPLITS},
    })


def _build_extract(cfg: Config, handle: _ModelHandle, examples: Sequence[Example], out: Path, log: Log) -> None:
    tok, model = handle.tokenizer, handle.model
    prompts = [render_prompt(tok, ex.question, cfg.prompt) for ex in examples]
    meta = {"model": cfg.model.name_or_path, "revision": cfg.model.revision, "dtype": cfg.model.dtype,
            "position": POSITION}
    run_extraction(model, tok, [ex.id for ex in examples], prompts, out, cfg.extraction.batch_size, meta, log=log)


def _check_class_counts(views: dict[str, SplitView], minimum: int) -> None:
    for split, view in views.items():
        n_pos = int(view.y.sum())
        n_neg = len(view.y) - n_pos
        if min(n_pos, n_neg) < minimum:
            raise RuntimeError(f"{split}: {n_pos} errors / {n_neg} correct labelled examples (< {minimum} per class)")


def _build_probe(cfg: Config, views: dict[str, SplitView], extract_dir: Path, out: Path, log: Log) -> None:
    _check_class_counts(views, cfg.probe.min_class_count)
    states, _ = open_states(extract_dir)
    train, val, test = views["train"], views["val"], views["test"]
    fits = sweep_layers(states, train.idx_lab, train.y, val.idx_lab, val.y, test.idx_all,
                        cfg.probe.C_grid, cfg.probe.max_iter, cfg.seed, log)
    del states
    np.save(out / "val_scores.npy", np.stack([f.val_scores for f in fits]))
    np.save(out / "test_scores.npy", np.stack([f.eval_scores for f in fits]))
    np.save(out / "coefs.npy", np.stack([f.coef for f in fits]))
    np.save(out / "intercepts.npy", np.array([f.intercept for f in fits]))
    atomic_write_json(out / "probe_report.json", {
        "layers": [{"layer": f.layer, "C": f.C, "val_auroc": f.val_auroc} for f in fits],
        "selected_layer": select_layer(fits),
    })


def _build_baselines(cfg: Config, examples: Sequence[Example], views: dict[str, SplitView], gens_by_id: dict,
                     out: Path) -> None:
    records = [gens_by_id[ex.id] for ex in examples]
    train = views["train"]
    raw = {"neg_mean_logprob": bl.fill_nan_as_max(bl.neg_mean_logprob(records)),
           "first_token_entropy": bl.first_token_entropy(records)}
    s_pop = np.array([ex.s_pop for ex in examples])
    relations = [ex.relation for ex in examples]
    vocabulary = sorted({relations[i] for i in train.idx_lab})
    arrays = {}
    for name, score in raw.items():
        arrays[f"score__{name}"] = score
        arrays[f"prob__{name}"] = bl.calibrate(score[train.idx_lab], train.y, score, cfg.seed)
    for name, X in (("popularity", bl.popularity_features(s_pop)),
                    ("metadata", bl.metadata_features(s_pop, relations, vocabulary))):
        (p,) = bl.fit_logreg(X[train.idx_lab], train.y, [X], cfg.seed)
        arrays[f"score__{name}"] = p
        arrays[f"prob__{name}"] = p
    np.savez(out / "scores.npz", **arrays)
    atomic_write_json(out / "baselines_report.json", {"relation_vocabulary": vocabulary})


def _new_run_dir(parent: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate, n = parent / stamp, 1
    while candidate.exists():
        candidate, n = parent / f"{stamp}-{n}", n + 1
    candidate.mkdir(parents=True)
    return candidate


@dataclass
class RunResult:
    run_dir: Path | None
    cache_hits: dict[str, bool]


def run(cfg: Config, artifacts_root: Path | str = Path("artifacts"), results_root: Path | str = Path("results"),
        until: str = "report", log: Log = print) -> RunResult:
    if until not in STAGES:
        raise ValueError(f"unknown stage {until!r}; expected one of {STAGES}")
    seed_everything(cfg.seed)
    dirs = stage_dirs(cfg, artifacts_root)
    handle = _ModelHandle(cfg)
    hits: dict[str, bool] = {}

    def stage(name: str, build: Callable[[Path], None]) -> Path:
        path, hits[name] = _run_stage(dirs[name], build, log)
        return path

    def stop_after(name: str) -> bool:
        return STAGES.index(name) >= STAGES.index(until)

    data_dir = stage("data", lambda out: _build_data(cfg, out))
    examples = load_examples(data_dir / "examples.jsonl")
    if stop_after("data"):
        return RunResult(None, hits)
    gen_dir = stage("generate", lambda out: _build_generate(cfg, handle, examples, out, log))
    gens_by_id = {r["id"]: r for r in read_jsonl(gen_dir / "generations.jsonl")}
    if stop_after("generate"):
        return RunResult(None, hits)
    label_dir = stage("label", lambda out: _build_label(cfg, examples, gens_by_id, out))
    labels_by_id = {r["id"]: r for r in read_jsonl(label_dir / "labels.jsonl")}
    if stop_after("label"):
        return RunResult(None, hits)
    extract_dir = stage("extract", lambda out: _build_extract(cfg, handle, examples, out, log))
    consistency = check_consistency(examples, gens_by_id, np.load(extract_dir / ARGMAX_FILE),
                                    cfg.extraction.consistency_threshold)
    log(f"[consistency] argmax == first generated token for {consistency['match_rate']:.4f} of examples")
    if stop_after("extract"):
        return RunResult(None, hits)
    views = split_views(examples, labels_by_id)
    probe_dir = stage("probe", lambda out: _build_probe(cfg, views, extract_dir, out, log))
    if stop_after("probe"):
        return RunResult(None, hits)
    baselines_dir = stage("baselines", lambda out: _build_baselines(cfg, examples, views, gens_by_id, out))
    if stop_after("baselines"):
        return RunResult(None, hits)
    run_dir = _new_run_dir(Path(results_root) / cfg.experiment)
    build_report(cfg, examples, labels_by_id, probe_dir, baselines_dir, consistency,
                 json.loads((data_dir / "data_report.json").read_text(encoding="utf-8")),
                 json.loads((label_dir / "label_report.json").read_text(encoding="utf-8")),
                 {name: sd.key for name, sd in dirs.items()}, run_dir)
    log(f"[report] wrote {run_dir}")
    return RunResult(run_dir, hits)
