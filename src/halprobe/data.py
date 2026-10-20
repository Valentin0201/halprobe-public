"""PopQA loading, deduplication, entity-grouped splitting, and split index views."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from halprobe.cache import read_jsonl, write_jsonl
from halprobe.text import normalize

POPQA_COLUMNS = (
    "id", "subj", "prop", "obj", "subj_id", "prop_id", "obj_id", "s_aliases", "o_aliases", "s_uri", "o_uri",
    "s_wiki_title", "o_wiki_title", "s_pop", "o_pop", "question", "possible_answers",
)
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    subject: str
    relation: str
    answers: tuple[str, ...]
    s_pop: int
    split: str


def fetch_popqa(repo_id: str, filename: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision))


def load_popqa_rows(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = sorted(set(POPQA_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return list(reader)


def subject_key(subject: str) -> str:
    return normalize(subject)


def _question_key(question: str) -> str:
    return " ".join(question.casefold().split())


def dedupe(rows: Sequence[dict]) -> tuple[list[dict], dict]:
    """One example per question text. Same-entity duplicates are merged (union of answers);
    question texts shared by different entities (same name, different subj_id) are dropped."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_question_key(row["question"])].append(row)
    merged, n_merged, n_dropped = [], 0, 0
    for group in groups.values():
        if len({r["subj_id"] for r in group}) > 1:
            n_dropped += 1
            continue
        if len(group) > 1:
            n_merged += 1
        answers: list[str] = []
        for r in group:
            for a in json.loads(r["possible_answers"]):
                if a not in answers:
                    answers.append(a)
        first = group[0]
        merged.append({"id": first["id"], "question": first["question"].strip(), "subject": first["subj"].strip(),
                       "relation": first["prop"], "answers": tuple(answers), "s_pop": int(first["s_pop"])})
    report = {"rows_in": len(rows), "question_groups": len(groups), "groups_merged": n_merged,
              "groups_dropped_multi_entity": n_dropped, "examples_out": len(merged)}
    return merged, report


def assign_splits(group_keys: Sequence[str], ratios: Sequence[float], seed: int) -> list[str]:
    unique = sorted(set(group_keys))
    order = np.random.default_rng(seed).permutation(len(unique))
    n_train = round(ratios[0] * len(unique))
    n_val = round(ratios[1] * len(unique))
    split_of = {}
    for rank, idx in enumerate(order):
        split_of[unique[idx]] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    return [split_of[k] for k in group_keys]


def check_no_leakage(examples: Iterable[Example]) -> None:
    seen: dict[str, str] = {}
    for ex in examples:
        key = subject_key(ex.subject)
        previous = seen.setdefault(key, ex.split)
        if previous != ex.split:
            raise AssertionError(f"subject {key!r} appears in both {previous} and {ex.split}")


def check_shot_subjects(subjects: Iterable[str], shot_subjects: Iterable[str]) -> None:
    present = {subject_key(s) for s in subjects}
    clashes = [s for s in shot_subjects if subject_key(s) in present]
    if clashes:
        raise ValueError(f"few-shot subjects occur in the dataset: {clashes}")


def build_examples(rows: Sequence[dict], ratios: Sequence[float], seed: int,
                   shot_subjects: Sequence[str]) -> tuple[list[Example], dict]:
    check_shot_subjects((r["subj"] for r in rows), shot_subjects)
    merged, report = dedupe(rows)
    splits = assign_splits([subject_key(m["subject"]) for m in merged], ratios, seed)
    examples = [Example(**m, split=s) for m, s in zip(merged, splits)]
    check_no_leakage(examples)
    report["subject_groups"] = len({subject_key(ex.subject) for ex in examples})
    report["split_sizes"] = dict(Counter(ex.split for ex in examples))
    report["relations_by_split"] = {s: dict(Counter(ex.relation for ex in examples if ex.split == s)) for s in SPLITS}
    return examples, report


def save_examples(path: Path | str, examples: Iterable[Example]) -> None:
    write_jsonl(path, [asdict(ex) for ex in examples])


def load_examples(path: Path | str) -> list[Example]:
    return [Example(**{**row, "answers": tuple(row["answers"])}) for row in read_jsonl(path)]


@dataclass(frozen=True)
class SplitView:
    idx_all: np.ndarray  # row indices (into the examples list) of every example in the split
    idx_lab: np.ndarray  # subset with an unambiguous label
    y: np.ndarray  # labels for idx_lab (1 = error)
    pos_lab: np.ndarray  # positions of idx_lab within idx_all


def split_views(examples: Sequence[Example], labels_by_id: dict[str, dict]) -> dict[str, SplitView]:
    views = {}
    for split in SPLITS:
        idx_all = np.array([i for i, ex in enumerate(examples) if ex.split == split], dtype=np.int64)
        ys = [labels_by_id[examples[i].id]["y"] for i in idx_all]
        pos_lab = np.array([p for p, y in enumerate(ys) if y is not None], dtype=np.int64)
        views[split] = SplitView(idx_all=idx_all, idx_lab=idx_all[pos_lab],
                                 y=np.array([ys[p] for p in pos_lab], dtype=np.int64), pos_lab=pos_lab)
    return views
