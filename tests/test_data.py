import csv
import json
from collections import Counter

import numpy as np
import pytest

from halprobe.data import (
    POPQA_COLUMNS,
    Example,
    assign_splits,
    build_examples,
    check_no_leakage,
    check_shot_subjects,
    dedupe,
    load_examples,
    load_popqa_rows,
    save_examples,
    split_views,
)


def _row(id, subj, subj_id, prop, question, answers, s_pop=100):
    return {"id": str(id), "subj": subj, "subj_id": str(subj_id), "prop": prop, "question": question,
            "possible_answers": json.dumps(answers), "s_pop": str(s_pop)}


def test_load_popqa_rows_checks_columns(tmp_path):
    good = tmp_path / "good.tsv"
    with open(good, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POPQA_COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerow({c: "1" for c in POPQA_COLUMNS})
    assert len(load_popqa_rows(good)) == 1
    bad = tmp_path / "bad.tsv"
    bad.write_text("id\tsubj\n1\tx\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_popqa_rows(bad)


def test_dedupe_merges_same_entity_duplicates():
    rows = [
        _row(1, "Taxi", 10, "genre", "What genre is Taxi?", ["comedy film"]),
        _row(2, "Taxi", 10, "genre", "What genre is Taxi? ", ["action film", "comedy film"]),
    ]
    merged, report = dedupe(rows)
    assert len(merged) == 1
    assert merged[0]["answers"] == ("comedy film", "action film")
    assert merged[0]["id"] == "1"
    assert report["groups_merged"] == 1 and report["groups_dropped_multi_entity"] == 0


def test_dedupe_drops_same_name_different_entities():
    rows = [
        _row(1, "Will Smith", 1, "occupation", "What is Will Smith's occupation?", ["film producer"], 345519),
        _row(2, "Will Smith", 2, "occupation", "What is Will Smith's occupation?", ["screenwriter"], 5209),
        _row(3, "Paris", 3, "country", "In what country is Paris?", ["France"]),
    ]
    merged, report = dedupe(rows)
    assert [m["question"] for m in merged] == ["In what country is Paris?"]
    assert report == {"rows_in": 3, "question_groups": 2, "groups_merged": 0,
                      "groups_dropped_multi_entity": 1, "examples_out": 1}


def test_assign_splits_is_deterministic_and_matches_ratios():
    keys = [f"s{i}" for i in range(1000)]
    a = assign_splits(keys, (0.7, 0.15, 0.15), seed=0)
    assert a == assign_splits(keys, (0.7, 0.15, 0.15), seed=0)
    assert a != assign_splits(keys, (0.7, 0.15, 0.15), seed=1)
    assert Counter(a) == {"train": 700, "val": 150, "test": 150}


def test_same_subject_string_stays_in_one_split():
    rows = [_row(i, f"Subject {i % 50}", 1000 + i, "occupation", f"Question {i}?", ["x"]) for i in range(300)]
    rows.append(_row(999, "Subject 3", 77, "country", "Where is Subject 3?", ["y"]))  # same name, other entity
    examples, _ = build_examples(rows, (0.7, 0.15, 0.15), seed=0, shot_subjects=[])
    by_subject = {}
    for ex in examples:
        by_subject.setdefault(ex.subject, set()).add(ex.split)
    assert all(len(splits) == 1 for splits in by_subject.values())


def test_check_no_leakage_raises():
    examples = [
        Example("1", "q1", "The Beatles", "genre", ("rock",), 1, "train"),
        Example("2", "q2", "Beatles", "genre", ("rock",), 1, "test"),
    ]
    with pytest.raises(AssertionError, match="beatles"):
        check_no_leakage(examples)


def test_shot_subject_guard():
    check_shot_subjects({"Paris", "Berlin"}, ["Ada Lovelace"])
    with pytest.raises(ValueError, match="paris"):
        check_shot_subjects({"Paris"}, ["paris"])


def test_build_examples_report_and_roundtrip(tmp_path):
    rows = [_row(i, f"S{i}", i, "genre" if i % 2 else "capital", f"Q{i}?", [f"a{i}"], 10 * i + 2) for i in range(40)]
    examples, report = build_examples(rows, (0.7, 0.15, 0.15), seed=0, shot_subjects=["Ada Lovelace"])
    assert report["examples_out"] == 40
    assert sum(report["split_sizes"].values()) == 40
    assert set(report["relations_by_split"]) == {"train", "val", "test"}
    save_examples(tmp_path / "e.jsonl", examples)
    assert load_examples(tmp_path / "e.jsonl") == examples


def test_split_views_exclude_ambiguous():
    examples = [
        Example("a", "q", "s1", "r", ("x",), 1, "test"),
        Example("b", "q", "s2", "r", ("x",), 1, "train"),
        Example("c", "q", "s3", "r", ("x",), 1, "test"),
        Example("d", "q", "s4", "r", ("x",), 1, "test"),
    ]
    labels = {"a": {"y": 1}, "b": {"y": 0}, "c": {"y": None}, "d": {"y": 0}}
    views = split_views(examples, labels)
    test = views["test"]
    assert test.idx_all.tolist() == [0, 2, 3]
    assert test.idx_lab.tolist() == [0, 3]
    assert test.y.tolist() == [1, 0]
    assert test.pos_lab.tolist() == [0, 2]
    assert views["val"].idx_all.size == 0 and views["val"].y.dtype == np.int64
