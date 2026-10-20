import csv

import pytest

from halprobe.data import Example
from halprobe.review import REVIEW_COLUMNS, review_rows, sample_for_review, score_review, write_review_csv


def _rows(n=30):
    examples = [Example(str(i), f"Q{i}?", f"S{i}", "genre", ("rock", "rock music"), 10, "test") for i in range(n)]
    gens = {str(i): {"answer": f"answer {i}"} for i in range(n)}
    labels = {str(i): {"label": ["correct", "incorrect", "ambiguous"][i % 3], "reason": "partial" if i % 3 == 2 else None}
              for i in range(n)}
    return review_rows(examples, gens, labels)


def test_review_rows_content():
    row = _rows()[2]
    assert row["gold_answers"] == "rock | rock music"
    assert row["auto_label"] == "ambiguous" and row["reason"] == "partial" and row["manual_label"] == ""
    assert list(row) == list(REVIEW_COLUMNS)


def test_sample_excludes_ambiguous_and_is_deterministic():
    rows = _rows()
    sample = sample_for_review(rows, 5, seed=0)
    assert len(sample) == 5 and all(r["auto_label"] != "ambiguous" for r in sample)
    assert sample == sample_for_review(rows, 5, seed=0)
    assert len(sample_for_review(rows, 1000, seed=0)) == 20


def _fill(path, labels, delimiter=","):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row, label in zip(rows, labels):
        row["manual_label"] = label
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_score_review_noise_rate(tmp_path):
    path = tmp_path / "review.csv"
    rows = [r for r in _rows(6) if r["auto_label"] != "ambiguous"]  # correct, incorrect, correct, incorrect
    write_review_csv(path, rows)
    _fill(path, ["correct", "correct", "ambiguous", "incorrect"])
    result = score_review(path)
    assert result["n_rows"] == 4 and result["n_judged"] == 3 and result["n_manual_ambiguous"] == 1
    assert result["n_disagree"] == 1 and result["noise_rate"] == pytest.approx(1 / 3)
    assert result["disagreeing_ids"] == ["1"]


def test_score_review_accepts_semicolon_csv_from_german_excel(tmp_path):
    path = tmp_path / "review.csv"
    write_review_csv(path, [r for r in _rows(3) if r["auto_label"] != "ambiguous"])
    _fill(path, ["Correct", "incorrect"], delimiter=";")
    assert score_review(path)["n_disagree"] == 0


def test_score_review_rejects_blank_and_invalid(tmp_path):
    path = tmp_path / "review.csv"
    write_review_csv(path, [r for r in _rows(3) if r["auto_label"] != "ambiguous"])
    with pytest.raises(ValueError, match="no manual_label"):
        score_review(path)
    _fill(path, ["correct", "maybe"])
    with pytest.raises(ValueError, match="maybe"):
        score_review(path)
