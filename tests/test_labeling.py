import pytest

from halprobe.labeling import LabelResult, label_answer, pseudo_label
from halprobe.text import contains_phrase, normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  The Émile-Zola  Street! ", "emile zola street"),
        ("Emily Brontë", "emily bronte"),
        ("A Tale of Two Cities", "tale of two cities"),
        ("polit.", "polit"),
        ("Don't", "don t"),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_contains_phrase_respects_word_boundaries():
    assert contains_phrase("he is a pol", "pol")
    assert not contains_phrase("police officer", "pol")
    assert contains_phrase("steven spielberg directed jaws", "steven spielberg")
    assert not contains_phrase("anything", "")


def test_correct_when_alias_contained_in_sentence():
    assert label_answer("Steven Spielberg directed Jaws.", ["Steven Spielberg"], "stop") == LabelResult("correct", None)


def test_correct_with_accents_and_case():
    assert label_answer("emily bronte", ["Emily Brontë"], "stop").label == "correct"


def test_short_alias_does_not_match_inside_word():
    result = label_answer("police officer", ["politician", "pol"], "stop")
    assert result == LabelResult("incorrect", None)


def test_punctuation_only_alias_never_matches_everything():
    assert label_answer("anything at all", ["."], "stop").label == "incorrect"


def test_empty_answer_is_ambiguous_even_if_truncated():
    assert label_answer("   ", ["Paris"], "length") == LabelResult("ambiguous", "empty")


@pytest.mark.parametrize("answer", [
    "I don't know.", "Unknown", "I'm not sure", "I cannot answer that",
    # Phrasings Llama-3.2-3B-Instruct actually uses on PopQA:
    "I couldn't find any information on Azharullah.", "I don't have information on Pathit Pisitkul.",
    "I'm not aware of any information about Storm Master.", "I'm not familiar with that title.",
    "I'm sorry, I don't have enough information.", "I could not find a record of that.",
    "I'm unable to verify that.",
])
def test_refusals_are_ambiguous(answer):
    assert label_answer(answer, ["Paris"], "stop") == LabelResult("ambiguous", "refusal")


def test_truncated_without_match_is_ambiguous():
    assert label_answer("a very long rambling answer", ["Paris"], "length") == LabelResult("ambiguous", "truncated")


def test_correct_takes_precedence_over_truncation():
    assert label_answer("Paris is the city that", ["Paris"], "length").label == "correct"


def test_partial_overlap_is_ambiguous():
    assert label_answer("producer", ["film producer"], "stop") == LabelResult("ambiguous", "partial")


def test_shared_generic_word_is_not_partial():
    assert label_answer("classical music", ["pop music"], "stop") == LabelResult("incorrect", None)


def test_refusal_wording_inside_correct_answer_stays_correct():
    assert label_answer("I couldn't find much, but Paris.", ["Paris"], "stop").label == "correct"


def test_y_encoding():
    assert LabelResult("incorrect", None).y == 1
    assert LabelResult("correct", None).y == 0
    assert LabelResult("ambiguous", "empty").y is None


def test_pseudo_label_is_deterministic_and_uses_both_classes():
    labels = [pseudo_label(f"id{i}").label for i in range(50)]
    assert labels == [pseudo_label(f"id{i}").label for i in range(50)]
    assert set(labels) == {"correct", "incorrect"}
