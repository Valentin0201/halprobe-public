"""Correct / incorrect / ambiguous labels from normalized alias matching."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from halprobe.text import contains_phrase, normalize

# Bump when the rules below change: it is part of the label stage's cache key.
RULES_VERSION = 2

_REFUSALS_RAW = (
    "i don't know", "i do not know", "unknown", "not sure", "unsure", "cannot", "can't",
    "no information", "not available", "unclear", "no answer",
    # v2: phrasings Llama-3.2-3B-Instruct uses when it declines to answer.
    "couldn't find", "could not find", "don't have", "do not have", "not aware of", "not familiar with",
    "no record", "not enough information", "unable to",
)
REFUSALS = tuple(normalize(p) for p in _REFUSALS_RAW)
STOPWORDS = frozenset({"of", "and", "or", "in", "on", "at", "to", "for", "by", "with", "from", "is", "was", "as", "s"})
# Category words shared by many unrelated gold answers ("pop music" vs "classical music").
GENERIC_WORDS = frozenset({
    "music", "film", "movie", "player", "series", "game", "album", "novel", "band", "song", "show",
    "television", "tv", "city", "country", "state", "province", "district", "county", "language",
    "fiction", "drama",
})


@dataclass(frozen=True)
class LabelResult:
    label: str  # "correct" | "incorrect" | "ambiguous"
    reason: str | None  # set only for ambiguous: empty | refusal | truncated | partial

    @property
    def y(self) -> int | None:
        """Probe target: 1 = incorrect (error), 0 = correct, None = ambiguous."""
        return {"correct": 0, "incorrect": 1}.get(self.label)


def _content_words(normalized: str) -> set[str]:
    return set(normalized.split()) - STOPWORDS - GENERIC_WORDS


def label_answer(answer: str, aliases: Iterable[str], finish_reason: str) -> LabelResult:
    ans = normalize(answer)
    gold = [g for g in (normalize(a) for a in aliases) if g]
    if any(contains_phrase(ans, g) for g in gold):
        return LabelResult("correct", None)
    if not ans:
        return LabelResult("ambiguous", "empty")
    if any(contains_phrase(ans, r) for r in REFUSALS):
        return LabelResult("ambiguous", "refusal")
    if finish_reason == "length":
        return LabelResult("ambiguous", "truncated")
    words = _content_words(ans)
    if any(words & _content_words(g) for g in gold):
        return LabelResult("ambiguous", "partial")
    return LabelResult("incorrect", None)


def pseudo_label(example_id: str) -> LabelResult:
    """TEST-ONLY deterministic labels for random tiny models. Never used for reported results."""
    bit = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest(), 16) % 2
    return LabelResult("incorrect" if bit else "correct", None)
