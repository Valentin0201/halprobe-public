"""Text normalization shared by data grouping and answer labeling."""

from __future__ import annotations

import re
import unicodedata

_ARTICLES = re.compile(r"\b(?:a|an|the)\b")


def normalize(text: str) -> str:
    """SQuAD-style normalization: strip accents, casefold, punctuation -> space, drop articles, collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contains_phrase(haystack: str, needle: str) -> bool:
    """Word-boundary containment of two normalized strings; an empty needle never matches."""
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "
