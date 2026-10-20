"""Manual label review: export random samples, score filled-in sheets."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from halprobe.data import Example

REVIEW_COLUMNS = ("id", "split", "relation", "question", "answer", "gold_answers", "auto_label", "reason",
                  "manual_label")
MANUAL_LABELS = ("correct", "incorrect", "ambiguous")


def review_rows(examples: Sequence[Example], gens_by_id: dict, labels_by_id: dict) -> list[dict]:
    return [{"id": ex.id, "split": ex.split, "relation": ex.relation, "question": ex.question,
             "answer": gens_by_id[ex.id]["answer"], "gold_answers": " | ".join(ex.answers),
             "auto_label": labels_by_id[ex.id]["label"], "reason": labels_by_id[ex.id]["reason"] or "",
             "manual_label": ""} for ex in examples]


def sample_for_review(rows: Sequence[dict], n: int, seed: int) -> list[dict]:
    """n seeded random rows among the unambiguous automatic labels (estimates their noise rate)."""
    pool = [r for r in rows if r["auto_label"] != "ambiguous"]
    if not pool:
        return []
    idx = np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


def write_review_csv(path: Path | str, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel shows umlauts correctly
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def score_review(path: Path | str) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        f.seek(0)
        delimiter = ";" if header.count(";") > header.count(",") else ","  # German Excel saves with ';'
        rows = list(csv.DictReader(f, delimiter=delimiter))
    manual = [r["manual_label"].strip().lower() for r in rows]
    blanks = [r["id"] for r, m in zip(rows, manual) if not m]
    if blanks:
        raise ValueError(f"{len(blanks)} row(s) have no manual_label, e.g. {blanks[:5]}")
    invalid = sorted({m for m in manual if m not in MANUAL_LABELS})
    if invalid:
        raise ValueError(f"invalid manual_label value(s) {invalid}; use one of {MANUAL_LABELS}")
    judged = [(r, m) for r, m in zip(rows, manual) if m != "ambiguous"]
    disagree = [r["id"] for r, m in judged if m != r["auto_label"]]
    return {"n_rows": len(rows), "n_judged": len(judged), "n_manual_ambiguous": len(rows) - len(judged),
            "n_disagree": len(disagree), "noise_rate": len(disagree) / len(judged) if judged else None,
            "confusion": dict(Counter(f"{r['auto_label']}->{m}" for r, m in zip(rows, manual))),
            "disagreeing_ids": disagree}
