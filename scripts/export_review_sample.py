"""Export labelled examples for manual review.

Usage: uv run python scripts/export_review_sample.py configs/mvp_popqa_llama3b.yaml [--n 100] [--seed 0]
Writes results/review/<experiment>/review_sample.csv (n random unambiguous examples) and ambiguous.csv.
Fill the manual_label column with correct / incorrect / ambiguous, then run scripts/score_review.py.
"""

import argparse
from pathlib import Path

from halprobe.cache import read_jsonl
from halprobe.config import load_config
from halprobe.data import load_examples
from halprobe.pipeline import stage_dirs
from halprobe.review import review_rows, sample_for_review, write_review_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export a manual review sample.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("results/review"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite an existing (possibly filled-in) sample")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dirs = stage_dirs(cfg, args.artifacts)
    for name in ("data", "generate", "label"):
        if not dirs[name].is_complete():
            raise SystemExit(f"stage {name!r} has not run for this config; run: python -m halprobe label {args.config}")
    out = args.out / cfg.experiment
    sample_path = out / "review_sample.csv"
    if sample_path.exists() and not args.force:
        raise SystemExit(f"{sample_path} exists (it may already be filled in); pass --force to overwrite")
    examples = load_examples(dirs["data"].final / "examples.jsonl")
    gens = {r["id"]: r for r in read_jsonl(dirs["generate"].final / "generations.jsonl")}
    labels = {r["id"]: r for r in read_jsonl(dirs["label"].final / "labels.jsonl")}
    rows = review_rows(examples, gens, labels)
    sample = sample_for_review(rows, args.n, args.seed)
    ambiguous = [r for r in rows if r["auto_label"] == "ambiguous"]
    write_review_csv(sample_path, sample)
    write_review_csv(out / "ambiguous.csv", ambiguous)
    print(f"wrote {len(sample)} rows to {sample_path} and {len(ambiguous)} rows to {out / 'ambiguous.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
