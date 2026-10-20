"""Score a filled-in review CSV (label-noise rate of the automatic labels).

Usage: uv run python scripts/score_review.py results/review/mvp_popqa_llama3b/review_sample.csv
Writes <name>_score.json next to the CSV.
"""

import argparse
import json
from pathlib import Path

from halprobe.cache import atomic_write_json
from halprobe.review import score_review


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a filled-in manual review CSV.")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args(argv)
    result = score_review(args.csv)
    out = args.csv.with_name(args.csv.stem + "_score.json")
    atomic_write_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "disagreeing_ids"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
