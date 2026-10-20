"""Render result tables from a run's metrics.json into README.md and docs/results.md.

Usage: uv run python scripts/render_results.py results/mvp_popqa_llama3b/<run_id>
Refuses runs with pseudo labels.
"""

import argparse
from pathlib import Path

from halprobe.report import render_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render README/docs result tables from a run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--results-md", type=Path, default=Path("docs/results.md"))
    args = parser.parse_args(argv)
    render_results(args.run_dir, args.readme, args.results_md)
    print(f"rendered {args.run_dir} into {args.readme} and {args.results_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
