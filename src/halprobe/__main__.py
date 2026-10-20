"""CLI: python -m halprobe {run,<stage>} CONFIG | python -m halprobe make-tiny DIR"""

from __future__ import annotations

import argparse
from pathlib import Path

from halprobe.config import load_config
from halprobe.pipeline import STAGES, run
from halprobe.tiny import write_tiny_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="halprobe", description="Linear-probe hallucination detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", *STAGES[:-1]):
        help_text = "run all stages" if name == "run" else f"run stages up to and including '{name}'"
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", type=Path)
        p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
        p.add_argument("--results", type=Path, default=Path("results"))
    tiny = sub.add_parser("make-tiny", help="write tiny offline assets for configs/tiny_cpu.yaml")
    tiny.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "make-tiny":
        assets = write_tiny_assets(args.out_dir)
        print(f"wrote {assets.tsv} and {assets.model_dir}")
        return 0
    cfg = load_config(args.config)
    result = run(cfg, args.artifacts, args.results, until="report" if args.command == "run" else args.command)
    if result.run_dir is not None:
        print(f"results written to {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
