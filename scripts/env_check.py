"""Environment check for halprobe. Usage: uv run python scripts/env_check.py [--cpu] [--skip-model]"""

from halprobe.envcheck import main

if __name__ == "__main__":
    raise SystemExit(main())
