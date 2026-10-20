"""Content-addressed stage directories and crash-safe file writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

COMPLETE_MARKER = "_COMPLETE"


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def stage_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class StageDir:
    """artifacts/<stage>/<key>/ (complete) and artifacts/<stage>/<key>.tmp/ (in progress)."""

    def __init__(self, root: Path | str, stage: str, key: str):
        self.stage = stage
        self.key = key
        self.final = Path(root) / stage / key
        self.tmp = Path(root) / stage / f"{key}.tmp"

    def is_complete(self) -> bool:
        return (self.final / COMPLETE_MARKER).is_file()

    def begin(self) -> Path:
        self.tmp.mkdir(parents=True, exist_ok=True)
        return self.tmp

    def commit(self) -> Path:
        (self.tmp / COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
        if self.final.exists():
            shutil.rmtree(self.final)
        os.replace(self.tmp, self.final)
        return self.final


def atomic_write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)


def atomic_write_json(path: Path | str, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=json_default) + "\n")


def atomic_save_npy(path: Path | str, array: np.ndarray) -> None:
    path = Path(path)
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)


def _dumps(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, default=json_default)


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    atomic_write_text(path, "".join(_dumps(r) + "\n" for r in rows))


def append_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(_dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path | str, tolerate_partial_tail: bool = False) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerate_partial_tail and i == len(lines) - 1:
                break
            raise ValueError(f"{path}: invalid JSON on line {i + 1}") from None
    return rows
