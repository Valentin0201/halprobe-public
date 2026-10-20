"""Hidden states at the last prompt token, stored in a resumable layer-major fp16 memmap."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from halprobe.batching import iter_with_backoff
from halprobe.cache import atomic_save_npy, atomic_write_json
from halprobe.prompts import encode_prompts, position_ids_from_mask

STATES_FILE = "states.f16.mmap"
DONE_FILE = "done.npy"
ARGMAX_FILE = "argmax.npy"
META_FILE = "meta.json"


class StateStore:
    """states.f16.mmap with shape (n_states, N, d_model); done.npy marks finished rows."""

    def __init__(self, directory: Path, arr: np.memmap, done: np.ndarray, argmax: np.ndarray):
        self.directory, self.arr, self.done, self.argmax = directory, arr, done, argmax

    @classmethod
    def open(cls, directory: Path, n_states: int, ids: Sequence[str], d_model: int, meta: dict) -> StateStore:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shape = (n_states, len(ids), d_model)
        full_meta = {**meta, "shape": list(shape), "dtype": "float16", "ids": list(ids)}
        meta_path = directory / META_FILE
        if meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if existing != full_meta:
                raise ValueError(f"{directory}: existing meta.json does not match this run; delete the directory to restart")
            arr = np.memmap(directory / STATES_FILE, dtype=np.float16, mode="r+", shape=shape)
            done = np.load(directory / DONE_FILE)
            argmax = np.load(directory / ARGMAX_FILE)
        else:
            arr = np.memmap(directory / STATES_FILE, dtype=np.float16, mode="w+", shape=shape)
            done = np.zeros(len(ids), dtype=bool)
            argmax = np.full(len(ids), -1, dtype=np.int64)
            atomic_save_npy(directory / DONE_FILE, done)
            atomic_save_npy(directory / ARGMAX_FILE, argmax)
            atomic_write_json(meta_path, full_meta)  # written last: its presence means the files exist
        return cls(directory, arr, done, argmax)

    def pending(self) -> list[int]:
        return np.flatnonzero(~self.done).tolist()

    def write(self, rows: Sequence[int], states: np.ndarray, argmax: np.ndarray) -> None:
        rows = list(rows)
        self.arr[:, rows, :] = states
        self.arr.flush()
        self.argmax[rows] = argmax
        atomic_save_npy(self.directory / ARGMAX_FILE, self.argmax)
        self.done[rows] = True
        atomic_save_npy(self.directory / DONE_FILE, self.done)  # last: rows count as done only after their data

    def close(self) -> None:
        if self.arr is not None:
            self.arr.flush()
            self.arr = None
            gc.collect()  # release the mapping so Windows can rename the directory

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@torch.no_grad()
def last_token_states(model, tokenizer, prompts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    enc = encode_prompts(tokenizer, prompts, model.device)
    out = model(**enc, position_ids=position_ids_from_mask(enc["attention_mask"]),
                output_hidden_states=True, logits_to_keep=1)
    states = torch.stack([h[:, -1, :] for h in out.hidden_states]).to(torch.float16).cpu().numpy()
    argmax = out.logits[:, -1, :].argmax(-1).cpu().numpy().astype(np.int64)
    return states, argmax


def run_extraction(model, tokenizer, ids: Sequence[str], prompts: Sequence[str], directory: Path, batch_size: int,
                   meta: dict, log: Callable[[str], None] = print, on_batch: Callable[[int], None] | None = None,
                   on_backoff: Callable[[int], None] | None = None) -> None:
    n_states = model.config.num_hidden_layers + 1
    with StateStore.open(Path(directory), n_states, ids, model.config.hidden_size, meta) as store:
        pending = store.pending()
        log(f"[extract] {len(ids) - len(pending)} cached, {len(pending)} to extract")

        def extract(rows):
            return last_token_states(model, tokenizer, [prompts[r] for r in rows])

        for n, (rows, (states, argmax)) in enumerate(iter_with_backoff(pending, batch_size, extract, on_backoff)):
            store.write(rows, states, argmax)
            if on_batch is not None:
                on_batch(n)


def open_states(directory: Path) -> tuple[np.memmap, dict]:
    directory = Path(directory)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    states = np.memmap(directory / STATES_FILE, dtype=np.float16, mode="r", shape=tuple(meta["shape"]))
    return states, meta
