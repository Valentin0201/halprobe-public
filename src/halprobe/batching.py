"""Batch iteration that halves the batch size on GPU out-of-memory errors."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

import torch

T = TypeVar("T")


def iter_with_backoff(items: Sequence[T], batch_size: int, fn: Callable[[Sequence[T]], Any],
                      on_backoff: Callable[[int], None] | None = None) -> Iterator[tuple[Sequence[T], Any]]:
    """Yield (batch, fn(batch)). On OOM, halve the batch size and retry (down to 1, then re-raise)."""
    i, size = 0, batch_size
    while i < len(items):
        batch = items[i : i + size]
        try:
            result = fn(batch)
        except torch.OutOfMemoryError:
            if size == 1:
                raise
            size = max(1, size // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if on_backoff is not None:
                on_backoff(size)
            continue
        yield batch, result
        i += len(batch)
