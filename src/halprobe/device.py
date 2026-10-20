"""Device selection and global seeding."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(spec: str, gpu_name_hint: str | None = None) -> torch.device:
    """Resolve 'cpu', 'cuda:N' or 'auto' (GPU chosen by name hint) to a torch device.

    'auto' never falls back to CPU: a real run on CPU would silently take hours.
    """
    if spec == "cpu":
        return torch.device("cpu")
    if not (spec == "auto" or spec.startswith("cuda:")):
        raise ValueError(f"unknown device spec {spec!r}; expected 'auto', 'cpu' or 'cuda:N'")
    if not torch.cuda.is_available():
        raise RuntimeError(f"device={spec!r} requested but no GPU is visible to torch; set device: cpu explicitly")
    count = torch.cuda.device_count()
    if spec.startswith("cuda:"):
        if int(spec.split(":", 1)[1]) >= count:
            raise RuntimeError(f"{spec} requested but only {count} GPU(s) are visible")
        return torch.device(spec)
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if gpu_name_hint is None:
        return torch.device("cuda:0")
    for index, name in enumerate(names):
        if gpu_name_hint.lower() in name.lower():
            return torch.device(f"cuda:{index}")
    raise RuntimeError(f"no GPU name contains {gpu_name_hint!r}; visible devices: {names}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
