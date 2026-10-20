"""Model and tokenizer loading."""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from halprobe.config import ModelConfig

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
WeightTransform = Callable[[torch.nn.Module], None]


def load_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path, revision=cfg.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(cfg: ModelConfig, device: torch.device, weight_transform: WeightTransform | None = None):
    """Load in the configured dtype. `weight_transform` (identity in the MVP) is the hook for Phase 3
    weight-quantization experiments; it runs before the model is moved to `device`."""
    model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, revision=cfg.revision,
                                                 dtype=DTYPES[cfg.dtype], attn_implementation="sdpa")
    if weight_transform is not None:
        weight_transform(model)
    return model.to(device).eval()


def stop_token_ids(model, tokenizer) -> set[int]:
    ids: set[int] = set()
    eos = model.generation_config.eos_token_id
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids
