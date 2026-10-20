"""Few-shot chat prompt construction and tokenization."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from halprobe.config import PromptConfig


def build_messages(question: str, prompt: PromptConfig) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": prompt.system}]
    for shot in prompt.shots:
        messages.append({"role": "user", "content": shot.question})
        messages.append({"role": "assistant", "content": shot.answer})
    messages.append({"role": "user", "content": question})
    return messages


def render_prompt(tokenizer, question: str, prompt: PromptConfig) -> str:
    # date_string pins the date Llama-3.2's template would otherwise take from today's clock.
    return tokenizer.apply_chat_template(build_messages(question, prompt), tokenize=False,
                                         add_generation_prompt=True, date_string=prompt.date_string)


def encode_prompts(tokenizer, prompts: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
    """Tokenize rendered prompts. The template already contains BOS, so no special tokens are added.
    Left padding makes position -1 the last prompt token of every row."""
    if tokenizer.padding_side != "left":
        raise ValueError("tokenizer must pad on the left")
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
