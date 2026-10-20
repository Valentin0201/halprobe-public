"""Batched greedy generation with per-token log-probs and first-token entropy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from transformers import GenerationConfig as HFGenerationConfig

from halprobe.batching import iter_with_backoff
from halprobe.cache import append_jsonl, read_jsonl, write_jsonl
from halprobe.prompts import encode_prompts


def postprocess(gen_ids: list[int], logprobs: list[float], first_entropy: float, tokenizer,
                stop_ids: set[int]) -> dict:
    """Cut the generation at the first stop token or newline-containing token.
    Greedy decoding is causal, so cutting afterwards equals stopping during generation."""
    stop_at = None
    for i, token_id in enumerate(gen_ids):
        if token_id in stop_ids or "\n" in tokenizer.decode([token_id]):
            stop_at = i
            break
    if stop_at is None:
        keep, n_answer, finish = len(gen_ids), len(gen_ids), "length"
    else:
        keep, n_answer, finish = stop_at + 1, stop_at, "stop"
    answer = tokenizer.decode(gen_ids[:keep], skip_special_tokens=True).split("\n")[0].strip()
    return {"answer": answer, "token_ids": gen_ids[:keep], "token_logprobs": logprobs[:keep],
            "n_answer_tokens": n_answer, "first_token_entropy": first_entropy, "finish_reason": finish}


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: Sequence[str], max_new_tokens: int, stop_ids: set[int]) -> list[dict]:
    enc = encode_prompts(tokenizer, prompts, model.device)
    config = HFGenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=sorted(stop_ids),
                                pad_token_id=tokenizer.pad_token_id, output_logits=True, return_dict_in_generate=True)
    out = model.generate(**enc, generation_config=config)
    gen = out.sequences[:, enc["input_ids"].shape[1] :]
    step_logprobs, entropy = [], None
    for t, logits in enumerate(out.logits):
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        step_logprobs.append(log_probs.gather(1, gen[:, t : t + 1]).squeeze(1))
        if t == 0:
            entropy = -(log_probs.exp() * log_probs).nan_to_num().sum(-1)
    logprobs = torch.stack(step_logprobs, dim=1).cpu().tolist()
    entropies = entropy.cpu().tolist()
    ids = gen.cpu().tolist()
    return [postprocess(ids[b], logprobs[b], entropies[b], tokenizer, stop_ids) for b in range(len(prompts))]


def run_generation(model, tokenizer, items: Sequence[tuple[str, str]], out_path: Path, batch_size: int,
                   max_new_tokens: int, stop_ids: set[int], log: Callable[[str], None] = print,
                   on_batch: Callable[[int], None] | None = None,
                   on_backoff: Callable[[int], None] | None = None) -> None:
    """Generate for (id, prompt) items, appending one JSON line per item; resumes from out_path."""
    out_path = Path(out_path)
    done: list[dict] = []
    if out_path.exists():
        done = read_jsonl(out_path, tolerate_partial_tail=True)
        write_jsonl(out_path, done)  # drop a torn last line before appending
    done_ids = {r["id"] for r in done}
    todo = [item for item in items if item[0] not in done_ids]
    log(f"[generate] {len(done_ids)} cached, {len(todo)} to generate")

    def generate(batch):
        return generate_batch(model, tokenizer, [prompt for _, prompt in batch], max_new_tokens, stop_ids)

    for n, (batch, records) in enumerate(iter_with_backoff(todo, batch_size, generate, on_backoff)):
        append_jsonl(out_path, [{"id": item_id, **rec} for (item_id, _), rec in zip(batch, records)])
        if on_batch is not None:
            on_batch(n)
