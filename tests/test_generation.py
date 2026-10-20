import math

import numpy as np
import pytest
import torch

from halprobe.cache import read_jsonl
from halprobe.generation import generate_batch, postprocess, run_generation
from halprobe.models import stop_token_ids
from halprobe.prompts import encode_prompts, position_ids_from_mask, render_prompt
from halprobe.tiny import EOT


def _quiet(_):
    pass


def test_postprocess_stops_at_stop_token(tiny_tokenizer):
    eot = tiny_tokenizer.convert_tokens_to_ids(EOT)
    paris = tiny_tokenizer.encode("Paris", add_special_tokens=False)
    ids = paris + [eot, 5, 6]
    lps = [-0.1] * len(paris) + [-0.2, -0.3, -0.4]
    rec = postprocess(ids, lps, 1.5, tiny_tokenizer, {eot})
    assert rec["answer"] == "Paris"
    assert rec["token_ids"] == paris + [eot]
    assert rec["token_logprobs"] == lps[: len(paris) + 1]
    assert rec["n_answer_tokens"] == len(paris)
    assert rec["finish_reason"] == "stop" and rec["first_token_entropy"] == 1.5


def test_postprocess_stops_at_newline(tiny_tokenizer):
    paris = tiny_tokenizer.encode("Paris", add_special_tokens=False)
    newline = tiny_tokenizer.encode("\n", add_special_tokens=False)
    more = tiny_tokenizer.encode("France", add_special_tokens=False)
    ids = paris + newline + more
    rec = postprocess(ids, [-0.5] * len(ids), 2.0, tiny_tokenizer, set())
    assert rec["answer"] == "Paris"
    assert rec["n_answer_tokens"] == len(paris)
    assert rec["finish_reason"] == "stop"


def test_postprocess_length_when_no_stop(tiny_tokenizer):
    ids = tiny_tokenizer.encode("Paris France", add_special_tokens=False)
    rec = postprocess(ids, [-0.5] * len(ids), 2.0, tiny_tokenizer, set())
    assert rec["finish_reason"] == "length" and rec["n_answer_tokens"] == len(ids)


def test_postprocess_empty_when_first_token_stops(tiny_tokenizer):
    eot = tiny_tokenizer.convert_tokens_to_ids(EOT)
    rec = postprocess([eot, 7], [-0.1, -0.2], 1.0, tiny_tokenizer, {eot})
    assert rec["answer"] == "" and rec["n_answer_tokens"] == 0 and rec["token_ids"] == [eot]


def test_generate_batch_records(tiny_model, tiny_tokenizer, tiny_prompt):
    prompts = [render_prompt(tiny_tokenizer, q, tiny_prompt)
               for q in ("What is Entity 1's occupation?", "Who was the director of Entity 3?")]
    recs = generate_batch(tiny_model, tiny_tokenizer, prompts, 8, stop_token_ids(tiny_model, tiny_tokenizer))
    assert len(recs) == 2
    for rec in recs:
        assert 1 <= len(rec["token_ids"]) <= 8
        assert len(rec["token_logprobs"]) == len(rec["token_ids"])
        assert all(lp <= 1e-6 for lp in rec["token_logprobs"])
        assert 0 < rec["first_token_entropy"] <= math.log(len(tiny_tokenizer)) + 1e-4
        assert rec["finish_reason"] in {"stop", "length"}


def test_first_generated_token_is_greedy_argmax(tiny_model, tiny_tokenizer, tiny_prompt):
    prompts = [render_prompt(tiny_tokenizer, q, tiny_prompt) for q in ("Who?", "What genre is Entity 9?")]
    recs = generate_batch(tiny_model, tiny_tokenizer, prompts, 4, stop_token_ids(tiny_model, tiny_tokenizer))
    enc = encode_prompts(tiny_tokenizer, prompts, torch.device("cpu"))
    with torch.no_grad():
        logits = tiny_model(**enc, position_ids=position_ids_from_mask(enc["attention_mask"])).logits[:, -1]
    assert [r["token_ids"][0] for r in recs] == logits.argmax(-1).tolist()


def _items(tok, prompt, n=10):
    return [(f"id{i}", render_prompt(tok, f"What is Entity {i}'s occupation?", prompt)) for i in range(n)]


def test_run_generation_resumes_after_interrupt(tmp_path, tiny_model, tiny_tokenizer, tiny_prompt):
    items = _items(tiny_tokenizer, tiny_prompt)
    stop = stop_token_ids(tiny_model, tiny_tokenizer)
    full = tmp_path / "full.jsonl"
    run_generation(tiny_model, tiny_tokenizer, items, full, 4, 6, stop, log=_quiet)

    def boom(n):
        if n == 0:
            raise KeyboardInterrupt

    part = tmp_path / "part.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run_generation(tiny_model, tiny_tokenizer, items, part, 4, 6, stop, log=_quiet, on_batch=boom)
    assert len(read_jsonl(part)) == 4
    run_generation(tiny_model, tiny_tokenizer, items, part, 4, 6, stop, log=_quiet)
    key = lambda r: r["id"]  # noqa: E731
    assert sorted(read_jsonl(part), key=key) == sorted(read_jsonl(full), key=key)


def test_run_generation_repairs_partial_tail(tmp_path, tiny_model, tiny_tokenizer, tiny_prompt):
    items = _items(tiny_tokenizer, tiny_prompt)
    stop = stop_token_ids(tiny_model, tiny_tokenizer)
    full = tmp_path / "full.jsonl"
    run_generation(tiny_model, tiny_tokenizer, items, full, 4, 6, stop, log=_quiet)
    first_line = full.read_text(encoding="utf-8").splitlines()[0]
    part = tmp_path / "part.jsonl"
    part.write_text(first_line + '\n{"id": "id1", "ans', encoding="utf-8")
    run_generation(tiny_model, tiny_tokenizer, items, part, 4, 6, stop, log=_quiet)
    got = {r["id"]: r for r in read_jsonl(part)}
    want = {r["id"]: r for r in read_jsonl(full)}
    assert sorted(got) == sorted(want)
    for i in want:
        assert got[i]["token_ids"] == want[i]["token_ids"] and got[i]["answer"] == want[i]["answer"]
        np.testing.assert_allclose(got[i]["token_logprobs"], want[i]["token_logprobs"], atol=1e-4)
