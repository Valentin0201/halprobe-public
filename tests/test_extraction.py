import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from halprobe.cache import StageDir
from halprobe.extraction import ARGMAX_FILE, DONE_FILE, last_token_states, open_states, run_extraction
from halprobe.prompts import render_prompt


def _quiet(_):
    pass


def _sharp_model(tokenizer):
    """Larger init than the tiny assets: attention is sharp, so padding/masking bugs become visible."""
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                         initializer_range=0.2, pad_token_id=tokenizer.pad_token_id)
    return LlamaForCausalLM(config).eval()


def test_batched_states_match_unbatched(tiny_tokenizer, tiny_prompt):
    model = _sharp_model(tiny_tokenizer)
    prompts = [render_prompt(tiny_tokenizer, "Who?", tiny_prompt),
               render_prompt(tiny_tokenizer, "What is the capital of Entity 12, and what is it known for?", tiny_prompt)]
    batch_states, batch_argmax = last_token_states(model, tiny_tokenizer, prompts)
    assert batch_states.shape == (3, 2, 64) and batch_states.dtype == np.float16
    for i, prompt in enumerate(prompts):
        single_states, single_argmax = last_token_states(model, tiny_tokenizer, [prompt])
        np.testing.assert_allclose(batch_states[:, i].astype(np.float32), single_states[:, 0].astype(np.float32),
                                   rtol=1e-2, atol=1e-3)
        assert batch_argmax[i] == single_argmax[0]


def _ids_prompts(tok, prompt, n=10):
    return [f"id{i}" for i in range(n)], [render_prompt(tok, f"What genre is Entity {i}?", prompt) for i in range(n)]


def test_extraction_resume_is_bit_identical(tmp_path, tiny_model, tiny_tokenizer, tiny_prompt):
    ids, prompts = _ids_prompts(tiny_tokenizer, tiny_prompt)
    run_extraction(tiny_model, tiny_tokenizer, ids, prompts, tmp_path / "full", 4, {"m": 1}, log=_quiet)

    def boom(n):
        if n == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_extraction(tiny_model, tiny_tokenizer, ids, prompts, tmp_path / "part", 4, {"m": 1}, log=_quiet, on_batch=boom)
    assert int(np.load(tmp_path / "part" / DONE_FILE).sum()) == 8
    run_extraction(tiny_model, tiny_tokenizer, ids, prompts, tmp_path / "part", 4, {"m": 1}, log=_quiet)
    full, meta = open_states(tmp_path / "full")
    part, _ = open_states(tmp_path / "part")
    assert meta["ids"] == ids and meta["shape"] == [3, 10, 64]
    assert np.array_equal(np.asarray(full), np.asarray(part))
    assert np.array_equal(np.load(tmp_path / "full" / ARGMAX_FILE), np.load(tmp_path / "part" / ARGMAX_FILE))
    del full, part


def test_extraction_rejects_mismatched_meta(tmp_path, tiny_model, tiny_tokenizer, tiny_prompt):
    ids, prompts = _ids_prompts(tiny_tokenizer, tiny_prompt, n=2)
    run_extraction(tiny_model, tiny_tokenizer, ids, prompts, tmp_path / "x", 4, {"m": 1}, log=_quiet)
    with pytest.raises(ValueError, match="meta"):
        run_extraction(tiny_model, tiny_tokenizer, ids, prompts, tmp_path / "x", 4, {"m": 2}, log=_quiet)


def test_stage_commit_after_extraction(tmp_path, tiny_model, tiny_tokenizer, tiny_prompt):
    ids, prompts = _ids_prompts(tiny_tokenizer, tiny_prompt, n=3)
    sd = StageDir(tmp_path, "extract", "k")
    run_extraction(tiny_model, tiny_tokenizer, ids, prompts, sd.begin(), 2, {"m": 1}, log=_quiet)
    final = sd.commit()  # fails on Windows if the memmap were still open
    assert sd.is_complete()
    states, _ = open_states(final)
    assert states.shape == (3, 3, 64)
    del states
