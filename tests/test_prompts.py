from dataclasses import replace

import torch

from halprobe.prompts import build_messages, encode_prompts, position_ids_from_mask, render_prompt


def test_build_messages_order(tiny_prompt):
    msgs = build_messages("Who?", tiny_prompt)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "Who?" and msgs[2]["content"] == "mathematician"


def test_render_prompt_uses_configured_date(tiny_tokenizer, tiny_prompt):
    text = render_prompt(tiny_tokenizer, "Who?", tiny_prompt)
    assert "Today Date: 26 Jul 2024" in text
    assert text == render_prompt(tiny_tokenizer, "Who?", tiny_prompt)
    other = render_prompt(tiny_tokenizer, "Who?", replace(tiny_prompt, date_string="01 Jan 2000"))
    assert "Today Date: 01 Jan 2000" in other
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_encode_prompts_single_bos_and_left_padding(tiny_tokenizer, tiny_prompt):
    prompts = [render_prompt(tiny_tokenizer, "Who?", tiny_prompt),
               render_prompt(tiny_tokenizer, "What is the capital of Entity 7 in the long run?", tiny_prompt)]
    enc = encode_prompts(tiny_tokenizer, prompts, torch.device("cpu"))
    ids, mask = enc["input_ids"], enc["attention_mask"]
    assert set(enc) == {"input_ids", "attention_mask"}
    assert bool(mask[:, -1].all())
    assert int(mask[0, 0]) == 0
    for row in range(2):
        real = ids[row][mask[row].bool()]
        assert int(real[0]) == tiny_tokenizer.bos_token_id
        assert int((real == tiny_tokenizer.bos_token_id).sum()) == 1


def test_position_ids_from_mask():
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert position_ids_from_mask(mask).tolist() == [[0, 0, 0, 1], [0, 1, 2, 3]]
