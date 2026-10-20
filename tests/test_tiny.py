import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from halprobe.data import build_examples, load_popqa_rows
from halprobe.tiny import BOS, EOT, PAD, write_tiny_popqa


def test_tiny_popqa_exercises_dedupe(tmp_path):
    write_tiny_popqa(tmp_path / "t.tsv", n_rows=40)
    rows = load_popqa_rows(tmp_path / "t.tsv")
    assert len(rows) == 40
    examples, report = build_examples(rows, (0.7, 0.15, 0.15), seed=0, shot_subjects=["Ada Lovelace"])
    assert report["groups_dropped_multi_entity"] == 1
    assert report["groups_merged"] == 1
    assert len(examples) == 37


def test_tiny_tokenizer_round_trip_and_template(tiny_assets):
    tok = AutoTokenizer.from_pretrained(tiny_assets.model_dir)
    assert (tok.bos_token, tok.eos_token, tok.pad_token) == (BOS, EOT, PAD)
    ids = tok("Hello\nWorld", add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == "Hello\nWorld"
    assert tok("Hi")["input_ids"][0] == tok.bos_token_id  # Llama-style BOS post-processor
    text = tok.apply_chat_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}],
        tokenize=False, add_generation_prompt=True, date_string="26 Jul 2024",
    )
    assert text.startswith(BOS)
    assert "Today Date: 26 Jul 2024" in text
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_tiny_model_forward(tiny_assets):
    model = AutoModelForCausalLM.from_pretrained(tiny_assets.model_dir)
    out = model(torch.tensor([[1, 2, 3]]), output_hidden_states=True)
    assert len(out.hidden_states) == 3
    assert out.logits.shape[-1] == model.config.vocab_size
