"""Tiny offline assets for tests and CPU smoke runs: tokenizer, random Llama, synthetic PopQA TSV."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from halprobe.data import POPQA_COLUMNS

BOS, EOT, PAD = "<|begin_of_text|>", "<|eot_id|>", "<|pad|>"
SPECIAL_TOKENS = [BOS, "<|start_header_id|>", "<|end_header_id|>", EOT, PAD]
# Mirrors Llama-3.2: injects today's date into the system turn unless date_string is passed.
CHAT_TEMPLATE = (
    "{% if date_string is not defined %}{% set date_string = strftime_now('%d %b %Y') %}{% endif %}"
    "{{ bos_token }}"
    "{% for m in messages %}"
    "<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{% if m['role'] == 'system' %}Today Date: {{ date_string }}\n\n{% endif %}"
    "{{ m['content'] }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)
TINY_RELATIONS = {
    "occupation": ("What is {s}'s occupation?", ("politician", "actor", "writer", "painter")),
    "genre": ("What genre is {s}?", ("rock music", "jazz", "comedy film", "drama")),
    "capital": ("What is the capital of {s}?", ("Alpha City", "Beta Town", "Gamma")),
    "director": ("Who was the director of {s}?", ("Ann Lee", "Bo Park", "Cy Diaz")),
}


def _row(id: int, subj: str, subj_id: int, prop: str, obj: str, question: str, answers: list[str], s_pop: int) -> dict:
    return {"id": str(id), "subj": subj, "prop": prop, "obj": obj, "subj_id": str(subj_id), "prop_id": "0",
            "obj_id": "0", "s_aliases": "[]", "o_aliases": "[]", "s_uri": "", "o_uri": "", "s_wiki_title": subj,
            "o_wiki_title": obj, "s_pop": str(s_pop), "o_pop": "0", "question": question,
            "possible_answers": json.dumps(answers)}


def write_tiny_popqa(path: Path | str, n_rows: int = 160, seed: int = 0) -> list[dict]:
    """Synthetic PopQA-format TSV. The last 3 rows add one same-entity duplicate (merged by dedupe)
    and one same-name/different-entity pair (dropped by dedupe)."""
    rng = random.Random(seed)
    relations = list(TINY_RELATIONS)
    rows = []
    for i in range(n_rows - 3):
        rel = relations[i % len(relations)]
        template, answers = TINY_RELATIONS[rel]
        subj, obj = f"Entity {i}", rng.choice(answers)
        rows.append(_row(1000 + i, subj, 5000 + i, rel, obj, template.format(s=subj), [obj, obj.lower()],
                         rng.randint(2, 100_000)))
    first = rows[0]
    rows.append(_row(1000 + n_rows - 3, first["subj"], 5000, first["prop"], "actor", first["question"], ["actor"],
                     int(first["s_pop"])))
    template = TINY_RELATIONS["occupation"][0]
    rows.append(_row(1000 + n_rows - 2, "Twin", 9001, "occupation", "actor", template.format(s="Twin"), ["actor"], 50))
    rows.append(_row(1000 + n_rows - 1, "Twin", 9002, "occupation", "writer", template.format(s="Twin"), ["writer"], 60))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POPQA_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_tiny_tokenizer(corpus: list[str]) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(corpus, trainer=trainer)
    # Like Llama: tokenizer(text) prepends BOS unless add_special_tokens=False.
    tok.post_processor = processors.TemplateProcessing(single=f"{BOS} $A", special_tokens=[(BOS, tok.token_to_id(BOS))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=BOS, eos_token=EOT, pad_token=PAD)
    fast.chat_template = CHAT_TEMPLATE
    fast.padding_side = "left"
    return fast


def build_tiny_model(tokenizer: PreTrainedTokenizerFast) -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model


@dataclass(frozen=True)
class TinyAssets:
    tsv: Path
    model_dir: Path


def write_tiny_assets(out_dir: Path | str) -> TinyAssets:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "popqa_tiny.tsv"
    rows = write_tiny_popqa(tsv)
    corpus = [f"{r['question']} {r['obj']}" for r in rows] + ["Answer the question with a short phrase."]
    tokenizer = build_tiny_tokenizer(corpus)
    model = build_tiny_model(tokenizer)
    model_dir = out_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return TinyAssets(tsv=tsv, model_dir=model_dir)
