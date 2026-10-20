import torch

from halprobe.models import load_model, stop_token_ids
from halprobe.tiny import EOT


def test_tokenizer_is_left_padded_with_pad_token(tiny_tokenizer):
    assert tiny_tokenizer.padding_side == "left"
    assert tiny_tokenizer.pad_token is not None


def test_model_is_in_eval_mode_on_cpu(tiny_model):
    assert not tiny_model.training
    assert tiny_model.device == torch.device("cpu")


def test_weight_transform_hook_is_applied(tiny_model_config):
    def zero_embeddings(model):
        model.model.embed_tokens.weight.data.zero_()

    model = load_model(tiny_model_config, torch.device("cpu"), weight_transform=zero_embeddings)
    assert float(model.model.embed_tokens.weight.abs().sum()) == 0.0


def test_stop_token_ids_include_eot(tiny_model, tiny_tokenizer):
    assert tiny_tokenizer.convert_tokens_to_ids(EOT) in stop_token_ids(tiny_model, tiny_tokenizer)
