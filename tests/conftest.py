import pytest
import torch

from halprobe.config import ModelConfig, PromptConfig, ShotConfig
from halprobe.models import load_model, load_tokenizer
from halprobe.tiny import write_tiny_assets


@pytest.fixture(scope="session")
def tiny_assets(tmp_path_factory):
    return write_tiny_assets(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="session")
def tiny_model_config(tiny_assets):
    return ModelConfig(name_or_path=str(tiny_assets.model_dir), dtype="float32")


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model_config):
    return load_tokenizer(tiny_model_config)


@pytest.fixture(scope="session")
def tiny_model(tiny_model_config):
    return load_model(tiny_model_config, torch.device("cpu"))


@pytest.fixture(scope="session")
def tiny_prompt():
    return PromptConfig(
        system="Answer the question with a short phrase.",
        shots=(ShotConfig(subject="Ada Lovelace", question="What is Ada Lovelace's occupation?", answer="mathematician"),),
        date_string="26 Jul 2024",
    )


from pathlib import Path

import yaml

from halprobe.config import config_from_dict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_config(tiny_assets):
    raw = yaml.safe_load((ROOT / "configs" / "tiny_cpu.yaml").read_text(encoding="utf-8"))
    raw["data"]["local_path"] = str(tiny_assets.tsv)
    raw["model"]["name_or_path"] = str(tiny_assets.model_dir)
    return config_from_dict(raw)
