from pathlib import Path

import pytest
import yaml

from halprobe.config import Config, ConfigError, config_from_dict, load_config, to_plain

ROOT = Path(__file__).resolve().parents[1]
MVP = ROOT / "configs" / "mvp_popqa_llama3b.yaml"


def _raw():
    return yaml.safe_load(MVP.read_text(encoding="utf-8"))


def test_mvp_config_loads():
    cfg = load_config(MVP)
    assert isinstance(cfg, Config)
    assert cfg.model.revision == "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    assert cfg.data.revision == "098765c79ea10a2cb19c828324e33281b8336ec0"
    assert len(cfg.prompt.shots) == 4
    assert cfg.prompt.shots[2].answer == "Emily Brontë"
    assert cfg.probe.C_grid[0] == pytest.approx(1e-4)
    assert cfg.gpu_name_hint == "6700"
    assert cfg.data.split_ratios == (0.7, 0.15, 0.15)


def test_round_trip_through_plain_dict():
    cfg = load_config(MVP)
    assert config_from_dict(to_plain(cfg)) == cfg


def test_unknown_nested_key_is_rejected():
    raw = _raw()
    raw["probe"]["C"] = 1.0
    with pytest.raises(ConfigError, match=r"probe: unknown key\(s\) \['C'\]"):
        config_from_dict(raw)


def test_missing_required_key():
    raw = _raw()
    del raw["model"]
    with pytest.raises(ConfigError, match="model: required key missing"):
        config_from_dict(raw)


def test_wrong_type_is_rejected():
    raw = _raw()
    raw["gpu_name_hint"] = 6700
    with pytest.raises(ConfigError, match="gpu_name_hint: expected a string"):
        config_from_dict(raw)


def test_string_floats_are_accepted():
    raw = _raw()
    raw["probe"]["C_grid"] = ["1e-4", 1]
    assert config_from_dict(raw).probe.C_grid == (1e-4, 1.0)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("data", "split_ratios"), [0.5, 0.5], "split_ratios"),
        (("model", "dtype"), "int8", "model.dtype"),
        (("labeling", "mode"), "judge", "labeling.mode"),
        (("device",), "gpu", "device"),
        (("experiment",), "Has Spaces", "experiment"),
        (("probe", "C_grid"), [], "C_grid"),
    ],
)
def test_invalid_values(path, value, message):
    raw = _raw()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)
