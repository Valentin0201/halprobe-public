import pytest
import torch

from halprobe.device import resolve_device, seed_everything


def _fake_gpus(monkeypatch, names):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(names))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(names))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: names[i])


def test_cpu_is_always_allowed(monkeypatch):
    _fake_gpus(monkeypatch, [])
    assert resolve_device("cpu") == torch.device("cpu")


def test_auto_picks_gpu_by_name_hint(monkeypatch):
    _fake_gpus(monkeypatch, ["AMD Radeon(TM) Graphics", "AMD Radeon RX 6700 XT"])
    assert resolve_device("auto", "6700") == torch.device("cuda:1")


def test_auto_hint_is_case_insensitive(monkeypatch):
    _fake_gpus(monkeypatch, ["AMD Radeon RX 6700 XT"])
    assert resolve_device("auto", "rx 6700") == torch.device("cuda:0")


def test_auto_without_gpu_raises_instead_of_falling_back_to_cpu(monkeypatch):
    _fake_gpus(monkeypatch, [])
    with pytest.raises(RuntimeError, match="no GPU"):
        resolve_device("auto", "6700")


def test_auto_hint_not_found_lists_visible_devices(monkeypatch):
    _fake_gpus(monkeypatch, ["AMD Radeon(TM) Graphics"])
    with pytest.raises(RuntimeError, match=r"Radeon\(TM\) Graphics"):
        resolve_device("auto", "6700")


def test_auto_without_hint_uses_first_gpu(monkeypatch):
    _fake_gpus(monkeypatch, ["A", "B"])
    assert resolve_device("auto") == torch.device("cuda:0")


def test_explicit_index_out_of_range(monkeypatch):
    _fake_gpus(monkeypatch, ["A"])
    with pytest.raises(RuntimeError, match="cuda:3"):
        resolve_device("cuda:3")


def test_unknown_spec(monkeypatch):
    _fake_gpus(monkeypatch, ["A"])
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("gpu")


def test_seed_everything_is_reproducible():
    seed_everything(3)
    a = torch.rand(3)
    seed_everything(3)
    b = torch.rand(3)
    assert torch.equal(a, b)
