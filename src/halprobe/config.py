"""Typed, strictly validated experiment configuration loaded from YAML."""

from __future__ import annotations

import re
import types
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Invalid, missing or unknown configuration value."""


DTYPE_NAMES = ("float16", "bfloat16", "float32")
LABEL_MODES = ("alias", "pseudo")
_DEVICE_RE = re.compile(r"^(auto|cpu|cuda:\d+)$")
_EXPERIMENT_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class DataConfig:
    repo_id: str = "akariasai/PopQA"
    revision: str | None = None
    filename: str = "test.tsv"
    local_path: str | None = None
    split_ratios: tuple[float, ...] = (0.7, 0.15, 0.15)

    def __post_init__(self) -> None:
        r = self.split_ratios
        if len(r) != 3 or min(r) <= 0 or abs(sum(r) - 1.0) > 1e-6:
            raise ConfigError(f"data.split_ratios must be three positive numbers summing to 1, got {r}")


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    revision: str | None = None
    dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.dtype not in DTYPE_NAMES:
            raise ConfigError(f"model.dtype must be one of {DTYPE_NAMES}, got {self.dtype!r}")


@dataclass(frozen=True)
class ShotConfig:
    subject: str
    question: str
    answer: str


@dataclass(frozen=True)
class PromptConfig:
    system: str
    shots: tuple[ShotConfig, ...] = ()
    date_string: str = "26 Jul 2024"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1 or self.batch_size < 1:
            raise ConfigError("generation.max_new_tokens and generation.batch_size must be >= 1")


@dataclass(frozen=True)
class LabelingConfig:
    mode: str = "alias"

    def __post_init__(self) -> None:
        if self.mode not in LABEL_MODES:
            raise ConfigError(f"labeling.mode must be one of {LABEL_MODES}, got {self.mode!r}")


@dataclass(frozen=True)
class ExtractionConfig:
    batch_size: int = 32
    consistency_threshold: float = 0.99

    def __post_init__(self) -> None:
        if self.batch_size < 1 or not 0.0 <= self.consistency_threshold <= 1.0:
            raise ConfigError("extraction.batch_size must be >= 1 and consistency_threshold in [0, 1]")


@dataclass(frozen=True)
class ProbeConfig:
    C_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
    max_iter: int = 5000
    min_class_count: int = 20

    def __post_init__(self) -> None:
        if not self.C_grid or min(self.C_grid) <= 0:
            raise ConfigError(f"probe.C_grid must be a non-empty list of positive numbers, got {self.C_grid}")
        if self.max_iter < 1 or self.min_class_count < 1:
            raise ConfigError("probe.max_iter and probe.min_class_count must be >= 1")


@dataclass(frozen=True)
class EvaluationConfig:
    n_bootstrap: int = 1000
    ece_bins: int = 15
    min_per_class_relation: int = 20

    def __post_init__(self) -> None:
        if min(self.n_bootstrap, self.ece_bins, self.min_per_class_relation) < 1:
            raise ConfigError("evaluation values must be >= 1")


@dataclass(frozen=True)
class Config:
    experiment: str
    model: ModelConfig
    prompt: PromptConfig
    seed: int = 0
    device: str = "auto"
    gpu_name_hint: str | None = None
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if not _EXPERIMENT_RE.match(self.experiment):
            raise ConfigError(f"experiment must match {_EXPERIMENT_RE.pattern}, got {self.experiment!r}")
        if not _DEVICE_RE.match(self.device):
            raise ConfigError(f"device must be auto, cpu or cuda:N, got {self.device!r}")


def _join(where: str, name: str) -> str:
    return f"{where}.{name}" if where else name


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if is_dataclass(tp):
        return _build(tp, value, where)
    origin = typing.get_origin(tp)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        inner = typing.get_args(tp)[0]
        return tuple(_coerce(inner, v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        inner = next(a for a in typing.get_args(tp) if a is not type(None))
        return _coerce(inner, value, where)
    if tp is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: expected a number, got {value!r}") from None
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported type {tp!r}")


def _build(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    unknown = sorted(set(data) - {f.name for f in fields(cls)})
    if unknown:
        raise ConfigError(f"{where or 'config'}: unknown key(s) {unknown}")
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _coerce(hints[f.name], data[f.name], _join(where, f.name))
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ConfigError(f"{_join(where, f.name)}: required key missing")
    return cls(**kwargs)


def config_from_dict(data: dict) -> Config:
    return _build(Config, data, "")


def load_config(path: Path | str) -> Config:
    return config_from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def to_plain(obj: Any) -> Any:
    """Dataclasses -> dicts, tuples -> lists (YAML/JSON friendly)."""
    if is_dataclass(obj):
        return {f.name: to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [to_plain(v) for v in obj]
    return obj
