# halprobe MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that trains per-layer linear probes on Llama-3.2-3B-Instruct's last-prompt-token hidden states to predict PopQA answer errors before generation, and compare them against log-prob, entropy and popularity/metadata shortcut baselines.

**Architecture:** A linear chain of stages: data → generate → label → extract → probe → baselines → report. Each stage writes to a content-addressed cache directory `artifacts/<stage>/<key>/`, where the key hashes the relevant config fields plus the upstream keys. Writes go to `<key>.tmp`, and only a completed directory is renamed into place. GPU stages (generate, extract) resume after an interruption. The report stage writes a timestamped, committed `results/<experiment>/<run_id>/` with the config, metrics, environment info and figures.

**Tech Stack:** Python 3.11, uv, PyTorch 2.9.1 (AMD TheRock gfx103X nightly on Windows, CPU wheel on Linux CI), transformers 4.57.x, tokenizers, huggingface_hub, scikit-learn, numpy, matplotlib, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-halprobe-mvp-design.md`

## Global Constraints

- Python `>=3.11,<3.12`.
- torch is pinned exactly: `torch==2.9.1+rocm7.14.0a20260612` on `sys_platform == 'win32'`, from `https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/` (together with `rocm[libraries]`, `rocm-sdk-core` and `rocm-sdk-libraries-gfx103x-all`, all `==7.14.0a20260612`), and `torch==2.9.1+cpu` on `sys_platform == 'linux'`, from `https://download.pytorch.org/whl/cpu`. Every other package comes from PyPI.
- `transformers>=4.57,<4.58`. The `datasets` library is **not** used; PopQA is fetched as a TSV with `huggingface_hub.hf_hub_download` at a pinned revision.
- The venv lives at `C:\venvs\halprobe` via `UV_PROJECT_ENVIRONMENT`, because Windows long paths are disabled on this machine. Every `uv` command in this plan assumes that variable is set: Task 1 sets it persistently, and in Git Bash you `export UV_PROJECT_ENVIRONMENT='C:\venvs\halprobe'`.
- Model: `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`. Dataset: `akariasai/PopQA`, revision `098765c79ea10a2cb19c828324e33281b8336ec0`, file `test.tsv`.
- The GPU is selected by name hint `"6700"`. `device: auto` without a matching GPU raises; there is never a silent CPU fallback.
- Label convention everywhere: `y = 1` means the answer is **incorrect** (an error), and a higher score means more likely an error.
- Model selection (C, best layer) uses validation only. The test-set maximum over layers is never reported as a result.
- No fabricated or hand-typed result numbers. README and `docs/results.md` tables are rendered from `metrics.json` by `scripts/render_results.py`, which refuses runs with `label_mode != "alias"`.
- `labeling.mode: pseudo` exists only for the tiny CPU config (a random model answers nothing correctly). Its outputs are never reported.
- All repo text (code, docs, commit messages) is in English.
- Seeds come from `cfg.seed` (Python, numpy, torch, sklearn `random_state`).
- The test suite runs offline on CPU, in about 60 s; `pytest` excludes `@pytest.mark.gpu` by default.
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The commit commands below add it with a second `-m`.

## Review Focus

1. **Llama-3.2's chat template inserts *today's* date** into the system turn unless `date_string` is passed. Prompts, and therefore every generation and hidden state, would silently change from day to day. Pinned by `test_render_prompt_uses_configured_date` (Task 8); the tiny template reproduces the `strftime_now` fallback.
2. **Tokenizing already-templated text with `add_special_tokens=True` adds a second BOS** for Llama tokenizers. That shifts every position and changes the hidden states. Pinned by `test_encode_prompts_single_bos_and_left_padding` (Task 8); the tiny tokenizer has a Llama-style BOS post-processor so the test is sensitive.
3. **Left-padded batches must give the same last-token hidden state as unbatched inputs.** Padding, masking and fully masked pad rows must not leak into real positions. Pinned by `test_batched_states_match_unbatched` (Task 10), on a model with sharp attention (`initializer_range=0.2`) so masking errors would be visible.
4. **Empty answers** (the first generated token is a newline or EOT) give a NaN mean log-prob, which must not reach the metrics. Pinned by `test_postprocess_empty_when_first_token_stops` (Task 9), `test_summarize_rejects_non_finite_scores` (Task 11) and `test_fill_nan_as_max` (Task 13).
5. **Windows cannot rename a stage `.tmp` directory while a memmap inside it is still open.** Pinned by `test_stage_commit_after_extraction` (Task 10); `StateStore.close()` must release the mapping.

---

## File Structure

```
pyproject.toml, uv.lock, .python-version, .gitignore, README.md
.github/workflows/tests.yml
configs/mvp_popqa_llama3b.yaml      # Task 4
configs/tiny_cpu.yaml               # Task 17
src/halprobe/
  __init__.py      # version (Task 1)
  device.py        # resolve_device, seed_everything (Task 2)
  envcheck.py      # GPU/CPU environment checks (Task 2)
  text.py          # normalize, contains_phrase (Task 3)
  labeling.py      # label_answer, pseudo_label (Task 3)
  config.py        # typed strict config (Task 4)
  cache.py         # stage keys, StageDir, atomic I/O, jsonl (Task 5)
  data.py          # PopQA load/dedupe/split, Example, split_views (Task 6)
  tiny.py          # offline tiny tokenizer/model/TSV (Task 7)
  prompts.py       # few-shot chat prompts, encoding (Task 8)
  models.py        # model/tokenizer loading (Task 8)
  batching.py      # OOM back-off batching (Task 9)
  generation.py    # greedy generation + logprobs (Task 9)
  extraction.py    # resumable memmap hidden states (Task 10)
  metrics.py       # AUROC/AUPRC/ECE/risk-coverage/bootstrap (Task 11)
  probes.py        # per-layer logistic probes (Task 12)
  baselines.py     # log-prob, entropy, popularity, metadata (Task 13)
  analysis.py      # shortcut + sensitivity analyses (Task 14)
  plotting.py      # figures (Task 15)
  report.py        # consistency, tables, build_report, render_results (Task 16)
  pipeline.py      # stage orchestration (Task 17)
  __main__.py      # CLI (Task 17)
  review.py        # manual review export/scoring (Task 18)
scripts/env_check.py                 # Task 2
scripts/export_review_sample.py      # Task 18
scripts/score_review.py              # Task 18
scripts/render_results.py            # Task 18
tests/                               # one test module per source module
docs/results.md                      # Task 20
```

---

### Task 1: Project scaffold, pinned environment, CI

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `README.md`, `src/halprobe/__init__.py`, `tests/test_smoke.py`, `.github/workflows/tests.yml`
- Generated: `uv.lock`

**Interfaces:**
- Produces: the importable package `halprobe` with `__version__ = "0.1.0"`, and a working `uv run pytest`.

- [ ] **Step 1: Install uv and set the venv location**

Run in Git Bash:
```bash
python -m pip install --user uv
python -m uv --version
setx UV_PROJECT_ENVIRONMENT "C:\venvs\halprobe"
export UV_PROJECT_ENVIRONMENT='C:\venvs\halprobe'
```
Expected: a uv version is printed, and `setx` prints `SUCCESS: Specified value was saved.` If `uv` is not on PATH afterwards, use `python -m uv` wherever this plan says `uv`.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "halprobe"
version = "0.1.0"
description = "Linear probes on pre-generation hidden states for hallucination detection in small LLMs"
readme = "README.md"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch==2.9.1+rocm7.14.0a20260612; sys_platform == 'win32'",
    "rocm[libraries]==7.14.0a20260612; sys_platform == 'win32'",
    "rocm-sdk-core==7.14.0a20260612; sys_platform == 'win32'",
    "rocm-sdk-libraries-gfx103x-all==7.14.0a20260612; sys_platform == 'win32'",
    "torch==2.9.1+cpu; sys_platform == 'linux'",
    "transformers>=4.57,<4.58",
    "tokenizers>=0.22,<0.23",
    "huggingface-hub>=0.34,<1.0",
    "scikit-learn>=1.5",
    "numpy>=2.0",
    "matplotlib>=3.9",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = ["pytest>=8.3"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/halprobe"]

[tool.uv]
environments = ["sys_platform == 'win32'", "sys_platform == 'linux'"]

[tool.uv.sources]
torch = [
    { index = "therock-gfx103x", marker = "sys_platform == 'win32'" },
    { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]
rocm = { index = "therock-gfx103x" }
rocm-sdk-core = { index = "therock-gfx103x" }
rocm-sdk-libraries-gfx103x-all = { index = "therock-gfx103x" }

[[tool.uv.index]]
name = "therock-gfx103x"
url = "https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gpu: requires the AMD GPU (run with: pytest -m gpu)"]
addopts = "-m 'not gpu'"
```

- [ ] **Step 3: Write the small files**

`.python-version`:
```
3.11
```

`.gitignore`:
```
artifacts/
results/tiny_cpu/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
```

`README.md` (replaced in Task 20):
```markdown
# halprobe

Linear probes on a small LLM's pre-generation hidden states for predicting factual errors. Work in progress; see `docs/superpowers/specs/2026-10-12-halprobe-mvp-design.md`.
```

`src/halprobe/__init__.py`:
```python
"""Linear probes for pre-generation hallucination detection in small LLMs."""

__version__ = "0.1.0"
```

`.github/workflows/tests.yml`:
```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.11"
      - run: uv sync --locked
      - run: uv run pytest -q
```

- [ ] **Step 4: Write the failing smoke test**

`tests/test_smoke.py`:
```python
import halprobe


def test_package_version():
    assert halprobe.__version__ == "0.1.0"


def test_torch_is_the_pinned_build():
    import torch

    assert torch.__version__.startswith("2.9.1")
```

- [ ] **Step 5: Lock and sync**

Run:
```bash
uv lock
uv sync
```
Expected: `uv lock` writes `uv.lock`, and `uv sync` installs the packages into `C:\venvs\halprobe` (a download of about 4 GB on Windows).

If `uv lock` fails, for example while building the `rocm` sdist or because an index serves no file for the pinned version: **stop**, report the exact error to the user, and do not change the pins on your own.

- [ ] **Step 6: Run the tests and check the GPU build**

Run:
```bash
uv run pytest -v
uv run python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```
Expected: 2 passed. The second command prints `2.9.1+rocm7.14.0a20260612 7.14... True ['AMD Radeon RX 6700 XT', 'AMD Radeon(TM) Graphics']`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore README.md src/halprobe/__init__.py tests/test_smoke.py .github/workflows/tests.yml
git commit -m "build: scaffold halprobe package with pinned uv environment and CI" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Device selection and environment check

**Files:**
- Create: `src/halprobe/device.py`, `src/halprobe/envcheck.py`, `scripts/env_check.py`
- Test: `tests/test_device.py`, `tests/test_envcheck.py`

**Interfaces:**
- Produces: `resolve_device(spec: str, gpu_name_hint: str | None = None) -> torch.device`, `seed_everything(seed: int) -> None`, `envcheck.main(argv: list[str] | None = None) -> int`, `envcheck.kernel_checks(device: torch.device) -> bool`, `envcheck.tiny_cpu_check() -> bool`

- [ ] **Step 1: Write the failing device tests**

`tests/test_device.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.device'`.

- [ ] **Step 3: Implement `src/halprobe/device.py`**

```python
"""Device selection and global seeding."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(spec: str, gpu_name_hint: str | None = None) -> torch.device:
    """Resolve 'cpu', 'cuda:N' or 'auto' (GPU chosen by name hint) to a torch device.

    'auto' never falls back to CPU: a real run on CPU would silently take hours.
    """
    if spec == "cpu":
        return torch.device("cpu")
    if not (spec == "auto" or spec.startswith("cuda:")):
        raise ValueError(f"unknown device spec {spec!r}; expected 'auto', 'cpu' or 'cuda:N'")
    if not torch.cuda.is_available():
        raise RuntimeError(f"device={spec!r} requested but no GPU is visible to torch; set device: cpu explicitly")
    count = torch.cuda.device_count()
    if spec.startswith("cuda:"):
        if int(spec.split(":", 1)[1]) >= count:
            raise RuntimeError(f"{spec} requested but only {count} GPU(s) are visible")
        return torch.device(spec)
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if gpu_name_hint is None:
        return torch.device("cuda:0")
    for index, name in enumerate(names):
        if gpu_name_hint.lower() in name.lower():
            return torch.device(f"cuda:{index}")
    raise RuntimeError(f"no GPU name contains {gpu_name_hint!r}; visible devices: {names}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

- [ ] **Step 4: Run the device tests**

Run: `uv run pytest tests/test_device.py -v`
Expected: 9 passed.

- [ ] **Step 5: Write the failing envcheck tests**

`tests/test_envcheck.py`:
```python
import pytest

from halprobe.device import resolve_device
from halprobe.envcheck import kernel_checks, main, tiny_cpu_check


def test_tiny_cpu_check_passes():
    assert tiny_cpu_check()


def test_main_cpu_mode_returns_zero():
    assert main(["--cpu"]) == 0


@pytest.mark.gpu
def test_kernel_checks_on_gpu():
    assert kernel_checks(resolve_device("auto", "6700"))
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest tests/test_envcheck.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.envcheck'`.

- [ ] **Step 7: Implement `src/halprobe/envcheck.py`**

```python
"""Environment check: GPU visibility, GPU-vs-CPU numerical agreement, model load, VRAM."""

from __future__ import annotations

import argparse
import platform
import time

import torch
import torch.nn.functional as F

from halprobe.device import resolve_device

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
QUESTIONS = (
    "Who was the director of Jaws?",
    "In what city was Marie Curie born?",
    "What is the capital of Burkina Faso?",
)


def kernel_checks(device: torch.device) -> bool:
    """Compare fp16 kernels on `device` against an fp32 CPU reference."""
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, dtype=torch.float16)
    b = torch.randn(1024, 1024, dtype=torch.float16)
    h = torch.randn(4, 256, 1024, dtype=torch.float16)
    w = torch.ones(1024, dtype=torch.float16)
    x = torch.randn(2, 8, 64, 64, dtype=torch.float16)
    checks = {
        "matmul": (torch.matmul, (a, b)),
        "softmax": (lambda t: torch.softmax(t, -1), (h,)),
        "layer_norm": (lambda t, g: F.layer_norm(t, (1024,), g), (h, w)),
        "sdpa_causal": (lambda q: F.scaled_dot_product_attention(q, q, q, is_causal=True), (x,)),
        "silu": (F.silu, (h,)),
    }
    ok = True
    for name, (fn, args) in checks.items():
        reference = fn(*[t.float() for t in args])
        result = fn(*[t.to(device) for t in args]).float().cpu()
        err = ((result - reference).abs().max() / reference.abs().max().clamp_min(1e-6)).item()
        passed = err < 2e-2
        ok = ok and passed
        print(f"  {name:<12} rel_err={err:.2e} {'OK' if passed else 'FAIL'}")
    return ok


def gpu_report(device: torch.device) -> dict:
    """Matmul throughput and free/total VRAM on `device`."""
    a = torch.randn(2048, 2048, dtype=torch.float16, device=device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize(device)
    seconds = (time.perf_counter() - start) / 20
    free, total = torch.cuda.mem_get_info(device)
    return {"tflops_fp16": 2 * 2048**3 / seconds / 1e12, "vram_free_gib": free / 2**30, "vram_total_gib": total / 2**30}


@torch.no_grad()
def model_check(model_id: str, revision: str, device: torch.device) -> bool:
    """Load the model in fp16 on `device` and fp32 on CPU; compare hidden states and greedy answers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, date_string="26 Jul 2024"
        )
        for q in QUESTIONS
    ]
    torch.cuda.reset_peak_memory_stats(device)
    gpu = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float16).to(device).eval()
    cpu = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float32).eval()
    ok = True
    for question, prompt in zip(QUESTIONS, prompts):
        results = {}
        for name, model, dev in (("gpu", gpu, device), ("cpu", cpu, torch.device("cpu"))):
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
            out = model(**enc, output_hidden_states=True)
            states = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
            gen = model.generate(
                **enc, max_new_tokens=16, do_sample=False, temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id
            )
            answer = tokenizer.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            results[name] = (states, answer)
        cos = F.cosine_similarity(results["gpu"][0][1:], results["cpu"][0][1:], dim=-1).min().item()
        same = results["gpu"][1] == results["cpu"][1]
        ok = ok and cos > 0.99
        print(f"  {question}\n    gpu={results['gpu'][1]!r} cpu={results['cpu'][1]!r} "
              f"{'same' if same else 'DIFFERENT (warning)'} | min hidden cosine {cos:.4f}")
    print(f"  peak VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")
    return ok


def tiny_cpu_check() -> bool:
    """Forward pass of a tiny random Llama on CPU (no downloads)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        out = model(torch.randint(0, 128, (2, 10)), output_hidden_states=True)
    ok = out.logits.shape == (2, 10, 128) and len(out.hidden_states) == 3 and bool(torch.isfinite(out.logits).all())
    print(f"  tiny CPU forward pass: {'OK' if ok else 'FAIL'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the GPU/ROCm setup for halprobe.")
    parser.add_argument("--cpu", action="store_true", help="CPU-only check with a tiny random model")
    parser.add_argument("--gpu-name-hint", default="6700")
    parser.add_argument("--skip-model", action="store_true", help="skip loading the real model")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args(argv)

    print(f"python {platform.python_version()} | torch {torch.__version__} | hip {torch.version.hip} "
          f"| GPU available {torch.cuda.is_available()}")
    if args.cpu:
        ok = tiny_cpu_check()
    else:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [cuda:{i}] {props.name} | {getattr(props, 'gcnArchName', 'n/a')} | {props.total_memory / 2**30:.2f} GiB")
        device = resolve_device("auto", args.gpu_name_hint)
        print(f"selected {device} ({torch.cuda.get_device_name(device)})")
        ok = kernel_checks(device)
        info = gpu_report(device)
        print(f"  matmul {info['tflops_fp16']:.1f} TFLOP/s | VRAM free {info['vram_free_gib']:.2f} / {info['vram_total_gib']:.2f} GiB")
        if not args.skip_model:
            ok = model_check(args.model, args.revision, device) and ok
    print("ENV CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1
```

`scripts/env_check.py`:
```python
"""Environment check for halprobe. Usage: uv run python scripts/env_check.py [--cpu] [--skip-model]"""

from halprobe.envcheck import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_device.py tests/test_envcheck.py -v`
Expected: 11 passed, 1 deselected (gpu).

- [ ] **Step 9: Run the real GPU checks**

Run:
```bash
uv run pytest -m gpu -v
uv run python scripts/env_check.py --skip-model
```
Expected: the gpu test passes, and the script lists both devices, selects `cuda:0 (AMD Radeon RX 6700 XT)`, shows all kernels `OK`, and ends with `ENV CHECK: PASS`.

- [ ] **Step 10: Run the real model check (needs Hugging Face access)**

The model is gated. The **user** must accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and log in once. Ask them to run `uv run hf auth login` in their own terminal. Then run:
```bash
uv run python scripts/env_check.py
```
Expected: three questions, each with min hidden cosine > 0.99, then a peak VRAM figure, then `ENV CHECK: PASS`. The first run downloads about 6.4 GB, and the fp32 CPU copy needs about 13 GB of RAM. If any answer shows `DIFFERENT (warning)`, report it to the user; if the cosine check fails, **stop**.

- [ ] **Step 11: Commit**

```bash
git add src/halprobe/device.py src/halprobe/envcheck.py scripts/env_check.py tests/test_device.py tests/test_envcheck.py
git commit -m "feat: add device resolution by GPU name and environment check script" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Answer normalization and labeling

**Files:**
- Create: `src/halprobe/text.py`, `src/halprobe/labeling.py`
- Test: `tests/test_labeling.py`

**Interfaces:**
- Produces: `text.normalize(text: str) -> str`, `text.contains_phrase(haystack: str, needle: str) -> bool`, `labeling.RULES_VERSION: int`, `labeling.LabelResult(label, reason)` with property `.y -> int | None`, `labeling.label_answer(answer: str, aliases: Iterable[str], finish_reason: str) -> LabelResult`, `labeling.pseudo_label(example_id: str) -> LabelResult`

- [ ] **Step 1: Write the failing tests**

`tests/test_labeling.py`:
```python
import pytest

from halprobe.labeling import LabelResult, label_answer, pseudo_label
from halprobe.text import contains_phrase, normalize


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  The Émile-Zola  Street! ", "emile zola street"),
        ("Emily Brontë", "emily bronte"),
        ("A Tale of Two Cities", "tale of two cities"),
        ("polit.", "polit"),
        ("Don't", "don t"),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_contains_phrase_respects_word_boundaries():
    assert contains_phrase("he is a pol", "pol")
    assert not contains_phrase("police officer", "pol")
    assert contains_phrase("steven spielberg directed jaws", "steven spielberg")
    assert not contains_phrase("anything", "")


def test_correct_when_alias_contained_in_sentence():
    assert label_answer("Steven Spielberg directed Jaws.", ["Steven Spielberg"], "stop") == LabelResult("correct", None)


def test_correct_with_accents_and_case():
    assert label_answer("emily bronte", ["Emily Brontë"], "stop").label == "correct"


def test_short_alias_does_not_match_inside_word():
    result = label_answer("police officer", ["politician", "pol"], "stop")
    assert result == LabelResult("incorrect", None)


def test_punctuation_only_alias_never_matches_everything():
    assert label_answer("anything at all", ["."], "stop").label == "incorrect"


def test_empty_answer_is_ambiguous_even_if_truncated():
    assert label_answer("   ", ["Paris"], "length") == LabelResult("ambiguous", "empty")


@pytest.mark.parametrize("answer", ["I don't know.", "Unknown", "I'm not sure", "I cannot answer that"])
def test_refusals_are_ambiguous(answer):
    assert label_answer(answer, ["Paris"], "stop") == LabelResult("ambiguous", "refusal")


def test_truncated_without_match_is_ambiguous():
    assert label_answer("a very long rambling answer", ["Paris"], "length") == LabelResult("ambiguous", "truncated")


def test_correct_takes_precedence_over_truncation():
    assert label_answer("Paris is the city that", ["Paris"], "length").label == "correct"


def test_partial_overlap_is_ambiguous():
    assert label_answer("producer", ["film producer"], "stop") == LabelResult("ambiguous", "partial")


def test_shared_generic_word_is_not_partial():
    assert label_answer("classical music", ["pop music"], "stop") == LabelResult("incorrect", None)


def test_y_encoding():
    assert LabelResult("incorrect", None).y == 1
    assert LabelResult("correct", None).y == 0
    assert LabelResult("ambiguous", "empty").y is None


def test_pseudo_label_is_deterministic_and_uses_both_classes():
    labels = [pseudo_label(f"id{i}").label for i in range(50)]
    assert labels == [pseudo_label(f"id{i}").label for i in range(50)]
    assert set(labels) == {"correct", "incorrect"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_labeling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.labeling'`.

- [ ] **Step 3: Implement `src/halprobe/text.py`**

```python
"""Text normalization shared by data grouping and answer labeling."""

from __future__ import annotations

import re
import unicodedata

_ARTICLES = re.compile(r"\b(?:a|an|the)\b")


def normalize(text: str) -> str:
    """SQuAD-style normalization: strip accents, casefold, punctuation -> space, drop articles, collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contains_phrase(haystack: str, needle: str) -> bool:
    """Word-boundary containment of two normalized strings; an empty needle never matches."""
    if not needle:
        return False
    return f" {needle} " in f" {haystack} "
```

- [ ] **Step 4: Implement `src/halprobe/labeling.py`**

```python
"""Correct / incorrect / ambiguous labels from normalized alias matching."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from halprobe.text import contains_phrase, normalize

# Bump when the rules below change: it is part of the label stage's cache key.
RULES_VERSION = 1

_REFUSALS_RAW = (
    "i don't know", "i do not know", "unknown", "not sure", "unsure", "cannot", "can't",
    "no information", "not available", "unclear", "no answer",
)
REFUSALS = tuple(normalize(p) for p in _REFUSALS_RAW)
STOPWORDS = frozenset({"of", "and", "or", "in", "on", "at", "to", "for", "by", "with", "from", "is", "was", "as", "s"})
# Category words shared by many unrelated gold answers ("pop music" vs "classical music").
GENERIC_WORDS = frozenset({
    "music", "film", "movie", "player", "series", "game", "album", "novel", "band", "song", "show",
    "television", "tv", "city", "country", "state", "province", "district", "county", "language",
    "fiction", "drama",
})


@dataclass(frozen=True)
class LabelResult:
    label: str  # "correct" | "incorrect" | "ambiguous"
    reason: str | None  # set only for ambiguous: empty | refusal | truncated | partial

    @property
    def y(self) -> int | None:
        """Probe target: 1 = incorrect (error), 0 = correct, None = ambiguous."""
        return {"correct": 0, "incorrect": 1}.get(self.label)


def _content_words(normalized: str) -> set[str]:
    return set(normalized.split()) - STOPWORDS - GENERIC_WORDS


def label_answer(answer: str, aliases: Iterable[str], finish_reason: str) -> LabelResult:
    ans = normalize(answer)
    gold = [g for g in (normalize(a) for a in aliases) if g]
    if any(contains_phrase(ans, g) for g in gold):
        return LabelResult("correct", None)
    if not ans:
        return LabelResult("ambiguous", "empty")
    if any(contains_phrase(ans, r) for r in REFUSALS):
        return LabelResult("ambiguous", "refusal")
    if finish_reason == "length":
        return LabelResult("ambiguous", "truncated")
    words = _content_words(ans)
    if any(words & _content_words(g) for g in gold):
        return LabelResult("ambiguous", "partial")
    return LabelResult("incorrect", None)


def pseudo_label(example_id: str) -> LabelResult:
    """TEST-ONLY deterministic labels for random tiny models. Never used for reported results."""
    bit = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest(), 16) % 2
    return LabelResult("incorrect" if bit else "correct", None)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_labeling.py -v`
Expected: all pass (22 tests).

- [ ] **Step 6: Commit**

```bash
git add src/halprobe/text.py src/halprobe/labeling.py tests/test_labeling.py
git commit -m "feat: add answer normalization and alias-based labeling with ambiguity reasons" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Typed, strict configuration

**Files:**
- Create: `src/halprobe/config.py`, `configs/mvp_popqa_llama3b.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `ConfigError`, the frozen dataclasses `DataConfig`, `ModelConfig`, `ShotConfig`, `PromptConfig`, `GenerationConfig`, `LabelingConfig`, `ExtractionConfig`, `ProbeConfig`, `EvaluationConfig` and `Config`; `config_from_dict(data: dict) -> Config`, `load_config(path: Path | str) -> Config`, `to_plain(obj) -> dict | list | scalar`.
- Field names used later: `cfg.experiment, cfg.seed, cfg.device, cfg.gpu_name_hint, cfg.data.{repo_id,revision,filename,local_path,split_ratios}, cfg.model.{name_or_path,revision,dtype}, cfg.prompt.{system,shots,date_string}, shot.{subject,question,answer}, cfg.generation.{max_new_tokens,batch_size}, cfg.labeling.mode, cfg.extraction.{batch_size,consistency_threshold}, cfg.probe.{C_grid,max_iter,min_class_count}, cfg.evaluation.{n_bootstrap,ece_bins,min_per_class_relation}`

- [ ] **Step 1: Write the MVP config**

`configs/mvp_popqa_llama3b.yaml`:
```yaml
# MVP: PopQA, Llama-3.2-3B-Instruct, probes at the last prompt token.
experiment: mvp_popqa_llama3b
seed: 0
device: auto
gpu_name_hint: "6700"
data:
  repo_id: akariasai/PopQA
  revision: 098765c79ea10a2cb19c828324e33281b8336ec0
  filename: test.tsv
  split_ratios: [0.7, 0.15, 0.15]
model:
  name_or_path: meta-llama/Llama-3.2-3B-Instruct
  revision: 0cb88a4f764b7a12671c53f0838cd831a0843b95
  dtype: float16
prompt:
  system: "Answer the question with a short phrase (a name, place, date, or category). Do not explain."
  date_string: "26 Jul 2024"
  shots:
    - {subject: "Ada Lovelace", question: "What is Ada Lovelace's occupation?", answer: "mathematician"}
    - {subject: "Spirited Away", question: "Who was the director of Spirited Away?", answer: "Hayao Miyazaki"}
    - {subject: "Wuthering Heights", question: "Who is the author of Wuthering Heights?", answer: "Emily Brontë"}
    - {subject: "Blade Runner", question: "What genre is Blade Runner?", answer: "science fiction"}
generation:
  max_new_tokens: 32
  batch_size: 32
labeling:
  mode: alias
extraction:
  batch_size: 32
  consistency_threshold: 0.99
probe:
  C_grid: [1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0]
  max_iter: 5000
  min_class_count: 20
evaluation:
  n_bootstrap: 1000
  ece_bins: 15
  min_per_class_relation: 20
```

- [ ] **Step 2: Write the failing tests**

`tests/test_config.py`:
```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.config'`.

- [ ] **Step 4: Implement `src/halprobe/config.py`**

```python
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
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: all pass (12 tests).

- [ ] **Step 6: Commit**

```bash
git add src/halprobe/config.py configs/mvp_popqa_llama3b.yaml tests/test_config.py
git commit -m "feat: add strict typed YAML configuration and MVP experiment config" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Stage cache and crash-safe I/O

**Files:**
- Create: `src/halprobe/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces: `COMPLETE_MARKER`, `json_default(obj)`, `stage_key(payload: dict) -> str` (12 hex chars), `StageDir(root: Path, stage: str, key: str)` with `.stage .key .final .tmp`, `.is_complete() -> bool`, `.begin() -> Path` (the tmp dir; existing partial content is kept for resume) and `.commit() -> Path` (the final dir); `atomic_write_text(path, text)`, `atomic_write_json(path, obj)`, `atomic_save_npy(path, array)`, `write_jsonl(path, rows)`, `append_jsonl(path, rows)`, `read_jsonl(path, tolerate_partial_tail=False) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

`tests/test_cache.py`:
```python
import json

import numpy as np
import pytest

from halprobe.cache import (
    COMPLETE_MARKER,
    StageDir,
    append_jsonl,
    atomic_save_npy,
    atomic_write_json,
    read_jsonl,
    stage_key,
    write_jsonl,
)


def test_stage_key_is_stable_and_order_independent():
    a = stage_key({"x": 1, "y": [1, 2], "z": {"b": 2, "a": 1}})
    b = stage_key({"z": {"a": 1, "b": 2}, "y": [1, 2], "x": 1})
    assert a == b and len(a) == 12


def test_stage_key_changes_with_content():
    assert stage_key({"x": 1}) != stage_key({"x": 2})


def test_stage_dir_lifecycle(tmp_path):
    sd = StageDir(tmp_path, "data", "abc123")
    assert not sd.is_complete()
    work = sd.begin()
    assert work == tmp_path / "data" / "abc123.tmp"
    (work / "out.txt").write_text("hello", encoding="utf-8")
    assert not sd.is_complete()  # a .tmp dir is never a cache hit
    final = sd.commit()
    assert final == tmp_path / "data" / "abc123"
    assert sd.is_complete() and (final / COMPLETE_MARKER).exists()
    assert (final / "out.txt").read_text(encoding="utf-8") == "hello"
    assert not sd.tmp.exists()


def test_begin_keeps_partial_content_for_resume(tmp_path):
    sd = StageDir(tmp_path, "generate", "k")
    (sd.begin() / "partial.jsonl").write_text("{}\n", encoding="utf-8")
    assert (sd.begin() / "partial.jsonl").exists()


def test_commit_replaces_stale_final_without_marker(tmp_path):
    sd = StageDir(tmp_path, "data", "k")
    sd.final.mkdir(parents=True)
    (sd.final / "stale.txt").write_text("old", encoding="utf-8")
    sd.begin()
    sd.commit()
    assert not (sd.final / "stale.txt").exists() and sd.is_complete()


def test_atomic_json_and_numpy_types(tmp_path):
    path = tmp_path / "a.json"
    atomic_write_json(path, {"n": np.int64(3), "x": np.float32(0.5), "arr": np.array([1, 2])})
    assert json.loads(path.read_text(encoding="utf-8")) == {"n": 3, "x": 0.5, "arr": [1, 2]}
    assert not (tmp_path / "a.json.part").exists()


def test_atomic_save_npy_round_trip(tmp_path):
    path = tmp_path / "done.npy"
    atomic_save_npy(path, np.array([True, False]))
    atomic_save_npy(path, np.array([True, True]))
    assert np.load(path).tolist() == [True, True]


def test_jsonl_write_append_read(tmp_path):
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [{"id": "a"}])
    append_jsonl(path, [{"id": "b"}, {"id": "c"}])
    assert [r["id"] for r in read_jsonl(path)] == ["a", "b", "c"]


def test_read_jsonl_partial_tail(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b", "ans', encoding="utf-8")
    assert read_jsonl(path, tolerate_partial_tail=True) == [{"id": "a"}]
    with pytest.raises(ValueError, match="line 2"):
        read_jsonl(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.cache'`.

- [ ] **Step 3: Implement `src/halprobe/cache.py`**

```python
"""Content-addressed stage directories and crash-safe file writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

COMPLETE_MARKER = "_COMPLETE"


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def stage_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class StageDir:
    """artifacts/<stage>/<key>/ (complete) and artifacts/<stage>/<key>.tmp/ (in progress)."""

    def __init__(self, root: Path | str, stage: str, key: str):
        self.stage = stage
        self.key = key
        self.final = Path(root) / stage / key
        self.tmp = Path(root) / stage / f"{key}.tmp"

    def is_complete(self) -> bool:
        return (self.final / COMPLETE_MARKER).is_file()

    def begin(self) -> Path:
        self.tmp.mkdir(parents=True, exist_ok=True)
        return self.tmp

    def commit(self) -> Path:
        (self.tmp / COMPLETE_MARKER).write_text("ok\n", encoding="utf-8")
        if self.final.exists():
            shutil.rmtree(self.final)
        os.replace(self.tmp, self.final)
        return self.final


def atomic_write_text(path: Path | str, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)


def atomic_write_json(path: Path | str, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=json_default) + "\n")


def atomic_save_npy(path: Path | str, array: np.ndarray) -> None:
    path = Path(path)
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)


def _dumps(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, default=json_default)


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    atomic_write_text(path, "".join(_dumps(r) + "\n" for r in rows))


def append_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(_dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path | str, tolerate_partial_tail: bool = False) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if tolerate_partial_tail and i == len(lines) - 1:
                break
            raise ValueError(f"{path}: invalid JSON on line {i + 1}") from None
    return rows
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/cache.py tests/test_cache.py
git commit -m "feat: add content-addressed stage dirs and crash-safe file writes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: PopQA loading, dedupe and entity-grouped split

**Files:**
- Create: `src/halprobe/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `text.normalize`, `cache.write_jsonl`, `cache.read_jsonl`
- Produces: `POPQA_COLUMNS: tuple[str, ...]`, `SPLITS = ("train", "val", "test")`, the frozen `Example(id, question, subject, relation, answers: tuple[str, ...], s_pop: int, split)`, `fetch_popqa(repo_id, filename, revision) -> Path`, `load_popqa_rows(path) -> list[dict]`, `subject_key(subject) -> str`, `dedupe(rows) -> tuple[list[dict], dict]`, `assign_splits(group_keys, ratios, seed) -> list[str]`, `check_no_leakage(examples)`, `check_shot_subjects(subjects, shot_subjects)`, `build_examples(rows, ratios, seed, shot_subjects) -> tuple[list[Example], dict]`, `save_examples(path, examples)`, `load_examples(path) -> list[Example]`, the frozen `SplitView(idx_all, idx_lab, y, pos_lab)` and `split_views(examples, labels_by_id) -> dict[str, SplitView]`

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:
```python
import csv
import json
from collections import Counter

import numpy as np
import pytest

from halprobe.data import (
    POPQA_COLUMNS,
    Example,
    assign_splits,
    build_examples,
    check_no_leakage,
    check_shot_subjects,
    dedupe,
    load_examples,
    load_popqa_rows,
    save_examples,
    split_views,
)


def _row(id, subj, subj_id, prop, question, answers, s_pop=100):
    return {"id": str(id), "subj": subj, "subj_id": str(subj_id), "prop": prop, "question": question,
            "possible_answers": json.dumps(answers), "s_pop": str(s_pop)}


def test_load_popqa_rows_checks_columns(tmp_path):
    good = tmp_path / "good.tsv"
    with open(good, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POPQA_COLUMNS, delimiter="\t")
        w.writeheader()
        w.writerow({c: "1" for c in POPQA_COLUMNS})
    assert len(load_popqa_rows(good)) == 1
    bad = tmp_path / "bad.tsv"
    bad.write_text("id\tsubj\n1\tx\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_popqa_rows(bad)


def test_dedupe_merges_same_entity_duplicates():
    rows = [
        _row(1, "Taxi", 10, "genre", "What genre is Taxi?", ["comedy film"]),
        _row(2, "Taxi", 10, "genre", "What genre is Taxi? ", ["action film", "comedy film"]),
    ]
    merged, report = dedupe(rows)
    assert len(merged) == 1
    assert merged[0]["answers"] == ("comedy film", "action film")
    assert merged[0]["id"] == "1"
    assert report["groups_merged"] == 1 and report["groups_dropped_multi_entity"] == 0


def test_dedupe_drops_same_name_different_entities():
    rows = [
        _row(1, "Will Smith", 1, "occupation", "What is Will Smith's occupation?", ["film producer"], 345519),
        _row(2, "Will Smith", 2, "occupation", "What is Will Smith's occupation?", ["screenwriter"], 5209),
        _row(3, "Paris", 3, "country", "In what country is Paris?", ["France"]),
    ]
    merged, report = dedupe(rows)
    assert [m["question"] for m in merged] == ["In what country is Paris?"]
    assert report == {"rows_in": 3, "question_groups": 2, "groups_merged": 0,
                      "groups_dropped_multi_entity": 1, "examples_out": 1}


def test_assign_splits_is_deterministic_and_matches_ratios():
    keys = [f"s{i}" for i in range(1000)]
    a = assign_splits(keys, (0.7, 0.15, 0.15), seed=0)
    assert a == assign_splits(keys, (0.7, 0.15, 0.15), seed=0)
    assert a != assign_splits(keys, (0.7, 0.15, 0.15), seed=1)
    assert Counter(a) == {"train": 700, "val": 150, "test": 150}


def test_same_subject_string_stays_in_one_split():
    rows = [_row(i, f"Subject {i % 50}", 1000 + i, "occupation", f"Question {i}?", ["x"]) for i in range(300)]
    rows.append(_row(999, "Subject 3", 77, "country", "Where is Subject 3?", ["y"]))  # same name, other entity
    examples, _ = build_examples(rows, (0.7, 0.15, 0.15), seed=0, shot_subjects=[])
    by_subject = {}
    for ex in examples:
        by_subject.setdefault(ex.subject, set()).add(ex.split)
    assert all(len(splits) == 1 for splits in by_subject.values())


def test_check_no_leakage_raises():
    examples = [
        Example("1", "q1", "The Beatles", "genre", ("rock",), 1, "train"),
        Example("2", "q2", "Beatles", "genre", ("rock",), 1, "test"),
    ]
    with pytest.raises(AssertionError, match="beatles"):
        check_no_leakage(examples)


def test_shot_subject_guard():
    check_shot_subjects({"Paris", "Berlin"}, ["Ada Lovelace"])
    with pytest.raises(ValueError, match="paris"):
        check_shot_subjects({"Paris"}, ["paris"])


def test_build_examples_report_and_roundtrip(tmp_path):
    rows = [_row(i, f"S{i}", i, "genre" if i % 2 else "capital", f"Q{i}?", [f"a{i}"], 10 * i + 2) for i in range(40)]
    examples, report = build_examples(rows, (0.7, 0.15, 0.15), seed=0, shot_subjects=["Ada Lovelace"])
    assert report["examples_out"] == 40
    assert sum(report["split_sizes"].values()) == 40
    assert set(report["relations_by_split"]) == {"train", "val", "test"}
    save_examples(tmp_path / "e.jsonl", examples)
    assert load_examples(tmp_path / "e.jsonl") == examples


def test_split_views_exclude_ambiguous():
    examples = [
        Example("a", "q", "s1", "r", ("x",), 1, "test"),
        Example("b", "q", "s2", "r", ("x",), 1, "train"),
        Example("c", "q", "s3", "r", ("x",), 1, "test"),
        Example("d", "q", "s4", "r", ("x",), 1, "test"),
    ]
    labels = {"a": {"y": 1}, "b": {"y": 0}, "c": {"y": None}, "d": {"y": 0}}
    views = split_views(examples, labels)
    test = views["test"]
    assert test.idx_all.tolist() == [0, 2, 3]
    assert test.idx_lab.tolist() == [0, 3]
    assert test.y.tolist() == [1, 0]
    assert test.pos_lab.tolist() == [0, 2]
    assert views["val"].idx_all.size == 0 and views["val"].y.dtype == np.int64
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.data'`.

- [ ] **Step 3: Implement `src/halprobe/data.py`**

```python
"""PopQA loading, deduplication, entity-grouped splitting, and split index views."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from halprobe.cache import read_jsonl, write_jsonl
from halprobe.text import normalize

POPQA_COLUMNS = (
    "id", "subj", "prop", "obj", "subj_id", "prop_id", "obj_id", "s_aliases", "o_aliases", "s_uri", "o_uri",
    "s_wiki_title", "o_wiki_title", "s_pop", "o_pop", "question", "possible_answers",
)
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    subject: str
    relation: str
    answers: tuple[str, ...]
    s_pop: int
    split: str


def fetch_popqa(repo_id: str, filename: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision))


def load_popqa_rows(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = sorted(set(POPQA_COLUMNS) - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        return list(reader)


def subject_key(subject: str) -> str:
    return normalize(subject)


def _question_key(question: str) -> str:
    return " ".join(question.casefold().split())


def dedupe(rows: Sequence[dict]) -> tuple[list[dict], dict]:
    """One example per question text. Same-entity duplicates are merged (union of answers);
    question texts shared by different entities (same name, different subj_id) are dropped."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_question_key(row["question"])].append(row)
    merged, n_merged, n_dropped = [], 0, 0
    for group in groups.values():
        if len({r["subj_id"] for r in group}) > 1:
            n_dropped += 1
            continue
        if len(group) > 1:
            n_merged += 1
        answers: list[str] = []
        for r in group:
            for a in json.loads(r["possible_answers"]):
                if a not in answers:
                    answers.append(a)
        first = group[0]
        merged.append({"id": first["id"], "question": first["question"].strip(), "subject": first["subj"].strip(),
                       "relation": first["prop"], "answers": tuple(answers), "s_pop": int(first["s_pop"])})
    report = {"rows_in": len(rows), "question_groups": len(groups), "groups_merged": n_merged,
              "groups_dropped_multi_entity": n_dropped, "examples_out": len(merged)}
    return merged, report


def assign_splits(group_keys: Sequence[str], ratios: Sequence[float], seed: int) -> list[str]:
    unique = sorted(set(group_keys))
    order = np.random.default_rng(seed).permutation(len(unique))
    n_train = round(ratios[0] * len(unique))
    n_val = round(ratios[1] * len(unique))
    split_of = {}
    for rank, idx in enumerate(order):
        split_of[unique[idx]] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    return [split_of[k] for k in group_keys]


def check_no_leakage(examples: Iterable[Example]) -> None:
    seen: dict[str, str] = {}
    for ex in examples:
        key = subject_key(ex.subject)
        previous = seen.setdefault(key, ex.split)
        if previous != ex.split:
            raise AssertionError(f"subject {key!r} appears in both {previous} and {ex.split}")


def check_shot_subjects(subjects: Iterable[str], shot_subjects: Iterable[str]) -> None:
    present = {subject_key(s) for s in subjects}
    clashes = [s for s in shot_subjects if subject_key(s) in present]
    if clashes:
        raise ValueError(f"few-shot subjects occur in the dataset: {clashes}")


def build_examples(rows: Sequence[dict], ratios: Sequence[float], seed: int,
                   shot_subjects: Sequence[str]) -> tuple[list[Example], dict]:
    check_shot_subjects((r["subj"] for r in rows), shot_subjects)
    merged, report = dedupe(rows)
    splits = assign_splits([subject_key(m["subject"]) for m in merged], ratios, seed)
    examples = [Example(**m, split=s) for m, s in zip(merged, splits)]
    check_no_leakage(examples)
    report["subject_groups"] = len({subject_key(ex.subject) for ex in examples})
    report["split_sizes"] = dict(Counter(ex.split for ex in examples))
    report["relations_by_split"] = {s: dict(Counter(ex.relation for ex in examples if ex.split == s)) for s in SPLITS}
    return examples, report


def save_examples(path: Path | str, examples: Iterable[Example]) -> None:
    write_jsonl(path, [asdict(ex) for ex in examples])


def load_examples(path: Path | str) -> list[Example]:
    return [Example(**{**row, "answers": tuple(row["answers"])}) for row in read_jsonl(path)]


@dataclass(frozen=True)
class SplitView:
    idx_all: np.ndarray  # row indices (into the examples list) of every example in the split
    idx_lab: np.ndarray  # subset with an unambiguous label
    y: np.ndarray  # labels for idx_lab (1 = error)
    pos_lab: np.ndarray  # positions of idx_lab within idx_all


def split_views(examples: Sequence[Example], labels_by_id: dict[str, dict]) -> dict[str, SplitView]:
    views = {}
    for split in SPLITS:
        idx_all = np.array([i for i, ex in enumerate(examples) if ex.split == split], dtype=np.int64)
        ys = [labels_by_id[examples[i].id]["y"] for i in idx_all]
        pos_lab = np.array([p for p, y in enumerate(ys) if y is not None], dtype=np.int64)
        views[split] = SplitView(idx_all=idx_all, idx_lab=idx_all[pos_lab],
                                 y=np.array([ys[p] for p in pos_lab], dtype=np.int64), pos_lab=pos_lab)
    return views
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_data.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/data.py tests/test_data.py
git commit -m "feat: add PopQA loading, entity-safe dedupe and subject-grouped splits" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Tiny offline assets (tokenizer, random Llama, synthetic PopQA)

**Files:**
- Create: `src/halprobe/tiny.py`, `tests/conftest.py`
- Test: `tests/test_tiny.py`

**Interfaces:**
- Consumes: `data.POPQA_COLUMNS`
- Produces: `BOS, EOT, PAD, SPECIAL_TOKENS, CHAT_TEMPLATE, TINY_RELATIONS`, `write_tiny_popqa(path, n_rows=160, seed=0) -> list[dict]`, `build_tiny_tokenizer(corpus: list[str]) -> PreTrainedTokenizerFast`, `build_tiny_model(tokenizer) -> LlamaForCausalLM`, the frozen `TinyAssets(tsv: Path, model_dir: Path)`, `write_tiny_assets(out_dir) -> TinyAssets`; the pytest fixture `tiny_assets` (session scope)

- [ ] **Step 1: Write the fixture and the failing tests**

`tests/conftest.py`:
```python
import pytest

from halprobe.tiny import write_tiny_assets


@pytest.fixture(scope="session")
def tiny_assets(tmp_path_factory):
    return write_tiny_assets(tmp_path_factory.mktemp("tiny"))
```

`tests/test_tiny.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_tiny.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.tiny'`.

- [ ] **Step 3: Implement `src/halprobe/tiny.py`**

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_tiny.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/tiny.py tests/conftest.py tests/test_tiny.py
git commit -m "test: add offline tiny tokenizer, random Llama and synthetic PopQA assets" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Prompts and model loading

**Files:**
- Create: `src/halprobe/prompts.py`, `src/halprobe/models.py`
- Modify: `tests/conftest.py` (full replacement below)
- Test: `tests/test_prompts.py`, `tests/test_models.py`

**Interfaces:**
- Consumes: `config.PromptConfig`, `config.ModelConfig`, `config.ShotConfig`, `tiny.write_tiny_assets`
- Produces: `build_messages(question, prompt) -> list[dict[str, str]]`, `render_prompt(tokenizer, question, prompt) -> str`, `encode_prompts(tokenizer, prompts, device) -> dict[str, Tensor]` (keys `input_ids`, `attention_mask`), `position_ids_from_mask(attention_mask) -> Tensor`; `DTYPES`, `load_tokenizer(cfg: ModelConfig)`, `load_model(cfg: ModelConfig, device, weight_transform=None)`, `stop_token_ids(model, tokenizer) -> set[int]`; the session fixtures `tiny_model_config`, `tiny_tokenizer`, `tiny_model` and `tiny_prompt`

- [ ] **Step 1: Replace `tests/conftest.py`**

```python
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
```

- [ ] **Step 2: Write the failing tests**

`tests/test_prompts.py`:
```python
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
```

`tests/test_models.py`:
```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.models'` (raised from the conftest import).

- [ ] **Step 4: Implement `src/halprobe/prompts.py`**

```python
"""Few-shot chat prompt construction and tokenization."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from halprobe.config import PromptConfig


def build_messages(question: str, prompt: PromptConfig) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": prompt.system}]
    for shot in prompt.shots:
        messages.append({"role": "user", "content": shot.question})
        messages.append({"role": "assistant", "content": shot.answer})
    messages.append({"role": "user", "content": question})
    return messages


def render_prompt(tokenizer, question: str, prompt: PromptConfig) -> str:
    # date_string pins the date Llama-3.2's template would otherwise take from today's clock.
    return tokenizer.apply_chat_template(build_messages(question, prompt), tokenize=False,
                                         add_generation_prompt=True, date_string=prompt.date_string)


def encode_prompts(tokenizer, prompts: Sequence[str], device: torch.device) -> dict[str, torch.Tensor]:
    """Tokenize rendered prompts. The template already contains BOS, so no special tokens are added.
    Left padding makes position -1 the last prompt token of every row."""
    if tokenizer.padding_side != "left":
        raise ValueError("tokenizer must pad on the left")
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False)
    return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
```

- [ ] **Step 5: Implement `src/halprobe/models.py`**

```python
"""Model and tokenizer loading."""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from halprobe.config import ModelConfig

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
WeightTransform = Callable[[torch.nn.Module], None]


def load_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.name_or_path, revision=cfg.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(cfg: ModelConfig, device: torch.device, weight_transform: WeightTransform | None = None):
    """Load in the configured dtype. `weight_transform` (identity in the MVP) is the hook for Phase 3
    weight-quantization experiments; it runs before the model is moved to `device`."""
    model = AutoModelForCausalLM.from_pretrained(cfg.name_or_path, revision=cfg.revision,
                                                 dtype=DTYPES[cfg.dtype], attn_implementation="sdpa")
    if weight_transform is not None:
        weight_transform(model)
    return model.to(device).eval()


def stop_token_ids(model, tokenizer) -> set[int]:
    ids: set[int] = set()
    eos = model.generation_config.eos_token_id
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_prompts.py tests/test_models.py tests/test_tiny.py -v`
Expected: all pass (11 tests).

- [ ] **Step 7: Commit**

```bash
git add src/halprobe/prompts.py src/halprobe/models.py tests/conftest.py tests/test_prompts.py tests/test_models.py
git commit -m "feat: add few-shot chat prompts with pinned date and model loading" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: OOM back-off batching and greedy generation

**Files:**
- Create: `src/halprobe/batching.py`, `src/halprobe/generation.py`
- Test: `tests/test_batching.py`, `tests/test_generation.py`

**Interfaces:**
- Consumes: `prompts.encode_prompts`, `cache.append_jsonl`, `cache.read_jsonl`, `cache.write_jsonl`, `models.stop_token_ids`
- Produces: `iter_with_backoff(items, batch_size, fn, on_backoff=None) -> Iterator[tuple[Sequence, Any]]`; `postprocess(gen_ids, logprobs, first_entropy, tokenizer, stop_ids) -> dict`; `generate_batch(model, tokenizer, prompts, max_new_tokens, stop_ids) -> list[dict]`; `run_generation(model, tokenizer, items: Sequence[tuple[str, str]], out_path, batch_size, max_new_tokens, stop_ids, log=print, on_batch=None, on_backoff=None) -> None`
- Generation record keys (JSONL): `id, answer, token_ids, token_logprobs, n_answer_tokens, first_token_entropy, finish_reason`

- [ ] **Step 1: Write the failing batching tests**

`tests/test_batching.py`:
```python
import pytest
import torch

from halprobe.batching import iter_with_backoff


def test_backoff_halves_batch_on_oom():
    calls = []

    def fn(batch):
        calls.append(len(batch))
        if len(batch) > 2:
            raise torch.OutOfMemoryError("fake")
        return [x * 2 for x in batch]

    events = []
    out = list(iter_with_backoff(list(range(5)), 8, fn, on_backoff=events.append))
    assert [list(b) for b, _ in out] == [[0, 1], [2, 3], [4]]
    assert [r for _, r in out] == [[0, 2], [4, 6], [8]]
    assert events == [4, 2]
    assert calls == [5, 4, 2, 2, 1]


def test_backoff_reraises_at_batch_size_one():
    def fn(batch):
        raise torch.OutOfMemoryError("fake")

    with pytest.raises(torch.OutOfMemoryError):
        list(iter_with_backoff([1, 2], 2, fn))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_batching.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.batching'`.

- [ ] **Step 3: Implement `src/halprobe/batching.py`**

```python
"""Batch iteration that halves the batch size on GPU out-of-memory errors."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

import torch

T = TypeVar("T")


def iter_with_backoff(items: Sequence[T], batch_size: int, fn: Callable[[Sequence[T]], Any],
                      on_backoff: Callable[[int], None] | None = None) -> Iterator[tuple[Sequence[T], Any]]:
    """Yield (batch, fn(batch)). On OOM, halve the batch size and retry (down to 1, then re-raise)."""
    i, size = 0, batch_size
    while i < len(items):
        batch = items[i : i + size]
        try:
            result = fn(batch)
        except torch.OutOfMemoryError:
            if size == 1:
                raise
            size = max(1, size // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if on_backoff is not None:
                on_backoff(size)
            continue
        yield batch, result
        i += len(batch)
```

- [ ] **Step 4: Run the batching tests**

Run: `uv run pytest tests/test_batching.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write the failing generation tests**

`tests/test_generation.py`:
```python
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
```

- [ ] **Step 6: Run the tests to verify they fail**

Run: `uv run pytest tests/test_generation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.generation'`.

- [ ] **Step 7: Implement `src/halprobe/generation.py`**

```python
"""Batched greedy generation with per-token log-probs and first-token entropy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from transformers import GenerationConfig as HFGenerationConfig

from halprobe.batching import iter_with_backoff
from halprobe.cache import append_jsonl, read_jsonl, write_jsonl
from halprobe.prompts import encode_prompts


def postprocess(gen_ids: list[int], logprobs: list[float], first_entropy: float, tokenizer,
                stop_ids: set[int]) -> dict:
    """Cut the generation at the first stop token or newline-containing token.
    Greedy decoding is causal, so cutting afterwards equals stopping during generation."""
    stop_at = None
    for i, token_id in enumerate(gen_ids):
        if token_id in stop_ids or "\n" in tokenizer.decode([token_id]):
            stop_at = i
            break
    if stop_at is None:
        keep, n_answer, finish = len(gen_ids), len(gen_ids), "length"
    else:
        keep, n_answer, finish = stop_at + 1, stop_at, "stop"
    answer = tokenizer.decode(gen_ids[:keep], skip_special_tokens=True).split("\n")[0].strip()
    return {"answer": answer, "token_ids": gen_ids[:keep], "token_logprobs": logprobs[:keep],
            "n_answer_tokens": n_answer, "first_token_entropy": first_entropy, "finish_reason": finish}


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: Sequence[str], max_new_tokens: int, stop_ids: set[int]) -> list[dict]:
    enc = encode_prompts(tokenizer, prompts, model.device)
    config = HFGenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=sorted(stop_ids),
                                pad_token_id=tokenizer.pad_token_id, output_logits=True, return_dict_in_generate=True)
    out = model.generate(**enc, generation_config=config)
    gen = out.sequences[:, enc["input_ids"].shape[1] :]
    step_logprobs, entropy = [], None
    for t, logits in enumerate(out.logits):
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        step_logprobs.append(log_probs.gather(1, gen[:, t : t + 1]).squeeze(1))
        if t == 0:
            entropy = -(log_probs.exp() * log_probs).nan_to_num().sum(-1)
    logprobs = torch.stack(step_logprobs, dim=1).cpu().tolist()
    entropies = entropy.cpu().tolist()
    ids = gen.cpu().tolist()
    return [postprocess(ids[b], logprobs[b], entropies[b], tokenizer, stop_ids) for b in range(len(prompts))]


def run_generation(model, tokenizer, items: Sequence[tuple[str, str]], out_path: Path, batch_size: int,
                   max_new_tokens: int, stop_ids: set[int], log: Callable[[str], None] = print,
                   on_batch: Callable[[int], None] | None = None,
                   on_backoff: Callable[[int], None] | None = None) -> None:
    """Generate for (id, prompt) items, appending one JSON line per item; resumes from out_path."""
    out_path = Path(out_path)
    done: list[dict] = []
    if out_path.exists():
        done = read_jsonl(out_path, tolerate_partial_tail=True)
        write_jsonl(out_path, done)  # drop a torn last line before appending
    done_ids = {r["id"] for r in done}
    todo = [item for item in items if item[0] not in done_ids]
    log(f"[generate] {len(done_ids)} cached, {len(todo)} to generate")

    def generate(batch):
        return generate_batch(model, tokenizer, [prompt for _, prompt in batch], max_new_tokens, stop_ids)

    for n, (batch, records) in enumerate(iter_with_backoff(todo, batch_size, generate, on_backoff)):
        append_jsonl(out_path, [{"id": item_id, **rec} for (item_id, _), rec in zip(batch, records)])
        if on_batch is not None:
            on_batch(n)
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_batching.py tests/test_generation.py -v`
Expected: 10 passed.

- [ ] **Step 9: Commit**

```bash
git add src/halprobe/batching.py src/halprobe/generation.py tests/test_batching.py tests/test_generation.py
git commit -m "feat: add resumable greedy generation with token log-probs and OOM back-off" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Resumable hidden-state extraction

**Files:**
- Create: `src/halprobe/extraction.py`
- Test: `tests/test_extraction.py`

**Interfaces:**
- Consumes: `prompts.encode_prompts`, `prompts.position_ids_from_mask`, `batching.iter_with_backoff`, `cache.atomic_save_npy`, `cache.atomic_write_json`, `cache.StageDir`
- Produces: `STATES_FILE = "states.f16.mmap"`, `DONE_FILE = "done.npy"`, `ARGMAX_FILE = "argmax.npy"`, `META_FILE = "meta.json"`; `StateStore` (classmethod `open(directory, n_states, ids, d_model, meta)`, `.pending() -> list[int]`, `.write(rows, states, argmax)`, `.close()`, context manager); `last_token_states(model, tokenizer, prompts) -> tuple[np.ndarray (n_states, B, d) float16, np.ndarray (B,) int64]`; `run_extraction(model, tokenizer, ids, prompts, directory, batch_size, meta, log=print, on_batch=None, on_backoff=None)`; `open_states(directory) -> tuple[np.memmap (n_states, N, d), dict]`

- [ ] **Step 1: Write the failing tests**

`tests/test_extraction.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.extraction'`.

- [ ] **Step 3: Implement `src/halprobe/extraction.py`**

```python
"""Hidden states at the last prompt token, stored in a resumable layer-major fp16 memmap."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from halprobe.batching import iter_with_backoff
from halprobe.cache import atomic_save_npy, atomic_write_json
from halprobe.prompts import encode_prompts, position_ids_from_mask

STATES_FILE = "states.f16.mmap"
DONE_FILE = "done.npy"
ARGMAX_FILE = "argmax.npy"
META_FILE = "meta.json"


class StateStore:
    """states.f16.mmap with shape (n_states, N, d_model); done.npy marks finished rows."""

    def __init__(self, directory: Path, arr: np.memmap, done: np.ndarray, argmax: np.ndarray):
        self.directory, self.arr, self.done, self.argmax = directory, arr, done, argmax

    @classmethod
    def open(cls, directory: Path, n_states: int, ids: Sequence[str], d_model: int, meta: dict) -> StateStore:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shape = (n_states, len(ids), d_model)
        full_meta = {**meta, "shape": list(shape), "dtype": "float16", "ids": list(ids)}
        meta_path = directory / META_FILE
        if meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if existing != full_meta:
                raise ValueError(f"{directory}: existing meta.json does not match this run; delete the directory to restart")
            arr = np.memmap(directory / STATES_FILE, dtype=np.float16, mode="r+", shape=shape)
            done = np.load(directory / DONE_FILE)
            argmax = np.load(directory / ARGMAX_FILE)
        else:
            arr = np.memmap(directory / STATES_FILE, dtype=np.float16, mode="w+", shape=shape)
            done = np.zeros(len(ids), dtype=bool)
            argmax = np.full(len(ids), -1, dtype=np.int64)
            atomic_save_npy(directory / DONE_FILE, done)
            atomic_save_npy(directory / ARGMAX_FILE, argmax)
            atomic_write_json(meta_path, full_meta)  # written last: its presence means the files exist
        return cls(directory, arr, done, argmax)

    def pending(self) -> list[int]:
        return np.flatnonzero(~self.done).tolist()

    def write(self, rows: Sequence[int], states: np.ndarray, argmax: np.ndarray) -> None:
        rows = list(rows)
        self.arr[:, rows, :] = states
        self.arr.flush()
        self.argmax[rows] = argmax
        atomic_save_npy(self.directory / ARGMAX_FILE, self.argmax)
        self.done[rows] = True
        atomic_save_npy(self.directory / DONE_FILE, self.done)  # last: rows count as done only after their data

    def close(self) -> None:
        if self.arr is not None:
            self.arr.flush()
            self.arr = None
            gc.collect()  # release the mapping so Windows can rename the directory

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@torch.no_grad()
def last_token_states(model, tokenizer, prompts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    enc = encode_prompts(tokenizer, prompts, model.device)
    out = model(**enc, position_ids=position_ids_from_mask(enc["attention_mask"]),
                output_hidden_states=True, logits_to_keep=1)
    states = torch.stack([h[:, -1, :] for h in out.hidden_states]).to(torch.float16).cpu().numpy()
    argmax = out.logits[:, -1, :].argmax(-1).cpu().numpy().astype(np.int64)
    return states, argmax


def run_extraction(model, tokenizer, ids: Sequence[str], prompts: Sequence[str], directory: Path, batch_size: int,
                   meta: dict, log: Callable[[str], None] = print, on_batch: Callable[[int], None] | None = None,
                   on_backoff: Callable[[int], None] | None = None) -> None:
    n_states = model.config.num_hidden_layers + 1
    with StateStore.open(Path(directory), n_states, ids, model.config.hidden_size, meta) as store:
        pending = store.pending()
        log(f"[extract] {len(ids) - len(pending)} cached, {len(pending)} to extract")

        def extract(rows):
            return last_token_states(model, tokenizer, [prompts[r] for r in rows])

        for n, (rows, (states, argmax)) in enumerate(iter_with_backoff(pending, batch_size, extract, on_backoff)):
            store.write(rows, states, argmax)
            if on_batch is not None:
                on_batch(n)


def open_states(directory: Path) -> tuple[np.memmap, dict]:
    directory = Path(directory)
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    states = np.memmap(directory / STATES_FILE, dtype=np.float16, mode="r", shape=tuple(meta["shape"]))
    return states, meta
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_extraction.py -v`
Expected: 4 passed. If `test_batched_states_match_unbatched` fails, **stop and debug** with superpowers:systematic-debugging. Do not loosen the tolerance: a failure means padding or masking leaks into the probe features.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/extraction.py tests/test_extraction.py
git commit -m "feat: add resumable layer-major memmap extraction of last-prompt-token states" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Evaluation metrics

**Files:**
- Create: `src/halprobe/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `auroc(y, s) -> float`, `auprc(y, s) -> float`, `reliability_bins(y, p, n_bins) -> list[dict]` (keys `lo, hi, count, mean_pred, frac_pos`), `ece(y, p, n_bins=15) -> float`, `risk_coverage(y, s) -> tuple[np.ndarray, np.ndarray]`, `aurc(y, s) -> float`, `risk_at_coverage(y, s, coverage) -> float`, `bootstrap_ci(metric, y, s, n_boot, seed, alpha=0.05) -> tuple[float, float]`, `paired_bootstrap_delta(metric, y, s_a, s_b, n_boot, seed, alpha=0.05) -> dict` (keys `delta, lo, hi`), `summarize(y, score, prob=None, *, n_boot, seed, ece_bins) -> dict` (keys `n, base_rate, auroc, auroc_ci, auprc, auprc_ci, aurc, risk_at_50, risk_at_80`, plus `ece, reliability` when `prob` is given)

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from halprobe.metrics import (
    aurc,
    auprc,
    auroc,
    bootstrap_ci,
    ece,
    paired_bootstrap_delta,
    reliability_bins,
    risk_at_coverage,
    risk_coverage,
    summarize,
)


def test_auroc_auprc_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = y + rng.normal(0, 1, 200)
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert auprc(y, s) == pytest.approx(average_precision_score(y, s))


def test_ece_hand_computed():
    y = [0, 1, 1, 0]
    p = [0.1, 0.9, 0.8, 0.3]
    assert ece(y, p, n_bins=2) == pytest.approx(0.175)
    bins = reliability_bins(y, p, n_bins=2)
    assert [b["count"] for b in bins] == [2, 2]
    assert bins[0]["mean_pred"] == pytest.approx(0.2) and bins[1]["frac_pos"] == pytest.approx(1.0)


def test_reliability_bins_handle_empty_bins_and_p_equal_one():
    bins = reliability_bins([1], [1.0], n_bins=4)
    assert [b["count"] for b in bins] == [0, 0, 0, 1]
    assert bins[0]["mean_pred"] is None


def test_risk_coverage_hand_computed():
    cov, risk = risk_coverage([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
    assert aurc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)


def test_risk_coverage_ties_use_expected_risk_and_ignore_input_order():
    y, s = [0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]
    _, risk = risk_coverage(y, s)
    assert risk.tolist() == pytest.approx([0.0, 0.25, 1 / 3, 0.5])
    _, risk_permuted = risk_coverage([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1])
    assert risk_permuted.tolist() == pytest.approx(risk.tolist())


def test_risk_at_coverage():
    y, s = [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]
    assert risk_at_coverage(y, s, 0.5) == pytest.approx(0.0)
    assert risk_at_coverage(y, s, 0.8) == pytest.approx(0.5)


def test_bootstrap_ci_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1, 300)
    a = bootstrap_ci(auroc, y, s, n_boot=200, seed=0)
    assert a == bootstrap_ci(auroc, y, s, n_boot=200, seed=0)
    assert a[0] < auroc(y, s) < a[1]


def test_paired_delta_zero_for_identical_scores():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 100)
    s = rng.normal(size=100)
    d = paired_bootstrap_delta(auroc, y, s, s, n_boot=50, seed=0)
    assert d["delta"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0


def test_summarize_keys():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 100)
    s = rng.random(100)
    out = summarize(y, s, s, n_boot=20, seed=0, ece_bins=10)
    assert {"n", "base_rate", "auroc", "auroc_ci", "auprc", "auprc_ci", "aurc", "risk_at_50", "risk_at_80",
            "ece", "reliability"} <= set(out)
    assert out["n"] == 100 and len(out["reliability"]) == 10


def test_summarize_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        summarize([0, 1], [0.1, float("nan")], n_boot=5, seed=0, ece_bins=5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.metrics'`.

- [ ] **Step 3: Implement `src/halprobe/metrics.py`**

```python
"""Evaluation metrics. Convention: y = 1 means error; higher score means more likely error."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

Metric = Callable[[np.ndarray, np.ndarray], float]


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s))


def auprc(y, s) -> float:
    return float(average_precision_score(y, s))


def reliability_bins(y, p, n_bins: int) -> list[dict]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else None,
                     "frac_pos": float(y[m].mean()) if m.any() else None})
    return bins


def ece(y, p, n_bins: int = 15) -> float:
    n = len(y)
    return float(sum(b["count"] / n * abs(b["mean_pred"] - b["frac_pos"])
                     for b in reliability_bins(y, p, n_bins) if b["count"]))


def risk_coverage(y, s) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk at coverage k/N (k = 1..N), answering the lowest-score examples first.
    Tied scores are resolved by their expected risk under uniform random tie-breaking."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(y)
    order = np.argsort(s, kind="stable")
    ys, ss = y[order], s[order]
    risk = np.empty(n)
    errors_before, start = 0.0, 0
    while start < n:
        end = start
        while end + 1 < n and ss[end + 1] == ss[start]:
            end += 1
        group = ys[start : end + 1]
        rate = group.mean()
        for k in range(start + 1, end + 2):
            risk[k - 1] = (errors_before + (k - start) * rate) / k
        errors_before += group.sum()
        start = end + 1
    return np.arange(1, n + 1) / n, risk


def aurc(y, s) -> float:
    return float(risk_coverage(y, s)[1].mean())


def risk_at_coverage(y, s, coverage: float) -> float:
    risk = risk_coverage(y, s)[1]
    k = max(1, math.ceil(coverage * len(risk) - 1e-9))
    return float(risk[k - 1])


def bootstrap_ci(metric: Metric, y, s, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI; resamples containing a single class are skipped."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        values.append(metric(y[idx], s[idx]))
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_delta(metric: Metric, y, s_a, s_b, n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """metric(b) - metric(a) with a paired percentile bootstrap CI."""
    rng = np.random.default_rng(seed)
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(metric(y[idx], s_b[idx]) - metric(y[idx], s_a[idx]))
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2]) if deltas else (float("nan"), float("nan"))
    return {"delta": metric(y, s_b) - metric(y, s_a), "lo": float(lo), "hi": float(hi)}


def summarize(y, score, prob=None, *, n_boot: int, seed: int, ece_bins: int) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if not np.all(np.isfinite(score)):
        raise ValueError("scores must be finite; fill NaN scores before evaluation")
    out = {
        "n": int(len(y)), "base_rate": float(y.mean()),
        "auroc": auroc(y, score), "auroc_ci": list(bootstrap_ci(auroc, y, score, n_boot, seed)),
        "auprc": auprc(y, score), "auprc_ci": list(bootstrap_ci(auprc, y, score, n_boot, seed)),
        "aurc": aurc(y, score),
        "risk_at_50": risk_at_coverage(y, score, 0.5), "risk_at_80": risk_at_coverage(y, score, 0.8),
    }
    if prob is not None:
        out["ece"] = ece(y, prob, ece_bins)
        out["reliability"] = reliability_bins(y, prob, ece_bins)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/metrics.py tests/test_metrics.py
git commit -m "feat: add AUROC/AUPRC/ECE/risk-coverage metrics with bootstrap CIs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Per-layer probes

**Files:**
- Create: `src/halprobe/probes.py`
- Test: `tests/test_probes.py`

**Interfaces:**
- Consumes: `metrics.auroc`
- Produces: `LayerFit(layer, C, val_auroc, val_scores, eval_scores, coef, intercept)`, `fit_probe(X_train, y_train, X_val, y_val, X_eval, C_grid, max_iter, seed) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]` returning `(C, val_auroc, val_scores, eval_scores, coef, intercept)`, `sweep_layers(states, train_idx, y_train, val_idx, y_val, eval_idx, C_grid, max_iter, seed, log=print) -> list[LayerFit]`, `select_layer(fits) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_probes.py`:
```python
import numpy as np

from halprobe.probes import LayerFit, select_layer, sweep_layers


def _synthetic(n=300, d=6, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    states = rng.normal(size=(3, n, d)).astype(np.float16)
    states[1, :, 0] += (2.0 * y).astype(np.float16)  # layer 1 is informative
    states[2, :, 0] += (0.5 * y).astype(np.float16)  # layer 2 is weakly informative
    return states, y


def _sweep(states, y):
    idx = np.arange(len(y))
    tr, va, te = idx[:200], idx[200:250], idx[250:]
    return sweep_layers(states, tr, y[tr], va, y[va], te, C_grid=(0.01, 1.0), max_iter=1000, seed=0, log=lambda m: None)


def test_sweep_selects_informative_layer():
    states, y = _synthetic()
    fits = _sweep(states, y)
    assert [f.layer for f in fits] == [0, 1, 2]
    assert select_layer(fits) == 1
    assert fits[1].val_auroc > 0.85
    assert abs(fits[0].val_auroc - 0.5) < 0.25
    assert fits[1].val_scores.shape == (50,) and fits[1].eval_scores.shape == (50,)
    assert fits[1].coef.shape == (6,) and fits[1].C in (0.01, 1.0)
    assert np.all((fits[1].eval_scores >= 0) & (fits[1].eval_scores <= 1))


def test_sweep_is_deterministic():
    states, y = _synthetic()
    a, b = _sweep(states, y), _sweep(states, y)
    assert all(np.array_equal(x.eval_scores, z.eval_scores) for x, z in zip(a, b))


def test_select_layer_prefers_lower_layer_on_tie():
    def fit(layer, auc):
        return LayerFit(layer, 1.0, auc, np.zeros(1), np.zeros(1), np.zeros(1), 0.0)

    assert select_layer([fit(0, 0.6), fit(1, 0.8), fit(2, 0.8)]) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_probes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.probes'`.

- [ ] **Step 3: Implement `src/halprobe/probes.py`**

```python
"""Per-layer logistic-regression probes; C and the reported layer are selected on validation AUROC."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from halprobe.metrics import auroc


@dataclass
class LayerFit:
    layer: int
    C: float
    val_auroc: float
    val_scores: np.ndarray  # P(error) on the labelled validation rows
    eval_scores: np.ndarray  # P(error) on all evaluation rows (e.g. every test row, incl. ambiguous)
    coef: np.ndarray  # in standardized feature space
    intercept: float


def fit_probe(X_train, y_train, X_val, y_val, X_eval, C_grid: Sequence[float], max_iter: int, seed: int):
    best = None
    for C in C_grid:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        score = auroc(y_val, val_scores)
        if best is None or score > best[1]:
            best = (C, score, val_scores, clf)
    C, score, val_scores, clf = best
    lr = clf[-1]
    return C, score, val_scores, clf.predict_proba(X_eval)[:, 1], lr.coef_[0].copy(), float(lr.intercept_[0])


def sweep_layers(states, train_idx, y_train, val_idx, y_val, eval_idx, C_grid: Sequence[float], max_iter: int,
                 seed: int, log: Callable[[str], None] = print) -> list[LayerFit]:
    """states: array-like (n_states, N, d), e.g. the read-only extraction memmap."""
    fits = []
    for layer in range(states.shape[0]):
        view = states[layer]

        def rows(idx, view=view):
            return np.asarray(view[np.asarray(idx)], dtype=np.float32)

        C, val_auc, val_scores, eval_scores, coef, intercept = fit_probe(
            rows(train_idx), y_train, rows(val_idx), y_val, rows(eval_idx), C_grid, max_iter, seed)
        fits.append(LayerFit(layer, C, val_auc, val_scores, eval_scores, coef, intercept))
        log(f"[probe] layer {layer:2d}: C={C:g} val AUROC={val_auc:.3f}")
    return fits


def select_layer(fits: Sequence[LayerFit]) -> int:
    """Best validation AUROC; ties go to the lower layer."""
    return max(fits, key=lambda f: (f.val_auroc, -f.layer)).layer
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_probes.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/probes.py tests/test_probes.py
git commit -m "feat: add per-layer logistic probes with validation-selected C and layer" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Baselines

**Files:**
- Create: `src/halprobe/baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: generation records (Task 9 keys)
- Produces: `neg_mean_logprob(records) -> np.ndarray` (NaN for empty answers), `first_token_entropy(records) -> np.ndarray`, `fill_nan_as_max(scores) -> np.ndarray`, `popularity_features(s_pop) -> np.ndarray (N, 1)`, `metadata_features(s_pop, relations, vocabulary) -> np.ndarray (N, 1 + R)`, `fit_logreg(X_train, y_train, X_eval_list, seed) -> list[np.ndarray]`, `calibrate(score_train, y_train, score_eval, seed) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

`tests/test_baselines.py`:
```python
import numpy as np
import pytest

from halprobe.baselines import (
    calibrate,
    fill_nan_as_max,
    first_token_entropy,
    fit_logreg,
    metadata_features,
    neg_mean_logprob,
    popularity_features,
)
from halprobe.metrics import auroc


def test_neg_mean_logprob_excludes_stop_token():
    recs = [{"token_logprobs": [-1.0, -3.0, -0.5], "n_answer_tokens": 2}]
    assert neg_mean_logprob(recs).tolist() == [2.0]


def test_neg_mean_logprob_nan_for_empty_answer():
    assert np.isnan(neg_mean_logprob([{"token_logprobs": [-0.1], "n_answer_tokens": 0}])[0])


def test_fill_nan_as_max():
    assert fill_nan_as_max([1.0, np.nan, 3.0]).tolist() == [1.0, 4.0, 3.0]
    assert fill_nan_as_max([np.nan]).tolist() == [0.0]


def test_first_token_entropy():
    assert first_token_entropy([{"first_token_entropy": 1.5}, {"first_token_entropy": 0.2}]).tolist() == [1.5, 0.2]


def test_metadata_features_one_hot_and_unseen_relation():
    X = metadata_features([10, 1000, 0], ["genre", "capital", "sport"], ["capital", "genre"])
    assert X.tolist() == [[1.0, 0.0, 1.0], [3.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    assert popularity_features([100]).tolist() == [[2.0]]


def test_fit_logreg_learns_popularity_shortcut():
    rng = np.random.default_rng(0)
    s_pop = 10 ** rng.uniform(0, 6, 400)
    y = (np.log10(s_pop) + rng.normal(0, 0.5, 400) < 3).astype(int)  # rare entities -> errors
    X = popularity_features(s_pop)
    (p,) = fit_logreg(X[:300], y[:300], [X[300:]], seed=0)
    assert auroc(y[300:], p) > 0.9


def test_calibrate_returns_monotone_probabilities():
    rng = np.random.default_rng(1)
    score = rng.normal(size=300)
    y = (score + rng.normal(0, 1, 300) > 0).astype(int)
    p = calibrate(score[:200], y[:200], score[200:], seed=0)
    assert np.all((p > 0) & (p < 1))
    order = np.argsort(score[200:])
    assert np.all(np.diff(p[order]) >= -1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.baselines'`.

- [ ] **Step 3: Implement `src/halprobe/baselines.py`**

```python
"""Uncertainty and shortcut baselines. All scores: higher = more likely an error."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def neg_mean_logprob(records: Sequence[dict]) -> np.ndarray:
    """-mean log-prob over answer tokens (stop token excluded); NaN when the answer is empty."""
    out = []
    for r in records:
        n = r["n_answer_tokens"]
        out.append(-float(np.mean(r["token_logprobs"][:n])) if n > 0 else np.nan)
    return np.array(out, dtype=float)


def first_token_entropy(records: Sequence[dict]) -> np.ndarray:
    return np.array([r["first_token_entropy"] for r in records], dtype=float)


def fill_nan_as_max(scores) -> np.ndarray:
    """Treat undefined scores (empty answers) as maximally uncertain."""
    s = np.array(scores, dtype=float)
    finite = s[np.isfinite(s)]
    s[~np.isfinite(s)] = finite.max() + 1.0 if finite.size else 0.0
    return s


def popularity_features(s_pop) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(s_pop, dtype=float), 1.0))[:, None]


def metadata_features(s_pop, relations: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    index = {r: i for i, r in enumerate(vocabulary)}
    onehot = np.zeros((len(relations), len(vocabulary)))
    for row, rel in enumerate(relations):
        if rel in index:
            onehot[row, index[rel]] = 1.0
    return np.hstack([popularity_features(s_pop), onehot])


def fit_logreg(X_train, y_train, X_eval_list: Sequence[np.ndarray], seed: int) -> list[np.ndarray]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=seed))
    clf.fit(X_train, y_train)
    return [clf.predict_proba(X)[:, 1] for X in X_eval_list]


def calibrate(score_train, y_train, score_eval, seed: int) -> np.ndarray:
    """Platt-style 1-D logistic calibration of a raw score into P(error)."""
    return fit_logreg(np.asarray(score_train)[:, None], y_train, [np.asarray(score_eval)[:, None]], seed)[0]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/baselines.py tests/test_baselines.py
git commit -m "feat: add log-prob, entropy, popularity and metadata baselines" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Shortcut and sensitivity analyses

**Files:**
- Create: `src/halprobe/analysis.py`
- Test: `tests/test_analysis.py`

**Interfaces:**
- Consumes: `metrics.auroc`, `metrics.paired_bootstrap_delta`, `baselines.fit_logreg`
- Produces: `within_relation_auroc(y, s, relations, min_per_class) -> dict` (keys `macro` (float or None), `per_relation`, `n_relations`), `popularity_quintile_auroc(y, s, s_pop, n_bins=5) -> list[dict]`, `incremental_value(meta_val, probe_val, y_val, meta_test, probe_test, y_test, *, n_boot, seed) -> dict` (keys `auroc_metadata, auroc_combined, delta, delta_ci`), `label_sensitivity(labels, scores) -> dict` (keys `n_ambiguous, ambiguous_as_correct, ambiguous_as_incorrect`)

- [ ] **Step 1: Write the failing tests**

`tests/test_analysis.py`:
```python
import numpy as np
import pytest

from halprobe.analysis import incremental_value, label_sensitivity, popularity_quintile_auroc, within_relation_auroc
from halprobe.metrics import auroc


def test_within_relation_exposes_relation_shortcut():
    rng = np.random.default_rng(0)
    relations = np.array(["capital"] * 100 + ["director"] * 100)
    y = np.concatenate([rng.random(100) < 0.2, rng.random(100) < 0.8]).astype(int)
    s = (relations == "director").astype(float) + rng.normal(0, 0.01, 200)  # knows only the relation
    assert auroc(y, s) > 0.75
    res = within_relation_auroc(y, s, relations, min_per_class=5)
    assert res["n_relations"] == 2
    assert abs(res["macro"] - 0.5) < 0.15


def test_within_relation_skips_small_relations():
    y = np.array([0, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.2, 0.3])
    rel = np.array(["a", "a", "b", "b"])
    res = within_relation_auroc(y, s, rel, min_per_class=1)
    assert list(res["per_relation"]) == ["a"]
    none = within_relation_auroc(y, s, rel, min_per_class=5)
    assert none["macro"] is None and none["n_relations"] == 0


def test_popularity_quintiles_cover_all_examples():
    rng = np.random.default_rng(1)
    s_pop = rng.integers(1, 10**6, 500)
    y = rng.integers(0, 2, 500)
    bins = popularity_quintile_auroc(y, rng.random(500), s_pop)
    assert len(bins) == 5 and sum(b["n"] for b in bins) == 500
    assert all(b["auroc"] is None or 0 <= b["auroc"] <= 1 for b in bins)


def test_incremental_value_positive_when_probe_adds_information():
    rng = np.random.default_rng(2)
    n = 1200
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (a + b + rng.normal(0, 0.5, n) > 0).astype(int)
    meta = 1 / (1 + np.exp(-a))
    probe = 1 / (1 + np.exp(-b))
    res = incremental_value(meta[:600], probe[:600], y[:600], meta[600:], probe[600:], y[600:], n_boot=100, seed=0)
    assert res["auroc_combined"] > res["auroc_metadata"]
    assert res["delta"] > 0 and res["delta_ci"][0] > 0


def test_label_sensitivity_hand_computed():
    labels = ["correct", "incorrect", "ambiguous", "ambiguous"]
    res = label_sensitivity(labels, {"m": np.array([0.1, 0.9, 0.95, 0.05])})
    assert res["n_ambiguous"] == 2
    assert res["ambiguous_as_correct"]["m"] == pytest.approx(2 / 3)
    assert res["ambiguous_as_incorrect"]["m"] == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_analysis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.analysis'`.

- [ ] **Step 3: Implement `src/halprobe/analysis.py`**

```python
"""Shortcut analyses (relation, popularity, incremental value) and label-sensitivity analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from halprobe.baselines import fit_logreg
from halprobe.metrics import auroc, paired_bootstrap_delta


def within_relation_auroc(y, s, relations, min_per_class: int) -> dict:
    """AUROC inside each relation with at least `min_per_class` examples of both classes; macro mean."""
    y, s, relations = np.asarray(y), np.asarray(s), np.asarray(relations)
    per = {}
    for rel in sorted(set(relations.tolist())):
        m = relations == rel
        n_pos = int(y[m].sum())
        if n_pos >= min_per_class and int(m.sum()) - n_pos >= min_per_class:
            per[rel] = auroc(y[m], s[m])
    return {"macro": float(np.mean(list(per.values()))) if per else None, "per_relation": per, "n_relations": len(per)}


def popularity_quintile_auroc(y, s, s_pop, n_bins: int = 5) -> list[dict]:
    y, s, s_pop = np.asarray(y), np.asarray(s), np.asarray(s_pop, dtype=float)
    edges = np.quantile(s_pop, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges[1:-1], s_pop, side="right"), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        two_classes = m.any() and 0 < int(y[m].sum()) < int(m.sum())
        out.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                    "base_rate": float(y[m].mean()) if m.any() else None,
                    "auroc": auroc(y[m], s[m]) if two_classes else None})
    return out


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def incremental_value(meta_val, probe_val, y_val, meta_test, probe_test, y_test, *, n_boot: int, seed: int) -> dict:
    """Does the probe add information beyond metadata? The 2-feature combiner is fit on validation,
    where both scores are out-of-sample (train-set probe scores are in-sample and over-confident)."""
    X_val = np.column_stack([_logit(meta_val), _logit(probe_val)])
    X_test = np.column_stack([_logit(meta_test), _logit(probe_test)])
    (combined,) = fit_logreg(X_val, y_val, [X_test], seed)
    d = paired_bootstrap_delta(auroc, y_test, meta_test, combined, n_boot, seed)
    return {"auroc_metadata": auroc(y_test, meta_test), "auroc_combined": auroc(y_test, combined),
            "delta": d["delta"], "delta_ci": [d["lo"], d["hi"]]}


def label_sensitivity(labels: Sequence[str], scores: Mapping[str, np.ndarray]) -> dict:
    labels = np.asarray(labels)
    out: dict = {"n_ambiguous": int((labels == "ambiguous").sum())}
    for name, ambiguous_y in (("ambiguous_as_correct", 0), ("ambiguous_as_incorrect", 1)):
        y = np.where(labels == "incorrect", 1, np.where(labels == "correct", 0, ambiguous_y))
        out[name] = {method: auroc(y, s) for method, s in scores.items()}
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_analysis.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/analysis.py tests/test_analysis.py
git commit -m "feat: add within-relation, popularity, incremental-value and label-sensitivity analyses" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Figures

Before writing chart code, invoke the `dataviz` skill and apply its guidance: palette check, mark specs, legend placement. The code below is the baseline. Adjust only the styling, never the data passed in.

**Files:**
- Create: `src/halprobe/plotting.py`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Produces: `PALETTE`, `plot_layer_sweep(layers, val_auroc, test_auroc, test_ci, baselines, selected_layer, path)`, `plot_reliability(bins_by_method, path)`, `plot_risk_coverage(curves, base_rate, path)`, `plot_relation_auroc(per_relation, path)`

- [ ] **Step 1: Write the failing tests**

`tests/test_plotting.py`:
```python
import numpy as np

from halprobe.metrics import reliability_bins, risk_coverage
from halprobe.plotting import plot_layer_sweep, plot_relation_auroc, plot_reliability, plot_risk_coverage


def _png(path):
    return path.exists() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_all_figures_render(tmp_path):
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 100), rng.random(100)
    plot_layer_sweep([0, 1, 2], [0.5, 0.7, 0.65], [0.52, 0.68, 0.66], [(0.45, 0.6), (0.6, 0.75), (0.6, 0.72)],
                     {"metadata": 0.62, "neg_mean_logprob": 0.7}, 1, tmp_path / "sweep.png")
    plot_reliability({"probe": reliability_bins(y, p, 10)}, tmp_path / "rel.png")
    plot_risk_coverage({"probe": risk_coverage(y, p)}, float(y.mean()), tmp_path / "rc.png")
    plot_relation_auroc({"probe": {"capital": 0.7, "genre": 0.6}, "metadata": {"capital": 0.5}}, tmp_path / "relations.png")
    assert all(_png(tmp_path / f) for f in ("sweep.png", "rel.png", "rc.png", "relations.png"))


def test_relation_plot_with_no_relations(tmp_path):
    plot_relation_auroc({"probe": {}}, tmp_path / "empty.png")
    assert _png(tmp_path / "empty.png")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.plotting'`.

- [ ] **Step 3: Implement `src/halprobe/plotting.py`**

```python
"""Run-report figures (headless matplotlib)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito: colour-blind-safe categorical palette.
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")
NEUTRAL = "#888888"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layer_sweep(layers: Sequence[int], val_auroc: Sequence[float], test_auroc: Sequence[float],
                     test_ci: Sequence[Sequence[float]], baselines: Mapping[str, float], selected_layer: int,
                     path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lo = np.array([c[0] for c in test_ci], dtype=float)
    hi = np.array([c[1] for c in test_ci], dtype=float)
    ax.fill_between(layers, lo, hi, color=PALETTE[0], alpha=0.15, linewidth=0, label="probe test 95% CI")
    ax.plot(layers, test_auroc, color=PALETTE[0], marker="o", markersize=3, label="probe (test)")
    ax.plot(layers, val_auroc, color=PALETTE[0], linestyle="--", linewidth=1, label="probe (val)")
    for i, (name, value) in enumerate(baselines.items(), start=1):
        ax.axhline(value, color=PALETTE[i % len(PALETTE)], linestyle=":", linewidth=1.4, label=name)
    ax.axhline(0.5, color="#cccccc", linewidth=0.8)
    finite = [v for v in [*val_auroc, *test_auroc, *lo, *baselines.values()] if np.isfinite(v)]
    ax.set_ylim(min(0.45, min(finite) - 0.02), 1.0)
    ax.axvline(selected_layer, color=NEUTRAL, linewidth=0.8)
    ax.text(selected_layer, 0.99, " val-selected", va="top", fontsize=8, color="#555555")
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("AUROC (error = positive)")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style(ax)
    _save(fig, path)


def plot_reliability(bins_by_method: Mapping[str, Sequence[dict]], path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=0.8, linestyle="--", label="perfect calibration")
    for i, (name, bins) in enumerate(bins_by_method.items()):
        points = [(b["mean_pred"], b["frac_pos"]) for b in bins if b["count"]]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, marker="o", markersize=3, color=PALETTE[i % len(PALETTE)], label=name)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted P(error)", ylabel="observed error rate")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    _save(fig, path)


def plot_risk_coverage(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], base_rate: float,
                       path: Path | str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (name, (coverage, risk)) in enumerate(curves.items()):
        ax.plot(coverage, risk, color=PALETTE[i % len(PALETTE)], linewidth=1.4, label=name)
    ax.axhline(base_rate, color=NEUTRAL, linewidth=0.8, linestyle="--", label="no abstention")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="coverage (fraction answered)", ylabel="error rate among answered")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    _save(fig, path)


def plot_relation_auroc(per_relation: Mapping[str, Mapping[str, float]], path: Path | str) -> None:
    relations = sorted({r for d in per_relation.values() for r in d})
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.3 * len(relations) * max(1, len(per_relation)))))
    if not relations:
        ax.text(0.5, 0.5, "no relation has enough test examples of both classes", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return
    height = 0.8 / len(per_relation)
    ys = np.arange(len(relations))
    for i, (name, values) in enumerate(per_relation.items()):
        ax.barh(ys + i * height, [values.get(r, np.nan) for r in relations], height=height,
                color=PALETTE[i % len(PALETTE)], label=name)
    ax.set_yticks(ys + height * (len(per_relation) - 1) / 2, relations)
    ax.axvline(0.5, color=NEUTRAL, linewidth=0.8)
    ax.set(xlim=(0, 1), xlabel="AUROC within relation (test)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/plotting.py tests/test_plotting.py
git commit -m "feat: add layer-sweep, reliability, risk-coverage and per-relation figures" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: Report assembly and rendering

**Files:**
- Create: `src/halprobe/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `analysis.*`, `metrics.auroc/bootstrap_ci/risk_coverage/summarize`, `plotting.*`, `data.split_views`, `cache.atomic_write_json/atomic_write_text`, `config.to_plain`, `device.resolve_device`
- Produces: `METHOD_TIMING: dict[str, str]`, `BASELINE_NAMES`, `check_consistency(examples, gens_by_id, argmax, threshold) -> dict`, `environment_info(cfg) -> dict`, `format_summary_table(metrics) -> str`, `format_shortcut_table(metrics) -> str`, `format_sensitivity_table(metrics) -> str`, `replace_block(text, name, content) -> str`, `build_report(cfg, examples, labels_by_id, probe_dir, baselines_dir, consistency, data_report, label_report, keys, run_dir) -> dict`, `render_results(run_dir, readme, results_md) -> None`
- Consumed stage files: `probe_dir/probe_report.json` (`layers: [{layer, C, val_auroc}]`, `selected_layer`), `probe_dir/val_scores.npy` (n_states × len(val.idx_lab)), `probe_dir/test_scores.npy` (n_states × len(test.idx_all)), `baselines_dir/scores.npz` (keys `score__<name>`, `prob__<name>` over all N examples)

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
import json

import numpy as np
import pytest

from halprobe.data import Example
from halprobe.report import (
    check_consistency,
    format_sensitivity_table,
    format_shortcut_table,
    format_summary_table,
    render_results,
    replace_block,
)


def _metrics(label_mode="alias"):
    method = {"timing": "pre-generation", "n": 100, "base_rate": 0.7, "auroc": 0.75, "auroc_ci": [0.7, 0.8],
              "auprc": 0.85, "auprc_ci": [0.8, 0.9], "ece": 0.05, "aurc": 0.5, "risk_at_50": 0.4, "risk_at_80": 0.6}
    return {
        "run_id": "20261012-120000", "label_mode": label_mode, "selected_layer": 14,
        "methods": {"probe": method, "metadata": {**method, "auroc": 0.66}},
        "shortcut": {"within_relation": {"probe": {"macro": 0.7, "n_relations": 12},
                                         "metadata": {"macro": None, "n_relations": 0}},
                     "incremental": {"auroc_metadata": 0.66, "auroc_combined": 0.77, "delta": 0.11,
                                     "delta_ci": [0.08, 0.14]}},
        "sensitivity": {"n_ambiguous": 30, "ambiguous_as_correct": {"probe": 0.73, "metadata": 0.65},
                        "ambiguous_as_incorrect": {"probe": 0.74, "metadata": 0.64}},
    }


def test_check_consistency_pass_and_fail():
    examples = [Example(str(i), "q", f"s{i}", "r", ("a",), 1, "test") for i in range(4)]
    gens = {str(i): {"token_ids": [i + 10, 0]} for i in range(4)}
    ok = check_consistency(examples, gens, np.array([10, 11, 12, 99]), threshold=0.75)
    assert ok["match_rate"] == 0.75 and ok["mismatched_ids"] == ["3"]
    with pytest.raises(RuntimeError, match="consistency"):
        check_consistency(examples, gens, np.array([10, 11, 98, 99]), threshold=0.75)


def test_format_tables():
    m = _metrics()
    summary = format_summary_table(m)
    assert "| probe | pre-generation | 0.750 [0.700, 0.800] | 0.850 [0.800, 0.900] | 0.050 | 0.500 | 0.600 |" in summary
    assert "layer 14" in summary and "n = 100" in summary
    assert "| metadata | n/a | 0 |" in format_shortcut_table(m)
    assert "0.110 [0.080, 0.140]" in format_shortcut_table(m)
    assert "| probe | 0.750 | 0.730 | 0.740 |" in format_sensitivity_table(m)


def test_replace_block():
    text = "a\n<!-- results:start -->\nold\n<!-- results:end -->\nb\n"
    assert replace_block(text, "results", "new") == "a\n<!-- results:start -->\nnew\n<!-- results:end -->\nb\n"
    with pytest.raises(ValueError, match="other"):
        replace_block(text, "other", "x")


def _write_run(tmp_path, metrics):
    run_dir = tmp_path / "results" / "exp" / metrics["run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- results:start -->\n<!-- results:end -->\n", encoding="utf-8")
    results_md = tmp_path / "results.md"
    results_md.write_text("".join(f"<!-- {n}:start -->\n<!-- {n}:end -->\n"
                                  for n in ("run", "summary", "shortcut", "sensitivity")), encoding="utf-8")
    return run_dir, readme, results_md


def test_render_results_writes_blocks(tmp_path):
    run_dir, readme, results_md = _write_run(tmp_path, _metrics())
    render_results(run_dir, readme, results_md)
    assert "| probe | pre-generation | 0.750" in readme.read_text(encoding="utf-8")
    text = results_md.read_text(encoding="utf-8")
    assert "20261012-120000" in text and "0.110 [0.080, 0.140]" in text


def test_render_results_refuses_pseudo_labels(tmp_path):
    run_dir, readme, results_md = _write_run(tmp_path, _metrics(label_mode="pseudo"))
    with pytest.raises(ValueError, match="pseudo"):
        render_results(run_dir, readme, results_md)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.report'`.

- [ ] **Step 3: Implement `src/halprobe/report.py`**

```python
"""Run report: evaluation of all methods, shortcut/sensitivity analyses, figures, tables, provenance."""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from halprobe.analysis import incremental_value, label_sensitivity, popularity_quintile_auroc, within_relation_auroc
from halprobe.cache import atomic_write_json, atomic_write_text
from halprobe.config import Config, to_plain
from halprobe.data import Example, split_views
from halprobe.metrics import auroc, bootstrap_ci, risk_coverage, summarize
from halprobe.plotting import plot_layer_sweep, plot_relation_auroc, plot_reliability, plot_risk_coverage

METHOD_TIMING = {
    "probe": "pre-generation",
    "probe_layer0": "pre-generation",
    "first_token_entropy": "pre-generation",
    "popularity": "pre-generation",
    "metadata": "pre-generation",
    "neg_mean_logprob": "post-generation",
}
BASELINE_NAMES = ("first_token_entropy", "popularity", "metadata", "neg_mean_logprob")


def check_consistency(examples: Sequence[Example], gens_by_id: dict, argmax: np.ndarray, threshold: float) -> dict:
    """The extraction pass's argmax at the last prompt token must equal the first generated token."""
    first = np.array([gens_by_id[ex.id]["token_ids"][0] for ex in examples])
    match = first == np.asarray(argmax)
    rate = float(match.mean())
    result = {"match_rate": rate, "n": int(len(match)),
              "mismatched_ids": [ex.id for ex, ok in zip(examples, match) if not ok][:50]}
    if rate < threshold:
        raise RuntimeError(f"consistency check failed: argmax matches first generated token for {rate:.4f} "
                           f"< {threshold} of examples; padding/position bug? first mismatches: {result['mismatched_ids'][:5]}")
    return result


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info(cfg: Config) -> dict:
    import sklearn
    import torch
    import transformers

    from halprobe.device import resolve_device

    try:
        device = resolve_device(cfg.device, cfg.gpu_name_hint)
        device_name = "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device)
    except (RuntimeError, ValueError):
        device_name = None
    status = _git("status", "--porcelain")
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "hip": torch.version.hip, "cuda": torch.version.cuda, "device": device_name,
            "transformers": transformers.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def _fmt(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_ci(value, ci) -> str:
    return f"{_fmt(value)} [{_fmt(ci[0])}, {_fmt(ci[1])}]"


def format_summary_table(metrics: dict) -> str:
    lines = ["| Method | Timing | AUROC [95% CI] | AUPRC [95% CI] | ECE | AURC | Risk@80% |",
             "|---|---|---|---|---|---|---|"]
    for name, m in metrics["methods"].items():
        lines.append(f"| {name} | {m['timing']} | {_fmt_ci(m['auroc'], m['auroc_ci'])} | "
                     f"{_fmt_ci(m['auprc'], m['auprc_ci'])} | {_fmt(m.get('ece'))} | {_fmt(m['aurc'])} | "
                     f"{_fmt(m['risk_at_80'])} |")
    first = next(iter(metrics["methods"].values()))
    lines += ["", f"Test set: n = {first['n']}, error base rate = {_fmt(first['base_rate'])}. "
                  f"Probe uses layer {metrics['selected_layer']} (selected on validation AUROC)."]
    return "\n".join(lines)


def format_shortcut_table(metrics: dict) -> str:
    shortcut = metrics["shortcut"]
    lines = ["| Method | Within-relation AUROC (macro) | Relations used |", "|---|---|---|"]
    for name, w in shortcut["within_relation"].items():
        lines.append(f"| {name} | {_fmt(w['macro'])} | {w['n_relations']} |")
    inc = shortcut["incremental"]
    lines += ["", f"Metadata alone: AUROC {_fmt(inc['auroc_metadata'])}; metadata + probe: "
                  f"{_fmt(inc['auroc_combined'])}; ΔAUROC {_fmt_ci(inc['delta'], inc['delta_ci'])}."]
    return "\n".join(lines)


def format_sensitivity_table(metrics: dict) -> str:
    sens = metrics["sensitivity"]
    lines = [f"Ambiguous test examples: {sens['n_ambiguous']}.", "",
             "| Method | Headline AUROC | Ambiguous = correct | Ambiguous = incorrect |", "|---|---|---|---|"]
    for name, m in metrics["methods"].items():
        lines.append(f"| {name} | {_fmt(m['auroc'])} | {_fmt(sens['ambiguous_as_correct'][name])} | "
                     f"{_fmt(sens['ambiguous_as_incorrect'][name])} |")
    return "\n".join(lines)


def replace_block(text: str, name: str, content: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        raise ValueError(f"markers for block {name!r} not found")
    return text[: i + len(start)] + "\n" + content.strip() + "\n" + text[j:]


def build_report(cfg: Config, examples: Sequence[Example], labels_by_id: dict, probe_dir: Path, baselines_dir: Path,
                 consistency: dict, data_report: dict, label_report: dict, keys: dict, run_dir: Path) -> dict:
    ev, seed = cfg.evaluation, cfg.seed
    views = split_views(examples, labels_by_id)
    test, val = views["test"], views["val"]
    y = test.y
    probe_report = json.loads((Path(probe_dir) / "probe_report.json").read_text(encoding="utf-8"))
    selected = probe_report["selected_layer"]
    test_scores = np.load(Path(probe_dir) / "test_scores.npy")
    val_scores = np.load(Path(probe_dir) / "val_scores.npy")
    baseline_scores = np.load(Path(baselines_dir) / "scores.npz")

    score_all = {"probe": test_scores[selected], "probe_layer0": test_scores[0]}  # over test.idx_all
    prob_all = dict(score_all)
    for name in BASELINE_NAMES:
        score_all[name] = baseline_scores[f"score__{name}"][test.idx_all]
        prob_all[name] = baseline_scores[f"prob__{name}"][test.idx_all]
    labelled = {name: score_all[name][test.pos_lab] for name in METHOD_TIMING}

    methods = {name: {"timing": timing, **summarize(y, labelled[name], prob_all[name][test.pos_lab],
                                                    n_boot=ev.n_bootstrap, seed=seed, ece_bins=ev.ece_bins)}
               for name, timing in METHOD_TIMING.items()}
    sweep = []
    for info in probe_report["layers"]:
        s = test_scores[info["layer"]][test.pos_lab]
        sweep.append({**info, "test_auroc": auroc(y, s),
                      "test_auroc_ci": list(bootstrap_ci(auroc, y, s, ev.n_bootstrap, seed))})

    relations = np.array([examples[i].relation for i in test.idx_lab])
    s_pop = np.array([examples[i].s_pop for i in test.idx_lab])
    shortcut = {
        "within_relation": {n: within_relation_auroc(y, s, relations, ev.min_per_class_relation)
                            for n, s in labelled.items()},
        "popularity_quintiles": {n: popularity_quintile_auroc(y, s, s_pop) for n, s in labelled.items()},
        "incremental": incremental_value(baseline_scores["score__metadata"][val.idx_lab], val_scores[selected], val.y,
                                         labelled["metadata"], labelled["probe"], y, n_boot=ev.n_bootstrap, seed=seed),
    }
    test_labels = [labels_by_id[examples[i].id]["label"] for i in test.idx_all]
    metrics = {
        "experiment": cfg.experiment, "run_id": Path(run_dir).name, "label_mode": cfg.labeling.mode,
        "data": data_report, "labels": label_report, "consistency": consistency, "selected_layer": selected,
        "layer_sweep": sweep, "methods": methods, "shortcut": shortcut,
        "sensitivity": label_sensitivity(test_labels, score_all),
    }

    figures = Path(run_dir) / "figures"
    plot_layer_sweep([r["layer"] for r in sweep], [r["val_auroc"] for r in sweep], [r["test_auroc"] for r in sweep],
                     [r["test_auroc_ci"] for r in sweep], {n: methods[n]["auroc"] for n in BASELINE_NAMES},
                     selected, figures / "layer_sweep.png")
    plot_reliability({n: m["reliability"] for n, m in methods.items()}, figures / "reliability.png")
    plot_risk_coverage({n: risk_coverage(y, s) for n, s in labelled.items()}, float(y.mean()),
                       figures / "risk_coverage.png")
    plot_relation_auroc({n: shortcut["within_relation"][n]["per_relation"]
                         for n in ("probe", "metadata", "first_token_entropy")}, figures / "relation_auroc.png")

    atomic_write_text(Path(run_dir) / "tables" / "summary.md", "\n\n".join(
        [format_summary_table(metrics), format_shortcut_table(metrics), format_sensitivity_table(metrics)]) + "\n")
    atomic_write_json(Path(run_dir) / "metrics.json", metrics)
    atomic_write_text(Path(run_dir) / "config.yaml", yaml.safe_dump(to_plain(cfg), sort_keys=False, allow_unicode=True))
    atomic_write_json(Path(run_dir) / "env.json", environment_info(cfg))
    atomic_write_json(Path(run_dir) / "stage_keys.json", keys)
    return metrics


def render_results(run_dir: Path, readme: Path, results_md: Path) -> None:
    """Write result tables from run_dir/metrics.json into README.md and docs/results.md marker blocks."""
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("label_mode") != "alias":
        raise ValueError(f"{run_dir}: label_mode={metrics.get('label_mode')!r} (pseudo labels); "
                         "only alias-labelled runs may be reported")
    rel = run_dir.as_posix()
    head = f"Run `{metrics['run_id']}` · [metrics]({rel}/metrics.json) · [figures]({rel}/figures/)\n\n"
    readme = Path(readme)
    readme.write_text(replace_block(readme.read_text(encoding="utf-8"), "results", head + format_summary_table(metrics)),
                      encoding="utf-8")
    results_md = Path(results_md)
    text = results_md.read_text(encoding="utf-8")
    blocks = {
        "run": f"All numbers below come from run `{metrics['run_id']}` ([metrics.json](../{rel}/metrics.json)).",
        "summary": format_summary_table(metrics),
        "shortcut": format_shortcut_table(metrics),
        "sensitivity": format_sensitivity_table(metrics),
    }
    for name, content in blocks.items():
        text = replace_block(text, name, content)
    results_md.write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_report.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/halprobe/report.py tests/test_report.py
git commit -m "feat: add run report assembly, consistency check and results rendering" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 17: Pipeline orchestration, CLI, tiny config, end-to-end test

**Files:**
- Create: `src/halprobe/pipeline.py`, `src/halprobe/__main__.py`, `configs/tiny_cpu.yaml`
- Modify: `tests/conftest.py` (add the `tiny_config` fixture)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `STAGES = ("data", "generate", "label", "extract", "probe", "baselines", "report")`, `stage_keys(cfg) -> dict[str, str]`, `stage_dirs(cfg, artifacts_root) -> dict[str, StageDir]`, `RunResult(run_dir: Path | None, cache_hits: dict[str, bool])`, `run(cfg, artifacts_root=Path("artifacts"), results_root=Path("results"), until="report", log=print) -> RunResult`, and the CLI `python -m halprobe {run,data,generate,label,extract,probe,baselines} CONFIG [--artifacts DIR] [--results DIR]` plus `python -m halprobe make-tiny DIR`.

- [ ] **Step 1: Write the tiny config**

`configs/tiny_cpu.yaml`:
```yaml
# CPU smoke configuration: tiny random Llama + synthetic PopQA-format data.
# Labels are PSEUDO labels (hash of the example id), because a random model answers nothing correctly.
# Results from this config are never reported. Create the assets first:
#   uv run python -m halprobe make-tiny artifacts/tiny
experiment: tiny_cpu
seed: 0
device: cpu
data:
  local_path: artifacts/tiny/popqa_tiny.tsv
  split_ratios: [0.7, 0.15, 0.15]
model:
  name_or_path: artifacts/tiny/model
  dtype: float32
prompt:
  system: "Answer the question with a short phrase (a name, place, date, or category). Do not explain."
  date_string: "26 Jul 2024"
  shots:
    - {subject: "Ada Lovelace", question: "What is Ada Lovelace's occupation?", answer: "mathematician"}
    - {subject: "Spirited Away", question: "Who was the director of Spirited Away?", answer: "Hayao Miyazaki"}
generation: {max_new_tokens: 8, batch_size: 16}
labeling: {mode: pseudo}
extraction: {batch_size: 16, consistency_threshold: 0.9}
probe: {C_grid: [0.01, 1.0], max_iter: 2000, min_class_count: 3}
evaluation: {n_bootstrap: 50, ece_bins: 10, min_per_class_relation: 2}
```

- [ ] **Step 2: Add the `tiny_config` fixture to `tests/conftest.py`**

Append to `tests/conftest.py`:
```python
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
```

- [ ] **Step 3: Write the failing tests**

`tests/test_pipeline.py`:
```python
import json
from dataclasses import replace

import yaml

from halprobe.__main__ import main
from halprobe.config import to_plain
from halprobe.data import load_examples
from halprobe.extraction import open_states
from halprobe.pipeline import STAGES, run, stage_dirs, stage_keys


def _quiet(_):
    pass


def test_stage_keys_cascade_only_where_relevant(tiny_config):
    base = stage_keys(tiny_config)
    probe_only = stage_keys(replace(tiny_config, probe=replace(tiny_config.probe, C_grid=(0.1,))))
    assert {k for k in base if base[k] != probe_only[k]} == {"probe"}
    batch = stage_keys(replace(tiny_config, generation=replace(tiny_config.generation, batch_size=4)))
    assert batch == base
    prompt = stage_keys(replace(tiny_config, prompt=replace(tiny_config.prompt, system="Other.")))
    assert {k for k in base if base[k] != prompt[k]} == {"generate", "label", "extract", "probe", "baselines"}
    seed = stage_keys(replace(tiny_config, seed=1))
    assert all(seed[k] != base[k] for k in base)


def test_end_to_end_tiny(tmp_path, tiny_config):
    artifacts, results = tmp_path / "artifacts", tmp_path / "results"
    first = run(tiny_config, artifacts, results, log=_quiet)
    assert first.cache_hits == {s: False for s in STAGES if s != "report"}
    metrics = json.loads((first.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["label_mode"] == "pseudo"
    assert set(metrics["methods"]) == {"probe", "probe_layer0", "first_token_entropy", "popularity", "metadata",
                                       "neg_mean_logprob"}
    assert all(0 <= m["auroc"] <= 1 for m in metrics["methods"].values())
    assert len(metrics["layer_sweep"]) == 3
    assert metrics["consistency"]["match_rate"] >= 0.9
    for name in ("layer_sweep.png", "reliability.png", "risk_coverage.png", "relation_auroc.png"):
        assert (first.run_dir / "figures" / name).stat().st_size > 0
    for name in ("config.yaml", "env.json", "stage_keys.json", "tables/summary.md"):
        assert (first.run_dir / name).exists()
    assert "| probe |" in (first.run_dir / "tables" / "summary.md").read_text(encoding="utf-8")

    dirs = stage_dirs(tiny_config, artifacts)
    n_examples = len(load_examples(dirs["data"].final / "examples.jsonl"))
    states, _ = open_states(dirs["extract"].final)
    assert states.shape == (3, n_examples, 64)
    del states

    second = run(tiny_config, artifacts, results, log=_quiet)
    assert all(second.cache_hits.values())
    assert second.run_dir != first.run_dir


def test_cli_runs_single_stage(tmp_path, tiny_config):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(to_plain(tiny_config)), encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    assert main(["data", str(cfg_path), "--artifacts", str(artifacts), "--results", str(tmp_path / "r")]) == 0
    dirs = stage_dirs(tiny_config, artifacts)
    assert dirs["data"].is_complete() and not dirs["generate"].is_complete()


def test_cli_make_tiny(tmp_path):
    assert main(["make-tiny", str(tmp_path / "tiny")]) == 0
    assert (tmp_path / "tiny" / "popqa_tiny.tsv").exists() and (tmp_path / "tiny" / "model").is_dir()
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.__main__'`.

- [ ] **Step 5: Implement `src/halprobe/pipeline.py`**

```python
"""Stage orchestration with content-addressed caching and lazy model loading."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from halprobe import baselines as bl
from halprobe.cache import StageDir, atomic_write_json, read_jsonl, stage_key, write_jsonl
from halprobe.config import Config, to_plain
from halprobe.data import (
    SPLITS,
    Example,
    SplitView,
    build_examples,
    fetch_popqa,
    load_examples,
    load_popqa_rows,
    save_examples,
    split_views,
)
from halprobe.device import resolve_device, seed_everything
from halprobe.extraction import ARGMAX_FILE, open_states, run_extraction
from halprobe.generation import run_generation
from halprobe.labeling import RULES_VERSION, label_answer, pseudo_label
from halprobe.models import load_model, load_tokenizer, stop_token_ids
from halprobe.probes import select_layer, sweep_layers
from halprobe.prompts import render_prompt
from halprobe.report import build_report, check_consistency

STAGES = ("data", "generate", "label", "extract", "probe", "baselines", "report")
# Bump a stage's version when its code changes in a way that alters its outputs.
STAGE_VERSIONS = {"data": 1, "generate": 1, "label": RULES_VERSION, "extract": 1, "probe": 1, "baselines": 1}
POSITION = "last_prompt"
Log = Callable[[str], None]


def stage_keys(cfg: Config) -> dict[str, str]:
    plain = to_plain(cfg)
    model_prompt = {"model": plain["model"], "prompt": plain["prompt"]}
    v = STAGE_VERSIONS
    keys = {"data": stage_key({"stage": "data", "v": v["data"], "data": plain["data"], "seed": cfg.seed,
                               "shot_subjects": [s.subject for s in cfg.prompt.shots]})}
    keys["generate"] = stage_key({"stage": "generate", "v": v["generate"], "up": keys["data"], **model_prompt,
                                  "max_new_tokens": cfg.generation.max_new_tokens})
    keys["label"] = stage_key({"stage": "label", "v": v["label"], "up": keys["generate"], "labeling": plain["labeling"]})
    keys["extract"] = stage_key({"stage": "extract", "v": v["extract"], "up": keys["data"], **model_prompt,
                                 "position": POSITION})
    keys["probe"] = stage_key({"stage": "probe", "v": v["probe"], "up": [keys["extract"], keys["label"]],
                               "probe": plain["probe"], "seed": cfg.seed})
    keys["baselines"] = stage_key({"stage": "baselines", "v": v["baselines"], "up": [keys["generate"], keys["label"]],
                                   "seed": cfg.seed})
    return keys


def stage_dirs(cfg: Config, artifacts_root: Path | str) -> dict[str, StageDir]:
    return {name: StageDir(artifacts_root, name, key) for name, key in stage_keys(cfg).items()}


class _ModelHandle:
    """Loads the tokenizer/model only when a GPU stage actually has work to do."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_tokenizer(self.cfg.model)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.cfg.model, resolve_device(self.cfg.device, self.cfg.gpu_name_hint))
        return self._model


def _run_stage(sd: StageDir, build: Callable[[Path], None], log: Log) -> tuple[Path, bool]:
    if sd.is_complete():
        log(f"[{sd.stage}] cache hit {sd.key}")
        return sd.final, True
    log(f"[{sd.stage}] building {sd.key}")
    build(sd.begin())
    return sd.commit(), False


def _build_data(cfg: Config, out: Path) -> None:
    d = cfg.data
    path = Path(d.local_path) if d.local_path else fetch_popqa(d.repo_id, d.filename, d.revision)
    examples, report = build_examples(load_popqa_rows(path), d.split_ratios, cfg.seed,
                                      [s.subject for s in cfg.prompt.shots])
    save_examples(out / "examples.jsonl", examples)
    atomic_write_json(out / "data_report.json", report)


def _build_generate(cfg: Config, handle: _ModelHandle, examples: Sequence[Example], out: Path, log: Log) -> None:
    tok, model = handle.tokenizer, handle.model
    items = [(ex.id, render_prompt(tok, ex.question, cfg.prompt)) for ex in examples]
    backoffs: list[int] = []
    run_generation(model, tok, items, out / "generations.jsonl", cfg.generation.batch_size,
                   cfg.generation.max_new_tokens, stop_token_ids(model, tok), log=log, on_backoff=backoffs.append)
    atomic_write_json(out / "meta.json", {"batch_size": cfg.generation.batch_size, "oom_backoffs": backoffs,
                                          "device": str(model.device), "n": len(items)})


def _build_label(cfg: Config, examples: Sequence[Example], gens_by_id: dict, out: Path) -> None:
    rows = []
    for ex in examples:
        gen = gens_by_id[ex.id]
        result = (pseudo_label(ex.id) if cfg.labeling.mode == "pseudo"
                  else label_answer(gen["answer"], ex.answers, gen["finish_reason"]))
        rows.append({"id": ex.id, "label": result.label, "reason": result.reason, "y": result.y})
    write_jsonl(out / "labels.jsonl", rows)
    split_of = {ex.id: ex.split for ex in examples}
    counts = Counter(r["label"] for r in rows)
    decided = counts["correct"] + counts["incorrect"]
    atomic_write_json(out / "label_report.json", {
        "mode": cfg.labeling.mode, "counts": dict(counts),
        "ambiguous_reasons": dict(Counter(r["reason"] for r in rows if r["reason"])),
        "accuracy_unambiguous": counts["correct"] / decided if decided else None,
        "by_split": {s: dict(Counter(r["label"] for r in rows if split_of[r["id"]] == s)) for s in SPLITS},
    })


def _build_extract(cfg: Config, handle: _ModelHandle, examples: Sequence[Example], out: Path, log: Log) -> None:
    tok, model = handle.tokenizer, handle.model
    prompts = [render_prompt(tok, ex.question, cfg.prompt) for ex in examples]
    meta = {"model": cfg.model.name_or_path, "revision": cfg.model.revision, "dtype": cfg.model.dtype,
            "position": POSITION}
    run_extraction(model, tok, [ex.id for ex in examples], prompts, out, cfg.extraction.batch_size, meta, log=log)


def _check_class_counts(views: dict[str, SplitView], minimum: int) -> None:
    for split, view in views.items():
        n_pos = int(view.y.sum())
        n_neg = len(view.y) - n_pos
        if min(n_pos, n_neg) < minimum:
            raise RuntimeError(f"{split}: {n_pos} errors / {n_neg} correct labelled examples (< {minimum} per class)")


def _build_probe(cfg: Config, views: dict[str, SplitView], extract_dir: Path, out: Path, log: Log) -> None:
    _check_class_counts(views, cfg.probe.min_class_count)
    states, _ = open_states(extract_dir)
    train, val, test = views["train"], views["val"], views["test"]
    fits = sweep_layers(states, train.idx_lab, train.y, val.idx_lab, val.y, test.idx_all,
                        cfg.probe.C_grid, cfg.probe.max_iter, cfg.seed, log)
    del states
    np.save(out / "val_scores.npy", np.stack([f.val_scores for f in fits]))
    np.save(out / "test_scores.npy", np.stack([f.eval_scores for f in fits]))
    np.save(out / "coefs.npy", np.stack([f.coef for f in fits]))
    np.save(out / "intercepts.npy", np.array([f.intercept for f in fits]))
    atomic_write_json(out / "probe_report.json", {
        "layers": [{"layer": f.layer, "C": f.C, "val_auroc": f.val_auroc} for f in fits],
        "selected_layer": select_layer(fits),
    })


def _build_baselines(cfg: Config, examples: Sequence[Example], views: dict[str, SplitView], gens_by_id: dict,
                     out: Path) -> None:
    records = [gens_by_id[ex.id] for ex in examples]
    train = views["train"]
    raw = {"neg_mean_logprob": bl.fill_nan_as_max(bl.neg_mean_logprob(records)),
           "first_token_entropy": bl.first_token_entropy(records)}
    s_pop = np.array([ex.s_pop for ex in examples])
    relations = [ex.relation for ex in examples]
    vocabulary = sorted({relations[i] for i in train.idx_lab})
    arrays = {}
    for name, score in raw.items():
        arrays[f"score__{name}"] = score
        arrays[f"prob__{name}"] = bl.calibrate(score[train.idx_lab], train.y, score, cfg.seed)
    for name, X in (("popularity", bl.popularity_features(s_pop)),
                    ("metadata", bl.metadata_features(s_pop, relations, vocabulary))):
        (p,) = bl.fit_logreg(X[train.idx_lab], train.y, [X], cfg.seed)
        arrays[f"score__{name}"] = p
        arrays[f"prob__{name}"] = p
    np.savez(out / "scores.npz", **arrays)
    atomic_write_json(out / "baselines_report.json", {"relation_vocabulary": vocabulary})


def _new_run_dir(parent: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate, n = parent / stamp, 1
    while candidate.exists():
        candidate, n = parent / f"{stamp}-{n}", n + 1
    candidate.mkdir(parents=True)
    return candidate


@dataclass
class RunResult:
    run_dir: Path | None
    cache_hits: dict[str, bool]


def run(cfg: Config, artifacts_root: Path | str = Path("artifacts"), results_root: Path | str = Path("results"),
        until: str = "report", log: Log = print) -> RunResult:
    if until not in STAGES:
        raise ValueError(f"unknown stage {until!r}; expected one of {STAGES}")
    seed_everything(cfg.seed)
    dirs = stage_dirs(cfg, artifacts_root)
    handle = _ModelHandle(cfg)
    hits: dict[str, bool] = {}

    def stage(name: str, build: Callable[[Path], None]) -> Path:
        path, hits[name] = _run_stage(dirs[name], build, log)
        return path

    def stop_after(name: str) -> bool:
        return STAGES.index(name) >= STAGES.index(until)

    data_dir = stage("data", lambda out: _build_data(cfg, out))
    examples = load_examples(data_dir / "examples.jsonl")
    if stop_after("data"):
        return RunResult(None, hits)
    gen_dir = stage("generate", lambda out: _build_generate(cfg, handle, examples, out, log))
    gens_by_id = {r["id"]: r for r in read_jsonl(gen_dir / "generations.jsonl")}
    if stop_after("generate"):
        return RunResult(None, hits)
    label_dir = stage("label", lambda out: _build_label(cfg, examples, gens_by_id, out))
    labels_by_id = {r["id"]: r for r in read_jsonl(label_dir / "labels.jsonl")}
    if stop_after("label"):
        return RunResult(None, hits)
    extract_dir = stage("extract", lambda out: _build_extract(cfg, handle, examples, out, log))
    consistency = check_consistency(examples, gens_by_id, np.load(extract_dir / ARGMAX_FILE),
                                    cfg.extraction.consistency_threshold)
    log(f"[consistency] argmax == first generated token for {consistency['match_rate']:.4f} of examples")
    if stop_after("extract"):
        return RunResult(None, hits)
    views = split_views(examples, labels_by_id)
    probe_dir = stage("probe", lambda out: _build_probe(cfg, views, extract_dir, out, log))
    if stop_after("probe"):
        return RunResult(None, hits)
    baselines_dir = stage("baselines", lambda out: _build_baselines(cfg, examples, views, gens_by_id, out))
    if stop_after("baselines"):
        return RunResult(None, hits)
    run_dir = _new_run_dir(Path(results_root) / cfg.experiment)
    build_report(cfg, examples, labels_by_id, probe_dir, baselines_dir, consistency,
                 json.loads((data_dir / "data_report.json").read_text(encoding="utf-8")),
                 json.loads((label_dir / "label_report.json").read_text(encoding="utf-8")),
                 {name: sd.key for name, sd in dirs.items()}, run_dir)
    log(f"[report] wrote {run_dir}")
    return RunResult(run_dir, hits)
```

- [ ] **Step 6: Implement `src/halprobe/__main__.py`**

```python
"""CLI: python -m halprobe {run,<stage>} CONFIG | python -m halprobe make-tiny DIR"""

from __future__ import annotations

import argparse
from pathlib import Path

from halprobe.config import load_config
from halprobe.pipeline import STAGES, run
from halprobe.tiny import write_tiny_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="halprobe", description="Linear-probe hallucination detection pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", *STAGES[:-1]):
        help_text = "run all stages" if name == "run" else f"run stages up to and including '{name}'"
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", type=Path)
        p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
        p.add_argument("--results", type=Path, default=Path("results"))
    tiny = sub.add_parser("make-tiny", help="write tiny offline assets for configs/tiny_cpu.yaml")
    tiny.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "make-tiny":
        assets = write_tiny_assets(args.out_dir)
        print(f"wrote {assets.tsv} and {assets.model_dir}")
        return 0
    cfg = load_config(args.config)
    result = run(cfg, args.artifacts, args.results, until="report" if args.command == "run" else args.command)
    if result.run_dir is not None:
        print(f"results written to {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run the pipeline tests and the full suite**

Run:
```bash
uv run pytest tests/test_pipeline.py -v
uv run pytest -q --durations=5
```
Expected: 4 passed in the pipeline tests, and the whole suite passes. Note the total runtime (target: about 60 s). If the suite is much slower, report which tests dominate, and do not cut assertions.

- [ ] **Step 8: Smoke-run the CLI by hand**

Run:
```bash
uv run python -m halprobe make-tiny artifacts/tiny
uv run python -m halprobe run configs/tiny_cpu.yaml
uv run python -m halprobe run configs/tiny_cpu.yaml
```
Expected: the first run builds every stage and prints `results written to results/tiny_cpu/<timestamp>`. The second run prints `cache hit` for all six stages. `results/tiny_cpu/` is gitignored.

- [ ] **Step 9: Commit**

```bash
git add src/halprobe/pipeline.py src/halprobe/__main__.py configs/tiny_cpu.yaml tests/conftest.py tests/test_pipeline.py
git commit -m "feat: add cached stage pipeline, CLI and tiny end-to-end test" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 18: Manual review tools and results rendering script

**Files:**
- Create: `src/halprobe/review.py`, `scripts/export_review_sample.py`, `scripts/score_review.py`, `scripts/render_results.py`
- Test: `tests/test_review.py`

**Interfaces:**
- Consumes: `pipeline.stage_dirs`, `data.load_examples`, `cache.read_jsonl`, `cache.atomic_write_json`, `report.render_results`
- Produces: `REVIEW_COLUMNS`, `MANUAL_LABELS`, `review_rows(examples, gens_by_id, labels_by_id) -> list[dict]`, `sample_for_review(rows, n, seed) -> list[dict]`, `write_review_csv(path, rows)`, `score_review(path) -> dict` (keys `n_rows, n_judged, n_manual_ambiguous, n_disagree, noise_rate, confusion, disagreeing_ids`)

- [ ] **Step 1: Write the failing tests**

`tests/test_review.py`:
```python
import csv

import pytest

from halprobe.data import Example
from halprobe.review import REVIEW_COLUMNS, review_rows, sample_for_review, score_review, write_review_csv


def _rows(n=30):
    examples = [Example(str(i), f"Q{i}?", f"S{i}", "genre", ("rock", "rock music"), 10, "test") for i in range(n)]
    gens = {str(i): {"answer": f"answer {i}"} for i in range(n)}
    labels = {str(i): {"label": ["correct", "incorrect", "ambiguous"][i % 3], "reason": "partial" if i % 3 == 2 else None}
              for i in range(n)}
    return review_rows(examples, gens, labels)


def test_review_rows_content():
    row = _rows()[2]
    assert row["gold_answers"] == "rock | rock music"
    assert row["auto_label"] == "ambiguous" and row["reason"] == "partial" and row["manual_label"] == ""
    assert list(row) == list(REVIEW_COLUMNS)


def test_sample_excludes_ambiguous_and_is_deterministic():
    rows = _rows()
    sample = sample_for_review(rows, 5, seed=0)
    assert len(sample) == 5 and all(r["auto_label"] != "ambiguous" for r in sample)
    assert sample == sample_for_review(rows, 5, seed=0)
    assert len(sample_for_review(rows, 1000, seed=0)) == 20


def _fill(path, labels, delimiter=","):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row, label in zip(rows, labels):
        row["manual_label"] = label
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_score_review_noise_rate(tmp_path):
    path = tmp_path / "review.csv"
    rows = [r for r in _rows(6) if r["auto_label"] != "ambiguous"]  # correct, incorrect, correct, incorrect
    write_review_csv(path, rows)
    _fill(path, ["correct", "correct", "ambiguous", "incorrect"])
    result = score_review(path)
    assert result["n_rows"] == 4 and result["n_judged"] == 3 and result["n_manual_ambiguous"] == 1
    assert result["n_disagree"] == 1 and result["noise_rate"] == pytest.approx(1 / 3)
    assert result["disagreeing_ids"] == ["1"]


def test_score_review_accepts_semicolon_csv_from_german_excel(tmp_path):
    path = tmp_path / "review.csv"
    write_review_csv(path, [r for r in _rows(3) if r["auto_label"] != "ambiguous"])
    _fill(path, ["Correct", "incorrect"], delimiter=";")
    assert score_review(path)["n_disagree"] == 0


def test_score_review_rejects_blank_and_invalid(tmp_path):
    path = tmp_path / "review.csv"
    write_review_csv(path, [r for r in _rows(3) if r["auto_label"] != "ambiguous"])
    with pytest.raises(ValueError, match="no manual_label"):
        score_review(path)
    _fill(path, ["correct", "maybe"])
    with pytest.raises(ValueError, match="maybe"):
        score_review(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_review.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'halprobe.review'`.

- [ ] **Step 3: Implement `src/halprobe/review.py`**

```python
"""Manual label review: export random samples, score filled-in sheets."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from halprobe.data import Example

REVIEW_COLUMNS = ("id", "split", "relation", "question", "answer", "gold_answers", "auto_label", "reason",
                  "manual_label")
MANUAL_LABELS = ("correct", "incorrect", "ambiguous")


def review_rows(examples: Sequence[Example], gens_by_id: dict, labels_by_id: dict) -> list[dict]:
    return [{"id": ex.id, "split": ex.split, "relation": ex.relation, "question": ex.question,
             "answer": gens_by_id[ex.id]["answer"], "gold_answers": " | ".join(ex.answers),
             "auto_label": labels_by_id[ex.id]["label"], "reason": labels_by_id[ex.id]["reason"] or "",
             "manual_label": ""} for ex in examples]


def sample_for_review(rows: Sequence[dict], n: int, seed: int) -> list[dict]:
    """n seeded random rows among the unambiguous automatic labels (estimates their noise rate)."""
    pool = [r for r in rows if r["auto_label"] != "ambiguous"]
    if not pool:
        return []
    idx = np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


def write_review_csv(path: Path | str, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel shows umlauts correctly
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def score_review(path: Path | str) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        f.seek(0)
        delimiter = ";" if header.count(";") > header.count(",") else ","  # German Excel saves with ';'
        rows = list(csv.DictReader(f, delimiter=delimiter))
    manual = [r["manual_label"].strip().lower() for r in rows]
    blanks = [r["id"] for r, m in zip(rows, manual) if not m]
    if blanks:
        raise ValueError(f"{len(blanks)} row(s) have no manual_label, e.g. {blanks[:5]}")
    invalid = sorted({m for m in manual if m not in MANUAL_LABELS})
    if invalid:
        raise ValueError(f"invalid manual_label value(s) {invalid}; use one of {MANUAL_LABELS}")
    judged = [(r, m) for r, m in zip(rows, manual) if m != "ambiguous"]
    disagree = [r["id"] for r, m in judged if m != r["auto_label"]]
    return {"n_rows": len(rows), "n_judged": len(judged), "n_manual_ambiguous": len(rows) - len(judged),
            "n_disagree": len(disagree), "noise_rate": len(disagree) / len(judged) if judged else None,
            "confusion": dict(Counter(f"{r['auto_label']}->{m}" for r, m in zip(rows, manual))),
            "disagreeing_ids": disagree}
```

- [ ] **Step 4: Write the three scripts**

`scripts/export_review_sample.py`:
```python
"""Export labelled examples for manual review.

Usage: uv run python scripts/export_review_sample.py configs/mvp_popqa_llama3b.yaml [--n 100] [--seed 0]
Writes results/review/<experiment>/review_sample.csv (n random unambiguous examples) and ambiguous.csv.
Fill the manual_label column with correct / incorrect / ambiguous, then run scripts/score_review.py.
"""

import argparse
from pathlib import Path

from halprobe.cache import read_jsonl
from halprobe.config import load_config
from halprobe.data import load_examples
from halprobe.pipeline import stage_dirs
from halprobe.review import review_rows, sample_for_review, write_review_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export a manual review sample.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("results/review"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite an existing (possibly filled-in) sample")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dirs = stage_dirs(cfg, args.artifacts)
    for name in ("data", "generate", "label"):
        if not dirs[name].is_complete():
            raise SystemExit(f"stage {name!r} has not run for this config; run: python -m halprobe label {args.config}")
    out = args.out / cfg.experiment
    sample_path = out / "review_sample.csv"
    if sample_path.exists() and not args.force:
        raise SystemExit(f"{sample_path} exists (it may already be filled in); pass --force to overwrite")
    examples = load_examples(dirs["data"].final / "examples.jsonl")
    gens = {r["id"]: r for r in read_jsonl(dirs["generate"].final / "generations.jsonl")}
    labels = {r["id"]: r for r in read_jsonl(dirs["label"].final / "labels.jsonl")}
    rows = review_rows(examples, gens, labels)
    sample = sample_for_review(rows, args.n, args.seed)
    ambiguous = [r for r in rows if r["auto_label"] == "ambiguous"]
    write_review_csv(sample_path, sample)
    write_review_csv(out / "ambiguous.csv", ambiguous)
    print(f"wrote {len(sample)} rows to {sample_path} and {len(ambiguous)} rows to {out / 'ambiguous.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`scripts/score_review.py`:
```python
"""Score a filled-in review CSV (label-noise rate of the automatic labels).

Usage: uv run python scripts/score_review.py results/review/mvp_popqa_llama3b/review_sample.csv
Writes <name>_score.json next to the CSV.
"""

import argparse
import json
from pathlib import Path

from halprobe.cache import atomic_write_json
from halprobe.review import score_review


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a filled-in manual review CSV.")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args(argv)
    result = score_review(args.csv)
    out = args.csv.with_name(args.csv.stem + "_score.json")
    atomic_write_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "disagreeing_ids"}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`scripts/render_results.py`:
```python
"""Render result tables from a run's metrics.json into README.md and docs/results.md.

Usage: uv run python scripts/render_results.py results/mvp_popqa_llama3b/<run_id>
Refuses runs with pseudo labels.
"""

import argparse
from pathlib import Path

from halprobe.report import render_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render README/docs result tables from a run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--results-md", type=Path, default=Path("docs/results.md"))
    args = parser.parse_args(argv)
    render_results(args.run_dir, args.readme, args.results_md)
    print(f"rendered {args.run_dir} into {args.readme} and {args.results_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_review.py -v && uv run pytest -q`
Expected: 5 passed in the review tests, then the whole suite passes.

- [ ] **Step 6: Commit**

```bash
git add src/halprobe/review.py scripts/export_review_sample.py scripts/score_review.py scripts/render_results.py tests/test_review.py
git commit -m "feat: add manual label review export/scoring and results rendering scripts" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 19: Real MVP run on the RX 6700 XT

This task produces the only numbers the project may report. Do not edit any result file by hand. If something looks wrong, stop and report it.

**Files:**
- Generated (committed): `results/mvp_popqa_llama3b/<run_id>/**`, `results/review/mvp_popqa_llama3b/*.csv`, `results/review/mvp_popqa_llama3b/review_sample_score.json`

- [ ] **Step 1: Pre-flight**

Run:
```bash
uv run pytest -q
uv run python scripts/env_check.py
```
Expected: all tests pass, and `ENV CHECK: PASS` (Task 2 already covered Hugging Face access).

- [ ] **Step 2: Data stage**

Run: `uv run python -m halprobe data configs/mvp_popqa_llama3b.yaml`
Then inspect `artifacts/data/<key>/data_report.json`. Expected: `rows_in` 14267, `groups_dropped_multi_entity` 561, and `examples_out` = `question_groups` − 561 (the brainstorming analysis predicts 13066 − 561 = 12505). Report the actual numbers. If the dropped count is not 561, stop and investigate before going on.

- [ ] **Step 3: Generation and labeling**

Run in the background, because the expected wall time is tens of minutes:
```bash
uv run python -m halprobe label configs/mvp_popqa_llama3b.yaml
```
When it finishes, inspect `artifacts/label/<key>/label_report.json` and 20 random generations. Show the user `counts`, `ambiguous_reasons` and `accuracy_unambiguous`. **Decision point:** if ambiguous examples exceed 25% of the total, or if `accuracy_unambiguous` is below 0.05 or above 0.95, stop and discuss with the user before going on. Do not change the labeling rules or the prompt on your own.

- [ ] **Step 4: Extraction, probes, baselines, report**

Run in the background:
```bash
uv run python -m halprobe run configs/mvp_popqa_llama3b.yaml
```
Expected: data, generate and label are cache hits; then extraction (minutes) and the consistency log line `argmax == first generated token for 0.99xx`; then a probe sweep over 29 layers (this can take tens of minutes on CPU); then `results written to results/mvp_popqa_llama3b/<run_id>`. If the consistency check raises, stop and debug with superpowers:systematic-debugging.

- [ ] **Step 5: Verify caching**

Run: `uv run python -m halprobe run configs/mvp_popqa_llama3b.yaml`
Expected: `cache hit` for all six stages and a second run directory with identical `metrics.json` numbers. Then delete that duplicate run directory, after checking it differs from the first only in `run_id`. Compare with `diff <(jq 'del(.run_id)' A/metrics.json) <(jq 'del(.run_id)' B/metrics.json)`, or with Python if `jq` is missing.

- [ ] **Step 6: Manual review sample**

Run: `uv run python scripts/export_review_sample.py configs/mvp_popqa_llama3b.yaml`
Then ask the **user** to fill in `manual_label` in `results/review/mvp_popqa_llama3b/review_sample.csv`, with values correct / incorrect / ambiguous, judging whether the model answer is factually right given the gold aliases and world knowledge. When they are done, run:
`uv run python scripts/score_review.py results/review/mvp_popqa_llama3b/review_sample.csv`
and report the noise rate to the user.

- [ ] **Step 7: Commit the results**

```bash
git add results/mvp_popqa_llama3b results/review/mvp_popqa_llama3b
git commit -m "results: add MVP run on Llama-3.2-3B-Instruct and manual label review" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 20: README and results write-up

**Files:**
- Modify: `README.md` (full rewrite)
- Create: `docs/results.md`

Requirement: every number in these files either comes from `scripts/render_results.py` or is copied from the committed `metrics.json` / `review_sample_score.json` of the Task 19 run, next to the run id. No estimates, no rounding beyond 3 decimals, no numbers from memory.

- [ ] **Step 1: Write `README.md` with these sections, in this order**

1. **Title and one-paragraph motivation:** can a linear probe read *before generation* whether a small model is about to answer a factual question wrongly, and is that more than popularity/relation shortcuts?
2. **Key result:** the block `<!-- results:start -->` / `<!-- results:end -->` (filled by the script), followed by the layer-sweep figure, embedded from the committed run directory: `![layer sweep](results/mvp_popqa_llama3b/<run_id>/figures/layer_sweep.png)`.
3. **Setup (Windows, AMD RX 6700 XT / RDNA2):**
   - why TheRock nightly (official AMD Windows wheels exclude gfx103x; staging index URL);
   - the exact pinned wheel;
   - `pip install --user uv`;
   - `setx UV_PROJECT_ENVIRONMENT C:\venvs\halprobe` (the long-path limit; enabling `LongPathsEnabled` is the alternative);
   - `uv sync`;
   - `uv run hf auth login` and accepting the Llama license;
   - `uv run python scripts/env_check.py`;
   - the iGPU note (device picked by name).
4. **Setup (Linux, untested alternative):**
   - ROCm 6.x/7.x PyTorch with `HSA_OVERRIDE_GFX_VERSION=10.3.0`;
   - an explicit statement that this path was not tested for this project;
   - that the lockfile pins the CPU torch for Linux, so a ROCm user must swap the torch source.
5. **Reproduce:**
   - `uv run python -m halprobe run configs/mvp_popqa_llama3b.yaml`, plus the stage subcommands;
   - where outputs go (`artifacts/` cache, `results/` reports);
   - the manual review commands;
   - `scripts/render_results.py`.
6. **Tests:**
   - `uv run pytest` (CPU, offline, tiny model);
   - `uv run pytest -m gpu`;
   - the tiny CPU smoke run (`make-tiny` + `configs/tiny_cpu.yaml`, with pseudo labels and never reported).
7. **Method in brief:**
   - entity-grouped splits;
   - alias labeling with ambiguity exclusion;
   - per-layer logistic probes with C and layer chosen on validation;
   - the baselines and why the metadata baseline is the key control.
8. **Project layout:** the module list from this plan's File Structure.
9. **Related work:** Kadavath et al. 2022; Burns et al. 2022; Azaria & Mitchell 2023; Marks & Tegmark 2023; Kossen et al. 2024. Write one sentence each on how this project relates (pre-generation, small model, explicit shortcut control).
10. **Limitations:** a link to `docs/results.md#limitations`.

- [ ] **Step 2: Write `docs/results.md` with marker blocks and prose**

Required skeleton (the prose sections are written from the rendered numbers):
```markdown
# Results: MVP (PopQA, Llama-3.2-3B-Instruct, last prompt token)

<!-- run:start -->
<!-- run:end -->

## Setup recap
(data counts from metrics.json["data"], label counts and accuracy from metrics.json["labels"],
consistency match rate, manual-review noise rate from review_sample_score.json)

## Headline comparison
<!-- summary:start -->
<!-- summary:end -->
(2–4 sentences: does the probe beat each baseline? Cite CIs; say "not distinguishable" when CIs overlap.)

## Layer sweep
(embed figures/layer_sweep.png; describe where the signal emerges and how the val-selected layer compares)

## Shortcut controls
<!-- shortcut:start -->
<!-- shortcut:end -->
(interpret within-relation AUROC and ΔAUROC over metadata; embed relation_auroc.png. If the probe does
not add over metadata, say so plainly — this is a valid negative result.)

## Calibration and selective answering
(embed reliability.png and risk_coverage.png; compare AURC / risk@80%)

## Label sensitivity
<!-- sensitivity:start -->
<!-- sensitivity:end -->

## Limitations
- Label noise: alias matching; measured noise rate on 100 manual reviews; ambiguous exclusion.
- Shortcut learning: popularity/relation confounds and what the controls can and cannot rule out.
- Model size and scope: one 3B model, one dataset, one prompt, closed-book QA only.
- Platform: AMD nightly torch build; fp16 inference.
- Any anomalies observed in Task 19 (OOM back-offs, consistency mismatches, decision-point discussions).
```

- [ ] **Step 3: Render the tables**

Run: `uv run python scripts/render_results.py results/mvp_popqa_llama3b/<run_id>`
Expected: `rendered ... into README.md and docs/results.md`. Then read both files and check that every number in the prose matches the rendered tables and `metrics.json`.

- [ ] **Step 4: Final check and commit**

Run: `uv run pytest -q`
Expected: all pass.

```bash
git add README.md docs/results.md
git commit -m "docs: add README with setup/reproduction and MVP results write-up" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
