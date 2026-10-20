# halprobe: linear probes for pre-generation hallucination detection

Can a linear probe read, **before the model generates a single answer token**, whether a small instruct LLM is about to answer a factual question wrongly? And is that signal more than a shortcut for "obscure entity" or "hard relation type"? This repository trains per-layer logistic-regression probes on the hidden state of **Llama-3.2-3B-Instruct** at the last prompt token, on closed-book PopQA questions. It compares them with standard uncertainty baselines and with an explicit popularity + relation-type control. Splits are entity-disjoint, model selection uses validation data only, and every reported number comes with a bootstrap confidence interval and traces back to a committed run.

## Key result

<!-- results:start -->
Run `20261013-204454` · [metrics](results/mvp_popqa_llama3b/20261013-204454/metrics.json) · [figures](results/mvp_popqa_llama3b/20261013-204454/figures/)

| Method | Timing | AUROC [95% CI] | AUPRC [95% CI] | ECE | AURC | Risk@80% |
|---|---|---|---|---|---|---|
| probe | pre-generation | 0.897 [0.878, 0.915] | 0.951 [0.939, 0.961] | 0.021 | 0.417 | 0.638 |
| probe_layer0 | pre-generation | 0.500 [0.500, 0.500] | 0.709 [0.685, 0.731] | 0.007 | 0.709 | 0.709 |
| first_token_entropy | pre-generation | 0.878 [0.856, 0.898] | 0.941 [0.926, 0.953] | 0.035 | 0.437 | 0.644 |
| popularity | pre-generation | 0.673 [0.636, 0.708] | 0.769 [0.736, 0.803] | 0.073 | 0.548 | 0.693 |
| metadata | pre-generation | 0.826 [0.800, 0.850] | 0.907 [0.886, 0.927] | 0.045 | 0.469 | 0.649 |
| neg_mean_logprob | post-generation | 0.854 [0.830, 0.877] | 0.921 [0.902, 0.939] | 0.050 | 0.451 | 0.649 |

Test set: n = 1284, error base rate = 0.709. Probe uses layer 15 (selected on validation AUROC).
<!-- results:end -->

![layer sweep](results/mvp_popqa_llama3b/20261013-204454/figures/layer_sweep.png)

The full write-up, with shortcut controls, calibration, label sensitivity and limitations, is in [docs/results.md](docs/results.md).

## Setup (Windows, AMD RX 6700 XT / RDNA2)

This is the configuration the reported results were produced on.

- **Why a nightly build:** AMD's official Windows PyTorch wheels do not support RDNA2 (gfx103x). The project uses AMD's [TheRock](https://github.com/ROCm/TheRock) nightly builds from the staging index `https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/`.
- **Pinned wheel:** `torch==2.9.1+rocm7.14.0a20260612` with the matching `rocm[libraries]`, `rocm-sdk-core` and `rocm-sdk-libraries-gfx103x-all` (all `7.14.0a20260612`), locked in `uv.lock`. Nightly wheels can be deleted upstream. If that happens, pick a newer gfx103X nightly, update the pins in `pyproject.toml`, and run `uv lock`.
- The Linux TheRock wheel declares a `triton` dependency that the Windows wheel does not have. `pyproject.toml` disables it with an `override-dependencies` entry, so one lockfile resolves for both platforms.

Steps:

```bash
python -m pip install --user uv
# The venv lives at a short path: Windows long paths (LongPathsEnabled) are disabled on many machines.
# Enabling LongPathsEnabled is the alternative.
setx UV_PROJECT_ENVIRONMENT "C:\venvs\halprobe"
# open a new terminal so the variable is set, then:
uv sync                              # or: python -m uv sync, if uv is not on PATH
uv run hf auth login                 # Llama 3.2 is gated: accept the license on its Hugging Face page first
uv run python scripts/env_check.py   # GPU visible, GPU fp16 vs CPU fp32 agreement, model load, VRAM
```

Machines with an integrated GPU expose it as a second device (here `cuda:1`). The pipeline therefore selects the GPU by name (`gpu_name_hint: "6700"` in the config) and never falls back to CPU silently.

## Setup (Linux, untested alternative)

On Linux, the usual route for RDNA2 is a ROCm 6.x/7.x PyTorch build with `HSA_OVERRIDE_GFX_VERSION=10.3.0`. **This path was not tested for this project.** The lockfile pins the CPU build of torch for Linux, which CI uses, so a ROCm user has to point the `torch` source in `pyproject.toml` at a ROCm index and re-lock.

## Reproduce

```bash
uv run python -m halprobe run configs/mvp_popqa_llama3b.yaml
```

This runs every stage and skips stages whose outputs are already cached. The stages can also run one at a time, each running everything upstream of it: `python -m halprobe {data,generate,label,extract,probe,baselines} CONFIG`.

- `artifacts/<stage>/<key>/` holds the cached stage outputs (gitignored). The key hashes the config fields the stage depends on plus the upstream keys, so changing, say, the prompt rebuilds generation, labels, extraction, probes and baselines, but not the data split.
- `results/<experiment>/<run_id>/` holds the committed report: the resolved config, `metrics.json`, `env.json`, stage keys, figures and tables.
- On the RX 6700 XT, generating the 12,479 answers took about 31 minutes and extracting the hidden states about 15 minutes. Probes, baselines and the report then run on the CPU.

Manual label review:

```bash
uv run python scripts/export_review_sample.py configs/mvp_popqa_llama3b.yaml   # 100 random unambiguous examples + all ambiguous ones
# fill the manual_label column with correct / incorrect / ambiguous
uv run python scripts/score_review.py results/review/mvp_popqa_llama3b/review_sample.csv
```

Result tables in this README and in `docs/results.md` are rendered from a run's `metrics.json`. The script refuses runs labelled with pseudo labels:

```bash
uv run python scripts/render_results.py results/mvp_popqa_llama3b/<run_id>
```

## Tests

```bash
uv run pytest            # CPU only, offline, about 10 s: tiny random Llama + synthetic PopQA-format data
uv run pytest -m gpu     # GPU kernel checks on the real device
uv run python -m halprobe make-tiny artifacts/tiny
uv run python -m halprobe run configs/tiny_cpu.yaml    # end-to-end smoke run on CPU
```

The tiny configuration uses a randomly initialised model that answers nothing correctly, so its labels are deterministic **pseudo labels** (a hash of the example id). Its results exercise the pipeline and are never reported. CI runs the CPU suite on Linux.

## Method in brief

- **Data:** PopQA (`akariasai/PopQA` at a pinned revision). Question texts shared by several entities are dropped, because their popularity is undefined; same-entity duplicates are merged. The 70/15/15 split is grouped by normalised subject string, so no entity appears in two splits.
- **Prompt and generation:** Llama-3.2 chat template with a short-answer system prompt, four fixed few-shot turns and a pinned template date. Greedy decoding stops at end-of-turn or the first newline.
- **Labels:** SQuAD-style normalisation and word-boundary alias matching. Answers that are empty, refusals, truncated, or only partially overlapping an alias are labelled *ambiguous* and excluded from training and the headline metrics. A label-sensitivity analysis counts them as correct and, separately, as incorrect.
- **Probes:** one hidden state per layer at the last prompt token, before any answer token exists. Per layer, a standardised L2 logistic regression is trained on the training split; C and the reported layer are chosen by **validation** AUROC. The test-set maximum over layers is never reported as the result.
- **Baselines:** first-token entropy (same position as the probe), mean answer-token log-probability (post-generation), a popularity model (log10 s_pop), and a **metadata model** (popularity + one-hot relation). The metadata model is the key control: PopQA errors are strongly predictable from entity popularity and relation type, and a probe could reach a high AUROC just by encoding those. The write-up therefore also reports within-relation AUROC, AUROC per popularity quintile, and the AUROC gain of adding the probe to the metadata model.

## Project layout

```
configs/                 mvp_popqa_llama3b.yaml (the MVP run), tiny_cpu.yaml (CPU smoke run)
src/halprobe/
  config.py              strict typed YAML config
  cache.py               content-addressed stage dirs, crash-safe writes
  device.py, envcheck.py GPU selection by name, environment check
  data.py                PopQA loading, dedupe, entity-grouped split
  prompts.py, models.py  few-shot chat prompts, model loading
  batching.py            OOM back-off
  generation.py          greedy generation with token log-probs (resumable)
  labeling.py, text.py   normalisation, alias matching, ambiguity reasons
  extraction.py          last-prompt-token hidden states in a resumable fp16 memmap
  probes.py              per-layer logistic probes
  baselines.py           uncertainty and shortcut baselines
  metrics.py             AUROC, AUPRC, ECE, risk-coverage, bootstrap CIs
  analysis.py            shortcut and label-sensitivity analyses
  plotting.py, report.py figures, tables, run report
  review.py              manual review export and scoring
  pipeline.py, __main__.py  stage orchestration and CLI
  tiny.py                offline tiny tokenizer/model/data for tests
scripts/                 env_check, export_review_sample, score_review, render_results
tests/                   one test module per source module
results/                 committed run reports and the manual review
docs/results.md          results write-up
```

## Related work

- **Kadavath et al. 2022, "Language Models (Mostly) Know What They Know":** trains a P(IK) head that predicts from the question alone whether the model knows the answer. This project asks the same pre-generation question with a frozen 3B model and a linear probe instead of fine-tuning.
- **Burns et al. 2022, "Discovering Latent Knowledge in Language Models Without Supervision" (CCS):** finds truth directions without labels. This project is supervised and probes for the model's *own* future error, not the truth of a given statement.
- **Azaria & Mitchell 2023, "The Internal State of an LLM Knows When It's Lying" (SAPLMA):** probes hidden states to classify statement truthfulness. Here the probe reads the prompt before any statement is generated.
- **Marks & Tegmark 2023, "The Geometry of Truth":** shows linear truth representations on curated true/false datasets. This project tests whether a linear direction also predicts errors on open-ended factual QA.
- **Kossen et al. 2024, "Semantic Entropy Probes":** probes hidden states to predict semantic entropy, including before generation. This project targets correctness directly, in a small model, with an explicit popularity/relation shortcut control.

## Limitations

See [docs/results.md#limitations](docs/results.md#limitations).
