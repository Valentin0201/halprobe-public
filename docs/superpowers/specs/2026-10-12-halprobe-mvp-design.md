# halprobe MVP: Linear Probes for Pre-Generation Hallucination Detection

- **Date:** 2026-10-12
- **Status:** Approved in brainstorming (all five design sections). This file is pending written-spec review.
- **Scope:** MVP only. Phase 2 and Phase 3 each get their own spec → plan cycle once the MVP has produced real results.

## 1. Goal and research question

Can a linear probe on a small instruct LLM's hidden states, read at the **last prompt token (before any answer token is generated)**, predict whether the model's closed-book answer will be factually wrong? How does it compare with standard uncertainty baselines and with a popularity/metadata shortcut control?

The deliverable is a reproducible GitHub repo plus a results write-up for a CV and possibly a workshop paper. Rigor matters more than breadth: entity-disjoint splits, shortcut controls, selection on validation only, confidence intervals, and honest negative results. No number in the README, docs or plots may be fabricated or hand-typed. Every number must trace back to a recorded run.

**Related work to position against:** Kadavath et al. 2022 (P(IK), P(True)); Burns et al. 2022 (CCS); Azaria & Mitchell 2023 (SAPLMA); Marks & Tegmark 2023 (geometry of truth); Kossen et al. 2024 (Semantic Entropy Probes). Our angle: a *pre-generation* signal in a 3B model, evaluated against an explicit popularity + relation-type shortcut control.

## 2. Decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Platform | **Windows 11 + AMD TheRock nightly torch** (primary); CPU torch for tests/CI; Linux ROCm documented as an untested alternative | A spike on 2026-10-12 verified correctness on the RX 6700 XT (section 3) |
| MVP model | **meta-llama/Llama-3.2-3B-Instruct**, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` | More balanced error/correct labels than at 1.5B; no thinking mode; widely used in the literature. Gated: the user must accept the license and run `hf auth login` |
| Prompt | **Few-shot chat**: system prompt plus 4 fixed Q/A turns, greedy decoding, stop at newline/EOT, max 32 new tokens | Forces short answers so alias matching is clean |
| Ambiguous labels | **Alias matching only**. Ambiguous cases are excluded from the headline metrics, counted, and analysed for sensitivity. A manual 100-sample review estimates the noise rate | A local judge ≥7B does not fit 12 GB in fp16, and using the probed model as its own judge would be circular |
| Hidden-state storage | **One layer-major fp16 memmap** `(n_layers+1, N, d_model)` per extraction position, with a `done` mask and `meta.json`. All layers are kept, including the embeddings | About 2.5 GB in total; each layer is one contiguous slice |
| Orchestration | **Own stage CLI**; cache directories keyed by hashes of the relevant config fields and the upstream keys | Linear pipeline, Windows-friendly, and the caching logic stays testable under TDD |
| Phase 3 | **Decide after the MVP**; (a) quantization is the tentative favourite | The MVP only adds a weight-transform hook in `models.py` |

## 3. Platform and environment (spike-verified)

**Spike results (2026-10-12):**
- **torch build:** `torch==2.9.1+rocm7.14.0a20260612` from `https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/`. Only the staging index carries torch for gfx103X; `v2/` has the ROCm libraries only.
- **Device detection:** no `HSA_OVERRIDE_GFX_VERSION` is needed. The RX 6700 XT shows up as `cuda:0` (`gfx1031`, 11.98 GiB). The iGPU (`gfx1036`) shows up as `cuda:1`.
- **Kernel correctness:** GPU fp16 matched CPU fp32 for matmul, softmax, layer_norm, SDPA and SiLU (relative error ≈3e-4). A 2048³ fp16 matmul reached 19 TFLOP/s.
- **Real model (Qwen2.5-1.5B-Instruct):** GPU fp16 vs CPU fp32 gave 8/8 identical greedy answers and a minimum hidden-state cosine of 1.0000 across all layers. A batched forward pass over 64 prompts with all hidden states took 0.52 s at 3.69 GiB peak.

**Gotchas the setup must handle:**
- **Missing index dependency:** the nightly index lacks `mpmath<1.4`. Only `torch` and the `rocm*` packages come from the TheRock index (an explicit uv index); everything else comes from PyPI.
- **Nightly lifetime:** nightly wheels may be deleted. The exact version is pinned in `uv.lock`, and the README documents the URL and the failure mode.
- **Long paths:** `LongPathsEnabled=0` on this machine. The venv lives at a short path via `UV_PROJECT_ENVIRONMENT` (e.g. `C:\venvs\halprobe`). The README also mentions enabling long paths as an alternative.
- **Device selection:** the device is selected by name hint, never by bare `cuda`.
- **Unverified model:** Llama-3.2-3B itself was not run in the spike. `scripts/env_check.py` (plan task 1) verifies it.

**Tooling:**
- **Environment:** `uv` with `pyproject.toml` and `uv.lock`, Python 3.11.
- **Per-platform torch:** chosen via `[tool.uv.sources]` markers. Windows gets the TheRock wheel above; Linux gets `torch==2.9.1+cpu` from the PyTorch CPU index (for CI).
- **Dependencies:** transformers (4.57.x, pinned by the lock), datasets, scikit-learn, numpy, matplotlib, pyyaml, pytest.
- **CI:** GitHub Actions on Linux with CPU torch, running `pytest -m "not gpu"`.

## 4. Repository layout

```
pyproject.toml, uv.lock, README.md, .gitignore
configs/
  mvp_popqa_llama3b.yaml     # the MVP experiment
  tiny_cpu.yaml              # tiny random model, used by tests and smoke runs
src/halprobe/
  config.py      # YAML -> typed dataclasses; strict validation; stage-key hashing
  cache.py       # stage dirs, atomic writes (.tmp -> rename), completion markers
  device.py      # device resolution (name hint), seeding
  data.py        # PopQA load, dedupe, entity-grouped split
  prompts.py     # few-shot chat prompt construction
  models.py      # model/tokenizer loading (+ weight_transform hook, identity in MVP)
  generation.py  # batched greedy generation, token logprobs, first-token entropy
  labeling.py    # normalization, alias matching, ambiguity categories
  extraction.py  # hidden states -> resumable memmap
  probes.py      # per-layer logistic regression, C tuned on val
  baselines.py   # logprob, entropy, popularity, metadata
  metrics.py     # AUROC, AUPRC, ECE + reliability bins, risk-coverage, AURC, bootstrap
  analysis.py    # shortcut analyses + label sensitivity
  plotting.py
  pipeline.py    # stage graph + orchestration
  __main__.py    # CLI
scripts/
  env_check.py
  export_review_sample.py
  score_review.py
  render_results.py
tests/
docs/results.md
artifacts/        # gitignored: cached stage outputs
results/          # committed: per-run configs, metrics, figures
```

**CLI:**
- `python -m halprobe run <config>` runs all stages and skips cached ones.
- `python -m halprobe <stage> <config>` runs a single stage.

## 5. Pipeline stages

Each stage is a function `(config, upstream outputs) -> output directory`. The output directory is `artifacts/<stage>/<stage_key>/`, where `stage_key = sha256(relevant config subset + upstream stage keys)[:12]`. Stage keys therefore cascade: changing the prompt invalidates generation, labeling and extraction, but not data.

| Stage | Relevant config fields | Output |
|---|---|---|
| data | dataset name/revision, dedupe rules, split ratios, seed, few-shot subjects | `examples.jsonl` (with split column), `data_report.json` |
| generate | data key, model id/revision, dtype, prompt template + shots, decoding params | `generations.jsonl`, `meta.json` |
| label | generate key, normalization/matching rules | `labels.jsonl`, `label_report.json` |
| extract | data key, model id/revision, dtype, prompt template + shots, position | `states.f16.mmap`, `done.npy`, `meta.json` |
| probe | extract + label keys, probe hyperparameter grid, seed | per-layer scores/coefficients, `probe_report.json` |
| baselines | generate + label + data keys, seed | per-method scores |
| evaluate/report | all upstream keys | `results/<experiment>/<timestamp>/…` |

Batch size is **not** part of any stage key, because OOM back-off may change it. It is recorded in the stage `meta.json`.

### 5.1 Data (`data.py`)

**Source:** `akariasai/PopQA`, revision `098765c79ea10a2cb19c828324e33281b8336ec0` (14,267 rows).

**Observed in the data (2026-10-12):**
- 13,066 unique question texts.
- 883 question texts occur in more than one row; 561 of these have *different* gold answer sets because distinct entities share a name (e.g. "What is Will Smith's occupation?" → film producer, s_pop 345k, vs. a screenwriter, s_pop 5k).
- 1,101 subject strings map to more than one `subj_id`.

**Dedupe rules:**
1. Group rows by normalized question text.
2. If a group has exactly one `subj_id`, merge it into one example. Gold answers are the union of `possible_answers`, and `s_pop` is shared.
3. If a group has more than one `subj_id`, **drop it**. The question is inherently ambiguous and its popularity is undefined.
4. All counts go to `data_report.json`.

**Split:**
- 70/15/15 train/val/test, grouped by **normalized subject string** (not `subj_id`, since that would leak identical question texts across splits), with a fixed seed.
- At runtime, the stage asserts that no subject string appears in two splits.
- The relation distribution and positive rate per split are reported. Relation counts are very skewed, from director 1,999 down to color 34.

**Few-shot guard:** the stage asserts that no few-shot subject occurs as a PopQA subject.

### 5.2 Prompt (`prompts.py`)

- **Format:** the Llama-3.2 chat template with a system message: *"Answer the question with a short phrase (a name, place, date, or category). Do not explain."*
- **Shots:** four fixed user/assistant turns. None of the subjects occurs as a PopQA subject (checked on 2026-10-12):
  - What is Ada Lovelace's occupation? → mathematician
  - Who was the director of Spirited Away? → Hayao Miyazaki
  - Who is the author of Wuthering Heights? → Emily Brontë
  - What genre is Blade Runner? → science fiction
- **Probe position:** the target question comes last, followed by the assistant header (`add_generation_prompt=True`). The probe position is the final token of that header.

### 5.3 Generation (`generation.py`)

- **Decoding:** batched, left padding, greedy (`do_sample=False`), `max_new_tokens=32`. Generation stops at `<|eot_id|>` or the first newline.
- **Per-example record:** `id`, answer text, generated token ids, per-token log-probs (chosen token, log-softmax over the full vocabulary), first-token entropy (natural log, full vocabulary), number of tokens, and `finish_reason ∈ {stop, length}`.
- **Resume:** records are appended to `generations.jsonl` after each batch. On restart, IDs already present are skipped.

### 5.4 Labeling (`labeling.py`)

- **Normalization** (SQuAD-style): Unicode NFKD with combining marks stripped, lowercase, punctuation removed, articles (a/an/the) removed, whitespace collapsed.
- **correct:** some normalized gold alias occurs in the normalized answer **with word boundaries**. This prevents the PopQA alias "pol" from matching "police".
- **Check order:** first `correct`, then the ambiguity reasons in the order below, then `incorrect`.
- **ambiguous**, with a reason, checked in this order:
  - `empty`
  - `refusal`: a fixed pattern list such as "i don't know", "unknown", "not sure", "cannot".
  - `truncated`: `finish_reason=length` without a correct match.
  - `partial`: shares at least one content word (stopwords removed) with some alias but is not a full match, e.g. "producer" vs. "film producer".
- **incorrect:** everything else.
- **Probe target:** `y = 1` means incorrect. Ambiguous examples are excluded from training and headline evaluation.
- **Manual review:**
  - `export_review_sample.py` writes 100 seeded random **unambiguous** examples to CSV (question, answer, gold aliases, auto label, empty `manual_label`). All ambiguous examples go to a separate CSV.
  - `score_review.py` computes the label-noise rate from the filled CSV for `docs/results.md`.

### 5.5 Extraction (`extraction.py`)

- **Forward pass:** separate, prompt-only, batched with left padding, `output_hidden_states=True`. Each hidden state is read at position −1 (the last prompt token).
- **Coverage:** all examples are extracted, including ambiguous ones; filtering happens downstream.
- **Storage:**
  - `states.f16.mmap` has shape `(n_layers+1, N, d_model)`: `(29, N, 3072)` for Llama-3.2-3B. Index 0 is the embedding output. The last index is post-final-norm, as returned by HF.
  - `meta.json` holds model, revision, position, dtype, shape and the ordered example IDs.
- **Resume:** `done.npy` (bool[N]) is atomically replaced after each batch is flushed. On restart, only rows not yet done are computed.
- **Consistency check:** the argmax of the last-position logits must equal the first generated token id for at least 99% of examples. Mismatches are logged; falling below the threshold is an error. This catches padding and position bugs.

### 5.6 Probes (`probes.py`)

- **Per layer ℓ:** `StandardScaler` is fit on train only. Then `LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000)` with `C ∈ logspace(-4, 2, 7)`, choosing C by **validation AUROC**. `random_state` comes from the config.
- **Outputs:** validation and test scores per layer, the chosen C, coefficients, and intercept.
- **Headline probe:** the layer with the best **validation** AUROC, evaluated on test. The test-curve maximum is never reported as the result.

### 5.7 Baselines (`baselines.py`)

All scores are oriented so that higher means more likely an error.

| Baseline | Timing | Score |
|---|---|---|
| First-token entropy | pre-generation (same logits as the probe position) | entropy |
| Popularity | pre-generation | logistic regression on `log10(s_pop)`, fit on train |
| Metadata (shortcut control) | pre-generation | logistic regression on `log10(s_pop)` + one-hot relation, fit on train |
| Layer-0 probe (lexical control) | pre-generation | probe on the embedding layer (comes from `probes.py`) |
| Mean token log-prob | post-generation | −mean log-prob over answer tokens, excluding the stop token |

- **Raw scores** (entropy, log-prob) are used for AUROC, AUPRC and risk–coverage.
- **Calibrated scores** for ECE come from a 1-D logistic regression fit on train.

### 5.8 Evaluation (`metrics.py`, `analysis.py`)

**Headline metrics** (test set, each with a 95% bootstrap CI over 1,000 resamples):
- AUROC.
- AUPRC, with the error base rate reported alongside.
- ECE (15 equal-width bins) and reliability-diagram bins.
- Risk–coverage: sort by P(error) ascending; at coverage k/N answer the k lowest-risk examples; risk is the error rate among them. Summarized as AURC, plus risk at 50% and 80% coverage.

**Shortcut analyses:**
1. **Within-relation AUROC:** macro average over relations with at least `evaluation.min_per_class_relation` examples of each class in test (default 20). Per-relation values are reported.
2. **Incremental value:** metadata baseline vs. metadata + probe score. The 2-feature logistic regression is fit on **val**, where both scores are out-of-sample; train-set probe scores are in-sample and would be over-weighted. ΔAUROC on test with a paired bootstrap CI.
3. **AUROC per s_pop quintile.**

**Label sensitivity:** the headline metrics are recomputed with ambiguous examples counted as correct, and again with them counted as incorrect.

### 5.9 Outputs and reporting

**Per run:** `results/<experiment>/<timestamp>/` contains:
- `config.yaml` (resolved)
- `metrics.json`
- `env.json` (torch/HIP versions, GPU name, OS, git commit, package versions)
- `stage_keys.json`
- `figures/*.png`
- `tables/summary.md`

**Figures:**
- Layer sweep: validation and test AUROC per layer, a test CI band, baselines as horizontal lines, and the val-selected layer marked.
- Reliability diagram.
- Risk–coverage curves for all methods.
- Per-relation AUROC.

**Reporting:**
- `scripts/render_results.py` rewrites the README results table between `<!-- results:start -->` / `<!-- results:end -->` from a given run's `metrics.json`.
- `docs/results.md` is written after real runs and cites run IDs, including negative results and limitations: label noise, shortcut learning, model size, and the single dataset.

## 6. Configuration

- **Format:** one YAML per experiment, loaded into frozen dataclasses. Unknown keys are an error.
- **Sections:** `experiment`, `seed`, `device` (`auto` | `cpu` | `cuda:N`, plus `gpu_name_hint`), `data`, `model`, `prompt`, `generation`, `labeling` (`alias`, or `pseudo` for the tiny CPU config only), `extraction`, `probe`, `evaluation`.
- **Tests:** `configs/tiny_cpu.yaml` points to a tiny random model and a synthetic dataset fixture.

## 7. Error handling

- **Device:**
  - `device: auto` picks the GPU whose name contains `gpu_name_hint` ("6700"). If a hint is set and no match is found, it raises; there is **no silent CPU fallback** for real runs.
  - `device: cpu` must be chosen explicitly.
- **OOM:** on `torch.OutOfMemoryError`, the batch size is halved and the batch retried (down to 1). The back-off is logged.
- **Atomicity:**
  - Stages write to `<dir>.tmp` and rename it on completion. A `.tmp` directory is never a cache hit.
  - Resumable stages (generate, extract) continue inside `.tmp`.
- **Validation:** strict config schema; runtime leakage assertion; minimum class count per split before probe training (error if fewer than `probe.min_class_count` per class in val or test; default 20, lowered in `tiny_cpu.yaml`).
- **Seeds:** Python, numpy and torch seeds plus sklearn `random_state` all come from the config seed.

## 8. Testing

**Unit tests (written before the code, TDD):**
- **Labeling:** normalization (accents, articles, punctuation); word-boundary matching (`pol` vs `police`); each ambiguity category; refusal patterns.
- **Data:** dedupe on a fixture with same-name entities (the Will Smith case); split determinism per seed, ratios, and no subject leakage; few-shot guard.
- **Metrics:** AUROC/AUPRC vs. sklearn on small arrays; ECE on a hand-computed toy; risk–coverage and AURC on a hand-computed toy including ties; bootstrap determinism with a fixed seed.
- **Config/cache:** a stage key changes iff a relevant field changes; unknown keys are rejected; atomic write; a `.tmp` directory is not a hit.
- **Resume:** extraction interrupted after k batches and then resumed is bit-identical to an uninterrupted run; generation JSONL resume skips completed IDs and produces no duplicates.

**Pipeline integration test (CPU, a few seconds, fully offline):**
- **Model and tokenizer:** a randomly initialized `LlamaForCausalLM` (2 layers, hidden size 64) and a tiny byte-level BPE tokenizer built in the test, with a Llama-style chat template. No download and no HF token.
- **Data:** about 60 synthetic PopQA-like rows.
- **Labels:** a deterministic pseudo-labeler is injected so that probes see both classes (a random model answers nothing correctly). Real labeling is covered by the unit tests.
- **Asserts:** all outputs exist with the expected shapes; the consistency check runs; a second run is a full cache hit.

**GPU tests:** marked `@pytest.mark.gpu` and skipped without a GPU. `env_check.py` covers the real-hardware path.

**Target:** the full CPU suite finishes in under about 60 s.

## 9. MVP success criteria

1. `scripts/env_check.py` passes on the RX 6700 XT: device found, GPU vs CPU correctness, Llama-3.2-3B loads, and VRAM is reported.
2. `python -m halprobe run configs/mvp_popqa_llama3b.yaml` runs end to end on the GPU and a rerun hits every cache.
3. The run produces: the layer-sweep plot, the baseline comparison table with CIs, the reliability diagram, risk–coverage curves, shortcut analyses, the label sensitivity analysis, and the review CSVs.
4. The CPU test suite passes locally and in CI.
5. The README results table is generated from the run's `metrics.json`, and `docs/results.md` reports the findings, including negative results.

## 10. Out of scope (MVP)

- **Phase 2 items:** extraction at the first and last generated token, P(True), self-consistency, TriviaQA transfer, data-efficiency curves, a second model family. The interfaces are designed so each fits as a new config, position or baseline without restructuring.
- **Phase 3:** to be decided after the MVP. The only MVP accommodation is the `weight_transform` hook in `models.py`.
- **LLM judges:** none, local or remote.

## 11. Risks

| Risk | Mitigation |
|---|---|
| TheRock nightly wheel deleted upstream | Pinned in the lock; README documents the fallback (newer gfx103X nightly + re-lock, or Linux ROCm) |
| Llama-3.2-3B misbehaves on the gfx1031 nightly | `env_check.py` compares GPU and CPU outputs before any real run |
| Very low accuracy leads to class imbalance | Report AUPRC with the base rate; the 3B model was chosen partly for this |
| The probe only learns relation/popularity | Metadata baseline, within-relation AUROC, incremental ΔAUROC |
| Alias-matching label noise | Ambiguity exclusion, sensitivity analysis, manual review noise estimate |
