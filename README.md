# MedMCQA and MedQA: deduplicated splits with three MedMCQA evaluation subsets

One function returns fixed, leak-free splits for MedMCQA and MedQA in a common format, plus the single prompt format used by the model wrapper and training code.

```python
import os
os.environ["DATA_DIR"] = "/shared/data"   # set BEFORE importing medsplits
from medsplits import load_splits, build_messages

medmcqa = load_splits("medmcqa")   # {"train", "hp_dev", "calib", "thresh", "test"}
medqa = load_splits("medqa")       # {"train", "val", "test"}
messages = build_messages(medmcqa["train"][0])                       # chat messages for the model
messages = build_messages(medmcqa["train"][0], include_answer=True)  # adds the gold letter (training)
```

Example format (any number of options is supported):

```python
{"id": "medmcqa-123", "source": "medmcqa", "question": "...",
 "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "B"}
```

## What each split is for

| Dataset | Split | Rows | Use |
|---|---|---|---|
| MedMCQA | `train` | 158510 | Fine-tuning |
| MedMCQA | `hp_dev` | 1998 | Choose learning rate and LoRA rank |
| MedMCQA | `calib` | 1499 | Fit the temperature T of each checkpoint |
| MedMCQA | `thresh` | 2497 | Choose the abstention threshold of each checkpoint |
| MedMCQA | `test` | 4168 | Final reported numbers only (the official MedMCQA validation set) |
| MedQA | `train` | 9178 | Fine-tuning |
| MedQA | `val` | 1000 | Validation |
| MedQA | `test` | 1273 | Final numbers; source of MedAbstain (#4), never trimmed |

`hp_dev`, `calib`, `thresh` and `test` share no question with each other or with any train set, so tuning, temperature fitting and threshold selection never see the same questions and never touch `test`. MedMCQA has no `val` key any more.

## Folder structure

### What is in the repository

The repository contains only the README and the notebook. **No dataset files are uploaded: everyone builds their own copy** by running the notebook, and the checksums below prove that the copies are identical.

```
.
├── README.md
└── FME_MedMCQA_MedQA_splits_v6.ipynb   Runs end to end: sets DATA_DIR, writes the package and tests,
                                        builds the splits, runs the tests, prints the README numbers
```

Running the notebook creates the following in Colab's `/content/proj` (temporary, recreated on every run), so it does not need to be stored:

```
/content/proj/
├── medsplits/
│   ├── __init__.py                     Public names: load_splits, build_all, build_messages, ...
│   ├── data.py                         Loading, common format, deduplication, subsets, similarity filter, load_splits
│   └── prompts.py                      The one prompt format (PROMPT_VERSION, build_messages)
└── tests/
    ├── test_data.py                    Needs the real data: leakage, no shared questions, similarity filter,
    │                                   determinism in fresh processes, format, config guard
    ├── test_subsets.py                 Offline: subset carving, similarity filter, guard on subsets
    ├── test_dedupe_logic.py            Offline: key functions and the two-tier dedupe on tiny examples
    └── test_prompts.py                 Offline: prompt text for 4 and 6 options, message structure
```

`__pycache__/` and `.pytest_cache/` folders appear after running; they are generated and not part of the project.

### Where the data lives (not part of the submission)

Data files are kept outside the code, in the folder named by the `DATA_DIR` environment variable (`/shared/data` by default; the Colab run used Google Drive `MyDrive/shared/data`). They are not committed to git; each person produces them by running the notebook.

```
<DATA_DIR>/
├── splits/
│   ├── medmcqa.json        Question ids of train / hp_dev / calib / thresh / test (the source of truth)
│   ├── medqa.json          Same for MedQA (train / val / test)
│   └── stats.json          Settings used, dataset commits, library versions, sizes, removed counts
├── processed/
│   ├── medmcqa.jsonl       Every kept example in the common format, one JSON object per line
│   └── medqa.jsonl         (no split column: use load_splits, or look the ids up in splits/*.json)
├── hf_cache/               Raw Hugging Face downloads
└── cache/                  Similarity-search results (speed-up only, safe to delete)
```

`load_splits` reads the id lists in `splits/` and looks the records up in `processed/`. If these files do not exist yet, it builds them first (about 90 MB download; several minutes because of the similarity search).

### What is in `medsplits/data.py`

| Name | Role |
|---|---|
| Settings block (`SEED`, `SPLIT_SIZES`, `SIMILARITY_FILTER`, `SHORT_STEM_WORDS`, `DEDUPE_SCHEME`, `HF_REPOS`, `HF_REVISIONS`, `SPLIT_PRIORITY`, `DATASET_ORDER`) | Everything that shapes the splits, in one place |
| `current_config()` | Snapshot of those settings, saved in `stats.json` and checked by `load_splits` |
| `normalize()` | Lowercase, punctuation to spaces, collapse whitespace |
| `exact_key()`, `leak_key()` | Two definitions of "the same question" (see below) |
| `load_medmcqa_raw()`, `load_medqa_raw()` | Download (pinned commit) and convert to the common format |
| `dedupe_within()` | Remove repeats inside one list, keeping the first |
| `split_names()` | Split names of a dataset, in the order used everywhere |
| `near_duplicate_ids()` | Ids of rows with a near-copy (Levenshtein similarity at or above the threshold) in a reference set |
| `carve_subsets()` | Deterministic, disjoint named subsets from train, optionally skipping near-copy rows |
| `global_dedupe()` | Cross-split and cross-dataset removal with priority test > val > hp_dev > calib > thresh > train |
| `build_all()` | Runs the whole pipeline and writes `splits/`, `processed/`, `stats.json` |
| `load_splits()` | The entry point |

## How the splits are built

- **Sources (pinned):** MedMCQA `openlifescienceai/medmcqa` at commit `91c6572c454088bf71b679ad90aa8dffcd0d5868`; MedQA (English, 4 options) `GBaker/MedQA-USMLE-4-options` at commit `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`.
- **MedMCQA:** the official test set has no answers, so the official **validation** set is our `test`. `hp_dev` (2000), `calib` (1500) and `thresh` (2500) are carved from the deduplicated train set, as consecutive blocks of one seeded shuffle, so they cannot overlap.
- **MedQA:** Hugging Face has only train and test. `val` (1000 rows) is carved out of train. The official test set is untouched (all 1273 rows), so MedAbstain (#4), which is built from MedQA test questions, cannot overlap MedQA train or val.
- **Seed 42.** Rows are sorted by id before shuffling, so the result does not depend on load order.
- **Deduplication.** Questions are compared after lowercasing, replacing punctuation with spaces and collapsing whitespace. When a question appears in more than one place, one copy is kept with priority test > val > hp_dev > calib > thresh > train (MedQA before MedMCQA on ties).
  - Inside train, and between the two train sets: only true copies are removed (`exact_key`: same question and same options, option order ignored).
  - Eval splits, and train versus eval splits: strict matching (`leak_key`): stems of 10 or more words match on the question alone, shorter stems must also have the same options. No eval question appears in any train set or in another eval split.
- **Similarity filter (MedMCQA subsets only).** MedMCQA's authors left questions with Levenshtein similarity above 0.9 to other questions out of its dev/test sets. Random train rows are much more often near-copies of other train rows (21.8% at 0.9 or more, against 0.5% for `test`), so random subsets would be easier than `test` and would bias the fitted temperature and threshold. A candidate row may enter `hp_dev`, `calib` or `thresh` only if no other train question has similarity of 0.85 or more to it (identical text ignored). The comparison uses the whole deduplicated train set before carving, so the subsets are also free of near-copies of each other.

### Effect of the filter (share of rows with a different train question at or above a similarity, sample of up to 1000 rows)

| Similarity | `hp_dev` | `calib` | `thresh` | `test` | random train rows (no filter) |
|---|---|---|---|---|---|
| 0.9 | 0.0% | 0.0% | 0.0% | 0.6% | 21.8% |
| 0.85 | 0.3% | 0.0% | 0.2% | 2.7% | not measured |
| 0.8 | 11.1% | 8.2% | 10.0% | 8.0% | 34.6% |
| 0.7 | 33.9% | 31.5% | 31.9% | 22.5% | 51.5% |
| median nearest similarity | 0.64 | 0.63 | 0.63 | 0.56 | 0.71 |

The filter removes the near-copy excess at 0.8 and above (`hp_dev` 11.1% vs `test` 8.0%, where unfiltered train would be 34.6%). A smaller difference remains below the threshold (0.7 and the median), because `test` comes from different exams than train; this is a property of the source data, not something the filter can remove. Calibration and threshold choices should still be re-checked against `test` behaviour when final numbers are reported.

## Results

| Dataset | Raw train | Raw test | Final train | `val` | `hp_dev` | `calib` | `thresh` | Final test |
|---|---|---|---|---|---|---|---|---|
| MedMCQA | 182822 | 4183 (HF validation) | 158510 | none | 1998 | 1499 | 2497 | 4168 |
| MedQA | 10178 | 1273 | 9178 | 1000 | none | none | none | 1273 |

Subsets are slightly below their nominal sizes (2000 / 1500 / 2500) because rows that matched another eval question in the leak guard were dropped (2, 1 and 3 rows).

| Dataset | Train copies removed | Test copies removed | Train rows matching an eval question | Subset rows matching another eval question | True copies across the two train sets |
|---|---|---|---|---|---|
| MedMCQA | 18106 | 15 | 206 | 2 (`hp_dev`), 1 (`calib`), 3 (`thresh`), 0 (`test`) | 0 |
| MedQA | 0 | 0 | 0 | 0 | 0 |

Similarity filter on MedMCQA: 12000 shuffled candidates examined, 3540 flagged, subsets filled from the rest.

Total removed: 18333 (all MedMCQA). Removed copies whose correct answer text differed from the kept copy (label noise in the source): MedMCQA train 400, MedMCQA test 2.

## Running it

1. Open the notebook in Colab and run all cells in order. The setup cell sets `DATA_DIR` (a Google Drive path; edit that one line for your own Drive or folder). Running the notebook is all that is needed to obtain the dataset: it writes the code, downloads both datasets at the pinned commits and builds the splits. The test cell is optional for just obtaining the data.
2. Check your copy: run the checksum cell and compare with the reference checksums in the Reproducibility section.
3. Outside Colab: `pip install datasets pytest rapidfuzz==3.14.6`, extract the package and tests from the notebook, set `DATA_DIR`, then from the folder that contains `medsplits/` run `python -m pytest -q tests` (33 tests; about 8 minutes on Colab because two fresh builds recompute the similarity search; the first run also downloads the data).
4. Set `DATA_DIR` **before** importing `medsplits`; it is read once at import time.

## Reproducibility

The splits are a function of: the two pinned dataset commits, the settings block in `data.py`, `rapidfuzz` 3.14.6 and the code. With those unchanged, building again gives identical splits. `test_deterministic_in_fresh_processes` builds twice in separate Python processes with different `PYTHONHASHSEED` values and empty similarity caches (so the similarity search is genuinely recomputed) and compares the id files byte for byte with each other and with the saved splits.

- `load_splits` refuses to run if the settings saved in `splits/stats.json` differ from the code's settings. It does not detect edits to the function bodies (for example `normalize`). If you change how keys are built, bump `DEDUPE_SCHEME`.
- `stats.json` records the Python, `datasets` and `rapidfuzz` versions of the build, so a mismatch can be traced.
- To compare two runs or two machines, checksum the id files: `sha256sum <DATA_DIR>/splits/medmcqa.json <DATA_DIR>/splits/medqa.json` (or run the checksum cell). Equal checksums mean equal splits.
- Reference checksums (SHA-256) of the id files from the v6 build (seed 42, the pinned commits above, `DEDUPE_SCHEME = "two-tier-v1"`, similarity threshold 0.85, Python 3.13.18, `datasets` 4.8.5, `rapidfuzz` 3.14.6):

  | File | SHA-256 |
  |---|---|
  | `splits/medmcqa.json` | `7bc0abd2202114ea25c13b3d11aafde2fc9c7469db261042eaa4e5c1764ebaad` |
  | `splits/medqa.json` | `6e755d5bbd8c5c256a86e701a0557e1cb8d81e984f6abf7f6d6f57c550ffbc2c` |

  Every independent build with unchanged settings should print the same two values; this is how everyone confirms they have the identical dataset. If you change any setting, the values change and must be updated here.
- Python's random module does not formally guarantee identical shuffles across Python versions, and `rapidfuzz` could in principle change a similarity at the filter boundary in another version. If two checksums differ, compare the `environment` entries in the two `stats.json` files first.

## Freeze

Record the checksums above and do not rebuild once any tuning has started. A rebuild can change ids, which would invalidate every run that already used the old subsets. Earlier build versions (v3, v4, v5) produced different `medmcqa.json` files and must not be mixed with v6 results.

## Changing a setting

Anything in the settings block changes the splits. After changing one: rerun the build, rerun the tests, update the checksums here, and share the new `splits/` folder. Do not edit the prompt wording without bumping `PROMPT_VERSION` in `prompts.py`.
# found-model-eng-team-4
This is the repo for our small medical question answering model we are building as a part of the Foundation Model Engineering course

## Shared directory

All large files — data, model weights, and checkpoints — live in one shared folder on the team VM, **never in git**:

```
/shared/
  models/        # base model weights (HF_HOME points here so nobody re-downloads)
  data/          # raw downloads + processed splits
  checkpoints/   # training run outputs
```

### Rules for `/shared`
- **Don't delete or overwrite files you didn't create.** Ask the owner on Slack first.
- **Never re-upload MedAbstain anywhere.** Its license is CC-BY-NC; it stays in `/shared/data` only.
- **Point `HF_HOME` at `/shared/models`** so Hugging Face loads the shared copy of the base model (`Qwen/Qwen3-4B-Instruct-2507`) instead of downloading it again. Run this once on the VM:
  ```bash
  echo 'export HF_HOME=/shared/models' >> ~/.bashrc && source ~/.bashrc
  ```

### Using `paths.py`
[`src/utils/paths.py`](src/utils/paths.py) is the **only** place shared paths are defined. Never hard-code `/shared/...` anywhere else — import from `paths` instead:

```python
from src.utils import paths

paths.SHARED_DIR       # /shared
paths.MODELS_DIR       # /shared/models
paths.DATA_DIR         # /shared/data
paths.CHECKPOINTS_DIR  # /shared/checkpoints
```

The shared root comes from the `FME_SHARED_DIR` environment variable and defaults to `/shared`. On the VM you don't need to set anything. To run the code on another machine (e.g. your laptop), point it at a local folder with the same layout:

```bash
export FME_SHARED_DIR=~/fme_shared
```

If you need a new shared location (e.g. a new subfolder), add it to `paths.py` in a PR rather than building the path inline.

## Getting started

### 1. SSH into the VM
VM setup is tracked in #2. Once the VM is running:

```bash
gcloud compute ssh <vm-name> --zone <zone>
```

Check the VM is up and the GPU is visible with `nvidia-smi`.

### 2. Clone the repo into your home directory
Everyone works in their **own** clone in their home directory — never edit someone else's checkout.

```bash
cd ~
git clone git@github.com:tlelievre26/found-model-eng-team-4.git
cd found-model-eng-team-4
```

### 3. Create an environment and install the package
Python 3.11+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

`-e` installs the package in editable mode, so your code changes take effect without reinstalling. Dependencies are pinned to exact versions in [`pyproject.toml`](pyproject.toml) — change them only through a PR so everyone stays on the same stack.

### 4. Check it works

```bash
python -c "from src.utils import paths; print(paths.MODELS_DIR, paths.DATA_DIR, paths.CHECKPOINTS_DIR)"
# /shared/models /shared/data /shared/checkpoints
pytest
```

### 5. Set up your API keys
Keys live in your own `~/.bashrc` — **never commit them**. Run each command below and paste your key when prompted. The key isn't shown on screen or saved to your shell history.

```bash
read -rsp "W&B API key: " key && echo "export WANDB_API_KEY=$key" >> ~/.bashrc && unset key && echo
```

```bash
read -rsp "Hugging Face token: " key && echo "export HF_TOKEN=$key" >> ~/.bashrc && unset key && echo
```

Then reload your shell:

```bash
source ~/.bashrc
```

Get your keys from https://wandb.ai/authorize and https://huggingface.co/settings/tokens.

## Logging runs to W&B

Every training and eval run should log to the shared W&B project through [`src/utils/wandb_logging.py`](src/utils/wandb_logging.py). It records the full config, git commit, GPU type, wall-clock time, peak GPU memory, and estimated cost (hours × $0.71/hr for the L4 VM).

```python
from src.utils import wandb_logging as wl

with wl.start_run("lora-r16-lr2e-4", config):          # config: dict or dataclass
    tokens_seen = 0
    for step, batch in enumerate(loader, start=1):
        loss = train_step(batch)
        tokens_seen += int(batch["attention_mask"].sum())   # real tokens, not padding
        if step % 10 == 0:
            wl.log_train(step, tokens_seen, loss=loss, lr=scheduler.get_last_lr()[0])
        if step % 500 == 0:
            wl.log_eval({"accuracy": evaluate(model)})
# leaving the block logs the time / memory / cost summary, even if training crashes
```

For an eval-only script, use `wl.start_run(name, config, job_type="eval")` and call `wl.log_eval(...)`. You can also call `wl.start_run(...)` and `wl.finish_run()` yourself instead of using `with`.

What shows up in W&B:
- `train/*` and `eval/*`: your metrics plotted against optimizer step.
- `train_vs_tokens/*` and `eval_vs_tokens/*`: the same metrics plotted against tokens processed, so runs with different batch sizes or sequence lengths can be compared by compute.
- `perf/*`: GPU memory (allocated / reserved / peak), tokens per second, elapsed hours, running cost.
- Run summary: `gpu_name`, `wall_clock_hours`, `peak_gpu_mem_gb`, `est_cost_usd`, `final_step`, `total_tokens_seen`.

To check your setup, run the dummy script:

```bash
python scripts/wandb_smoke_test.py                       # logs a run to W&B
WANDB_MODE=disabled python scripts/wandb_smoke_test.py   # W&B off, prints the summary only
```

Use `WANDB_MODE=disabled` for quick local tests. Runs go to the `FME-project` project in the `tjl2152-columbia-university` team by default. To log somewhere else (e.g. a personal sandbox), set `WANDB_ENTITY` and/or `WANDB_PROJECT`, or pass `entity=` / `project=` to `start_run`.

## Team usage rules
- **Work on a branch.** Clone into your own home directory, create a branch for your work, and merge into `main` through a pull request.
- **Large files go in `/shared`, never in git.** This includes data, model weights, and checkpoints. MedAbstain's license (CC-BY-NC) means we must not re-upload it.
- **There is one GPU — share it.**
  - Check `nvidia-smi` before starting anything.
  - Run long jobs inside `tmux` so they survive disconnects (`tmux new -s <name>`, detach with `Ctrl-b d`, reattach with `tmux attach -t <name>`).
  - If the GPU is unavailable, please message in the Slack to coordinate.
- **Don't delete or overwrite files in `/shared` that you didn't create.**
- **API keys stay out of the repo.** Keep W&B and Hugging Face keys in your own environment or a git-ignored `.env`.
- **Stop the VM when you're done** and nobody else is using it — it costs money while running.

## Repo layout

```
src/              # importable package (import as `src`, e.g. `from src.utils import paths`)
  data/           # dataset loading + prompt formatting
  models/         # model wrapper
  training/       # training script(s)
  eval/           # evaluation + metrics
  utils/          # W&B logging, paths, misc helpers
configs/          # YAML config files for runs
scripts/          # entry points you run from the command line
tests/            # quick checks (pytest)
```
