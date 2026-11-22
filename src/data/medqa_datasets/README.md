# MedMCQA and MedQA: deduplicated splits with three MedMCQA evaluation subsets

One function returns fixed, leak-free splits for MedMCQA and MedQA in a common format, plus the single prompt format used by the model wrapper and training code.

```python
from src.data.medqa_datasets import load_splits, build_messages   # build the data once first (see Running it)

medmcqa = load_splits("medmcqa")   # {"train", "hp_dev", "calib", "thresh", "test"}
medqa = load_splits("medqa")       # {"train", "val", "test"}
messages = build_messages(medmcqa["train"][0])                       # chat messages for the model
messages = build_messages(medmcqa["train"][0], include_answer=True)  # adds the gold letter (training)
```

**Use the `.py` code** (`load_splits`, `build_messages`) to work with the datasets. The notebook `splits_build_and_analysis.ipynb` only exists for analysis and for running the build on Colab; it imports the same code from the repo and contains no pipeline code of its own.

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

The code is in this folder and the tests are in the repo's `tests/` folder. **No dataset files are committed: everyone builds their own copy** (see Running it), and the checksums below prove that the copies are identical.

```
src/data/medqa_datasets/
├── README.md
├── __init__.py                       Public names: load_splits, build_all, build_messages, ...
├── data.py                           Loading, common format, deduplication, subsets, similarity filter, load_splits
├── prompts.py                        The one prompt format (PROMPT_VERSION, build_messages)
└── splits_build_and_analysis.ipynb   Colab: clones the repo, builds the splits, runs the tests, analysis cells,
                                      prints the README numbers
tests/
├── test_data.py                      real_data (--run-real-data): leakage, no shared questions, similarity filter,
│                                     determinism in fresh processes, format, config guard
├── test_subsets.py                   Offline: subset carving, similarity filter, guard on subsets
├── test_dedupe_logic.py              Offline: key functions, the two-tier dedupe and conflict dropping on tiny examples,
│                                     load_splits refusing to build
└── test_prompts.py                   Offline: prompt text for 4 and 6 options, message structure
```

### Where the data lives (not part of the submission)

Data files are kept outside the code, in `DATA_DIR` = `$FME_SHARED_DIR/data` (`/shared/data` by default; the Colab run used Google Drive `MyDrive/shared/data`). All of these paths come from `src/utils/paths.py` (`DATA_DIR`, `SPLITS_DIR`, `PROCESSED_DIR`, `HF_CACHE_DIR`, `CACHE_DIR`); set the `FME_SHARED_DIR` environment variable to use another root, for example on Insomnia. `build_all(out_dir=...)` and `load_splits(..., data_dir=...)` accept another folder with the same `splits/` and `processed/` layout. The data files are not committed to git; each person produces them with `build_all()` (or by running the notebook).

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

`load_splits` reads the id lists in `splits/` and looks the records up in `processed/`. It never builds them: if they are missing it raises `FileNotFoundError` with the build command (see Running it).

### What is in `data.py`

| Name | Role |
|---|---|
| Settings block (`SEED`, `SPLIT_SIZES`, `SIMILARITY_FILTER`, `SHORT_STEM_WORDS`, `DEDUPE_SCHEME`, `HF_REPOS`, `HF_REVISIONS`, `SPLIT_PRIORITY`, `DATASET_ORDER`) | Everything that shapes the splits, in one place |
| `current_config()` | Snapshot of those settings, saved in `stats.json` and checked by `load_splits` |
| `normalize()` | Lowercase, punctuation to spaces, collapse whitespace |
| `exact_key()`, `leak_key()` | Two definitions of "the same question" (see below) |
| `load_medmcqa_raw()`, `load_medqa_raw()` | Download (pinned commit) and convert to the common format |
| `dedupe_within()` | Remove repeats inside one list, keeping the first (optionally dropping every copy of a question whose copies disagree on the answer) |
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
  - Inside each train set, if the true copies of a question disagree on the answer text (label noise), **all** copies are dropped instead of keeping the first. `stats.json` counts the extra rows dropped this way (one per such question; the other copies are already in `within`) as `<dataset>/train/conflict_dropped`. Deduplication of `test` (`leak_key`) still keeps the first copy.
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

1. **Build the data once, explicitly**, before anything calls `load_splits` (it does not build on its own). From the repo root:

   ```bash
   python -c "from src.data.medqa_datasets import build_all; build_all()"
   ```

   Output goes to `$FME_SHARED_DIR/data` (`/shared/data` by default, on the team VM); on other machines such as Insomnia, set `FME_SHARED_DIR` first. Then check `sha256sum $FME_SHARED_DIR/data/splits/*.json` against the checksums in Reproducibility. The team VM's `/shared/data` can be built this way, or get a copy of `splits/` and `processed/` from a verified build.
2. On Colab: open `splits_build_and_analysis.ipynb` and run all cells in order. The setup cell clones the repo (set `BRANCH` to run another branch) and sets `FME_SHARED_DIR` (a Google Drive path; edit that one line for your own Drive or folder). Running the notebook is all that is needed to obtain the dataset: it downloads both datasets at the pinned commits and builds the splits. The test cell is optional for just obtaining the data.
3. Check your copy: run the checksum cell and compare with the reference checksums in the Reproducibility section.
4. Outside Colab: install the repo (`pip install -e ".[dev]"`), set `FME_SHARED_DIR` if the data is not under `/shared`, then from the repo root run `python -m pytest -q`. Plain `pytest` runs only the offline tests: the tests in `tests/test_data.py` are marked `real_data` and skipped by default, because they need the built data (step 1) and network access and take minutes. To run them too, add `--run-real-data` (about 8 minutes on Colab because two fresh builds recompute the similarity search).

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
