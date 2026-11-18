# MedMCQA and MedQA: deduplicated train / val / test splits

One function returns fixed, leak-free splits for MedMCQA and MedQA in a common format, plus the single prompt format used by the model wrapper and training code.

```python
import os
os.environ["DATA_DIR"] = "/shared/data"   # set BEFORE importing medsplits
from medsplits import load_splits, build_messages

splits = load_splits("medmcqa")           # or "medqa" -> {"train": [...], "val": [...], "test": [...]}
messages = build_messages(splits["train"][0])                       # chat messages for the model
messages = build_messages(splits["train"][0], include_answer=True)  # adds the gold letter (training)
```

Example format (any number of options is supported):

```python
{"id": "medmcqa-123", "source": "medmcqa", "question": "...",
 "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "B"}
```

## Folder structure

### What is submitted (the code)

```
.
├── README.md
├── FME_MedMCQA_MedQA_splits_v3.ipynb   Runs end to end: sets DATA_DIR, writes the package and tests,
│                                       builds the splits, runs the tests, prints the README numbers
├── medsplits/                          The Python package (written by the notebook's %%writefile cells)
│   ├── __init__.py                     Public names: load_splits, build_all, build_messages, ...
│   ├── data.py                         Loading, common format, deduplication, splitting, load_splits
│   └── prompts.py                      The one prompt format (PROMPT_VERSION, build_messages)
└── tests/
    ├── test_data.py                    Needs the real data: leakage, duplicates, format, determinism, config guard
    ├── test_dedupe_logic.py            Offline: key functions and the two-tier dedupe on tiny examples
    └── test_prompts.py                 Offline: prompt text for 4 and 6 options, message structure
```

`__pycache__/` and `.pytest_cache/` folders appear after running; they are generated and not part of the project.

### Where the data lives (not part of the submission)

Data files are kept outside the code, in the folder named by the `DATA_DIR` environment variable (`/shared/data` by default; the Colab run used Google Drive `MyDrive/shared/data`). They are not committed to git.

```
<DATA_DIR>/
├── splits/
│   ├── medmcqa.json        Question ids of train / val / test (the source of truth for the splits)
│   ├── medqa.json          Same for MedQA
│   └── stats.json          Settings used, pinned dataset commits, raw and final sizes, removed counts
├── processed/
│   ├── medmcqa.jsonl       Every kept example in the common format, one JSON object per line
│   └── medqa.jsonl         (no split column: use load_splits, or look the ids up in splits/*.json)
└── hf_cache/               Raw Hugging Face downloads
```

`load_splits` reads the id lists in `splits/` and looks the records up in `processed/`. If these files do not exist yet, it builds them first (it downloads about 90 MB, takes a few minutes).

### What is in `medsplits/data.py`

| Name | Role |
|---|---|
| Settings block (`SEED`, `VAL_SIZE`, `SHORT_STEM_WORDS`, `DEDUPE_SCHEME`, `HF_REPOS`, `HF_REVISIONS`, `SPLIT_PRIORITY`, `DATASET_ORDER`) | Everything that shapes the splits, in one place |
| `current_config()` | Snapshot of those settings, saved in `stats.json` and checked by `load_splits` |
| `normalize()` | Lowercase, punctuation to spaces, collapse whitespace |
| `exact_key()`, `leak_key()` | Two definitions of "the same question" (see below) |
| `load_medmcqa_raw()`, `load_medqa_raw()` | Download (pinned commit) and convert to the common format |
| `dedupe_within()` | Remove repeats inside one list, keeping the first |
| `carve_val()` | Deterministic validation split from train |
| `global_dedupe()` | Cross-split and cross-dataset removal with priority test > val > train |
| `build_all()` | Runs the whole pipeline and writes `splits/`, `processed/`, `stats.json` |
| `load_splits()` | The entry point |

## How the splits are built

- **Sources (pinned):** MedMCQA `openlifescienceai/medmcqa` at commit `91c6572c454088bf71b679ad90aa8dffcd0d5868`; MedQA (English, 4 options) `GBaker/MedQA-USMLE-4-options` at commit `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`.
- **MedMCQA:** the official test set has no answers, so the official **validation** set is our **test** set. Our validation set (4000 rows) is carved out of train.
- **MedQA:** Hugging Face has only train and test. Our validation set (1000 rows) is carved out of train. The official test set is untouched (all 1273 rows), so MedAbstain (#4), which is built from MedQA test questions, cannot overlap MedQA train or val.
- **Seed 42.** Validation rows are chosen by sorting on id and shuffling with a seeded generator.
- **Deduplication.** Questions are compared after lowercasing, replacing punctuation with spaces and collapsing whitespace. When a question appears in more than one place, one copy is kept with priority test > val > train (MedQA before MedMCQA on ties).
  - Inside train, and between the two train sets: only true copies are removed (`exact_key`: same question and same options, option order ignored).
  - Val and test, and train versus val/test: strict matching (`leak_key`): stems of 10 or more words match on the question alone, shorter stems must also have the same options. No val or test question appears in any train set.

## Results

| Dataset | Raw train | Raw test | Final train | Val | Test |
|---|---|---|---|---|---|
| MedMCQA | 182822 | 4183 (HF validation) | 160536 | 3998 | 4168 |
| MedQA | 10178 | 1273 | 9178 | 1000 | 1273 |

MedMCQA val is 3998, not 4000, because 2 carved val rows matched another eval question and were dropped.

| Dataset | Train copies removed | Test copies removed | Train rows matching a val/test question | Val rows matching another eval question | True copies across the two train sets |
|---|---|---|---|---|---|
| MedMCQA | 18106 | 15 | 180 | 2 | 0 |
| MedQA | 0 | 0 | 0 | 0 | 0 |

Total removed: 18303 (all MedMCQA). Removed copies whose correct answer text differed from the kept copy (label noise in the source): MedMCQA train 400, MedMCQA test 2.

## Running it

1. Open the notebook in Colab and run all cells in order. Cell 1 sets `DATA_DIR`; edit that one line for your machine or shared folder.
2. Outside Colab: `pip install datasets pytest`, set `DATA_DIR`, then from the folder that contains `medsplits/` run `python -m pytest -q tests` (18 tests, about 75 seconds; the first run downloads the data).
3. Set `DATA_DIR` **before** importing `medsplits`; it is read once at import time.

## Reproducibility

The splits are a function of: the two pinned dataset commits, the settings block in `data.py`, and the code. With those unchanged, building again gives identical splits (`test_deterministic` builds twice and compares the id files byte for byte).

- `load_splits` refuses to run if the settings saved in `splits/stats.json` differ from the code's settings. It does not detect edits to the function bodies (for example `normalize`). If you change how keys are built, bump `DEDUPE_SCHEME`.
- To compare two runs or two machines, checksum the id files: `sha256sum <DATA_DIR>/splits/medmcqa.json <DATA_DIR>/splits/medqa.json`. Equal checksums mean equal splits.
- Reference checksums (SHA-256) of the id files from the v3 build (seed 42, the pinned commits above, `DEDUPE_SCHEME = "two-tier-v1"`):

  | File | SHA-256 |
  |---|---|
  | `splits/medmcqa.json` | `276f02ced18882ed75e9826abec5ce444551112385669fccee133f4400f5837e` |
  | `splits/medqa.json` | `6e755d5bbd8c5c256a86e701a0557e1cb8d81e984f6abf7f6d6f57c550ffbc2c` |

  A rebuild with unchanged settings should print the same two values. If you change any setting, the values change and must be updated here.
- Python's random module does not formally guarantee identical shuffles across Python versions. In practice they match, but if two checksums differ, compare Python versions first.

## Changing a setting

Anything in the settings block changes the splits. After changing one: rerun the build, rerun the tests, and share the new `splits/` folder. Do not edit the prompt wording without bumping `PROMPT_VERSION` in `prompts.py`.
