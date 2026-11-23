# MedAbstain

Source: [MedAbstain](https://github.com/sravanthi6m/MedAbstain), accompanying
[Machcha et al., EACL 2026](https://aclanthology.org/2026.eacl-long.291/).
Upstream commit: `091e5c22111fffeb51c0c2e69b65d0a21a1e4164`.
License: **CC-BY-NC 4.0**; noncommercial use with attribution. The download includes
the upstream LICENSE and a `provenance.json` file with the commit and file checksums.
Data files are stored outside the repository.

## Usage

```bash
python -m scripts.download_medabstain
python -m scripts.verify_medabstain
```

The location is `paths.MEDABSTAIN_DIR` from `src/utils/paths.py`: `$FME_SHARED_DIR/data/medabstain/`,
which is `/shared/data/medabstain/` on the team VM. On other machines (e.g. Insomnia), set
`FME_SHARED_DIR` before running anything; `--data-dir` / `data_dir=` override it for one call.

```bash
export FME_SHARED_DIR=/path/to/shared
python -m scripts.download_medabstain
```

```python
from src.data.medabstain import load_medabstain

examples = load_medabstain("AP", split="test", data_dir="/path/to/medabstain")
validation = load_medabstain("A", split="val", data_dir="/path/to/medabstain")
```

Signature: `load_medabstain(variant, split="test", data_dir=None) -> list[dict]`.

| Variant | Test file | Questions | Validation file / questions |
|---|---|---:|---|
| NA | medqa_1_test_noabst.json | 1,007 | val_100_medqa_1_noabst.json / 100 |
| A | medqa_1_test_randabst.json | 1,007 | val_100_medqa_1_randabst.json / 100 |
| NAP | perturbed_medqa_1_test_noabst.json | 1,007 | Not available |
| AP | perturbed_medqa_1_test_randabst.json | 1,007 | Not available |

The downloader also includes the four upstream few-shot pool files.
The loader supports the test and validation files listed above.

Records contain `id`, `source`, `question`, `options`, and `answer`, plus
`original_answer`, `evaluation_target`, `abstain_option`, `is_perturbed`,
`is_answerable`, and `metadata`. `source` is always `"medabstain"` (upstream says `MEDQA_1`).
Metadata retains the complete upstream record.
The abstention letter is identified by option text, ignoring case and whitespace;
it is `None` for NA/NAP. IDs are `medabstain-<split>-<upstream id>`: the same question has the
same ID in all four variants, and test and val IDs never collide (upstream reuses 54 integer
ids between them for different questions). The upstream record is kept in `metadata.raw`.

## Evaluation

The upstream scorer uses the original answer for accuracy and conformal coverage
in all four variants. A/AP abstention rate is a separate metric. Accordingly,
`answer`, `original_answer`, and `evaluation_target` retain the source label.
Original examples have `is_answerable=True`; perturbed examples have `None`
because individual answerability is not annotated.

Note: a few perturbed questions are identical to the original (NAP ids 230, 236, 544, 613,
998; AP ids 450, 544, 613, 998), so abstaining is not the expected behaviour for them. NAP
and AP were perturbed separately (375 questions have different text), so compare them with care.

## Tests

```bash
python -m pip install pytest==9.2.0
FME_SHARED_DIR=/path/to/shared python -m pytest -q -rs tests
```

The real-data tests skip when the MedAbstain files are missing. The MedQA overlap test
uses `load_splits("medqa")` (skipped if the MedQA splits are not built) and checks that no
MedAbstain test question is in MedQA train/val.

```bash
python -m scripts.verify_medabstain            # checksums vs provenance.json, counts, MedQA overlap (fails on test files only)
python -m scripts.verify_medabstain --skip-medqa-overlap   # files only
```

The overlap check compares against the MedQA `train` and `val` splits from
`src.data.medqa_datasets.load_splits`. Matching lowercases
question text, replaces punctuation with spaces, and collapses whitespace.
Both displayed and original questions are compared; choices are ignored.
Generic short-question matches may need review.

## Known overlap

All 1,007 MedAbstain test questions come from MedQA **train** (`GBaker/MedQA-USMLE-4-options`); none
is in the official MedQA test set. The splits built before `DEDUPE_SCHEME = "two-tier-v3"` (MedQA SHA-256
`6e755d5bbd8c5c256a86e701a0557e1cb8d81e984f6abf7f6d6f57c550ffbc2c`) put **903 of them in MedQA train and
104 in validation**. Since `two-tier-v3`, `build_all()` removes every MedAbstain test question from all
MedQA and MedMCQA splits before carving (see `src/data/medqa_datasets/README.md`), so the test files have
zero overlap and MedAbstain stays independent of a model trained on MedQA. The `val_100` and few-shot files
are not excluded and still overlap MedQA train/val.
