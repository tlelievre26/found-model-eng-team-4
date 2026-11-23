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

## Tests

```bash
python -m pip install pytest==9.2.0
FME_SHARED_DIR=/path/to/shared python -m pytest -q -rs tests
```

The real-data tests skip when the MedAbstain files are missing. The MedQA overlap test
uses `load_splits("medqa")` (skipped if the MedQA splits are not built) and is marked
`xfail` until the overlap below is resolved.

```bash
python -m scripts.verify_medabstain            # add --skip-medqa-overlap to check the files only
```

The overlap check compares against the MedQA `train` and `val` splits from
`src.data.medqa_datasets.load_splits`. Matching lowercases
question text, replaces punctuation with spaces, and collapses whitespace.
Both displayed and original questions are compared; choices are ignored.
Generic short-question matches may need review. Existing splits are unchanged.

## Known overlap

Against the reproduced Issue #3 splits, each MedAbstain test variant matches
**903 MedQA train / 104 validation** records. Each available validation variant
matches **88 train / 12 validation** records. Original test/NA questions match
zero official MedQA test records. Counts repeat across variants.

These counts are for the MedQA splits with SHA-256
`6e755d5bbd8c5c256a86e701a0557e1cb8d81e984f6abf7f6d6f57c550ffbc2c`, so the overlap check
currently fails. MedAbstain questions are therefore not an independent evaluation set for
a model trained on MedQA train/val. The planned unify-datasets PR will exclude MedAbstain
test questions from MedQA train/val when the splits are built.
