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

The default location is `/shared/data/medabstain/`. `DATA_DIR` sets the parent data
directory; `FME_SHARED_DIR` sets the shared root when `DATA_DIR` is unset.
These settings are read when `src.utils.paths` is imported.

```bash
export DATA_DIR=/path/to/data
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
`is_answerable`, and `metadata`. Metadata retains the complete upstream record.
The abstention letter is identified by option text, ignoring case and whitespace;
it is `None` for NA/NAP. IDs repeat across variants.

## Evaluation

The upstream scorer uses the original answer for accuracy and conformal coverage
in all four variants. A/AP abstention rate is a separate metric. Accordingly,
`answer`, `original_answer`, and `evaluation_target` retain the source label.
Original examples have `is_answerable=True`; perturbed examples have `None`
because individual answerability is not annotated.

## Tests

```bash
python -m pip install pytest==9.2.0
MEDABSTAIN_DATA_DIR=/path/to/medabstain python -m pytest -q -rs tests
```

Local result: **49 passed, 1 skipped** with the published MedAbstain files.
The skipped overlap test requires a MedQA export through `MEDQA_SPLITS_JSON`.

```bash
python -m scripts.verify_medabstain --data-dir /path/to/medabstain \
  --medqa-splits /path/to/medqa_train_val.json
```

The export format is `{"train": [records], "val": [records]}`. Matching lowercases
question text, replaces punctuation with spaces, and collapses whitespace.
Both displayed and original questions are compared; choices are ignored.
Generic short-question matches may need review. Existing splits are unchanged.

## Known overlap

Against the reproduced Issue #3 splits, each MedAbstain test variant matches
**903 MedQA train / 104 validation** records. Each available validation variant
matches **88 train / 12 validation** records. Original test/NA questions match
zero official MedQA test records. Counts repeat across variants.

The reproduction used Issue #3 commit
`696eadb1a1ddc1f44c2e312f3d0367cf887ceba9`. Its MedQA split-ID SHA-256 matches the
published reference: `6e755d5bbd8c5c256a86e701a0557e1cb8d81e984f6abf7f6d6f57c550ffbc2c`.
The overlap check fails for this export. These records are therefore not an
independent evaluation set after training on the overlapping MedQA records.
Shared-VM deployment and a resolution of this overlap are still pending.
