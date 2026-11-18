# Issue #5: Qwen model wrapper

## What this work does

This adds a wrapper for `Qwen/Qwen3-4B-Instruct-2507`. It loads the base model
in 4-bit precision and scores multiple-choice questions using the next-token
scores for the answer letters. It returns a probability for each available
letter and the predicted letter, with support for single questions and batches,
including questions with six options.

CPU tests: **10 passed, 1 skipped**. The skipped test needs the real Qwen
tokenizer. GPU loading, memory usage, medical-data accuracy, and integration
with the shared prompt/data loaders still need to be checked on the VM.
The LoRA interface is present but not implemented yet.

## Files

| File | Purpose |
|---|---|
| `src/models/qwen.py` | `QwenWrapper`: local 4-bit model loading, single-question scoring, batch scoring, answer-letter token checks, and the future LoRA adapter hook. |
| `tests/test_qwen.py` | CPU tests using a fake model/tokenizer to check probability normalization, four/six options, batching, input checks, and the adapter placeholder. No model download is needed. |
| `tests/test_qwen_tokenizer.py` | Optional test with the real local tokenizer. Checks answer-letter boundaries in a rendered chat prompt and prints bare versus space-prefixed token IDs. |
| `scripts/validate_qwen.py` | VM validation on labeled MedMCQA examples and a real six-option MedAbstain example. Prints accuracy and peak GPU memory, and checks that probabilities sum to one. |
| `issue_5_qwen.md` | Scope, file descriptions, dependencies, and usage instructions. |

## Dependencies and remaining integration

Runtime packages: `torch`, `transformers` with Qwen3 support, `accelerate`, and
`bitsandbytes`. Tests also need `pytest`. Use the team's pinned versions from
Issue #1 when available. Real 4-bit loading in this implementation requires a
CUDA GPU and local model/tokenizer files; it does not download missing weights.

| Dependency | Needed from the team | What may need changing after it is available |
|---|---|---|
| #1: repo setup (`tlelievre26`) | Shared package structure, pinned dependencies, and `src.utils.paths.MODELS_DIR`. | Connect the default model path to the shared paths helper and verify imports under the merged package setup. |
| #2: GPU VM (`victorianian610`) | L4 access, model/tokenizer files, and the exact model directory. | Override `model_path` if needed, then test 4-bit loading and record peak GPU memory. |
| #3: data and prompt (`woohyeonher`) | The actual shared prompt function's import path, common-format examples, and fixed MedMCQA split IDs. | Pass that function as `prompt_fn`; update the validation data input to use the final loader or its JSONL export. Rerun token checks with the real shared prompt and adjust token handling if needed. |
| #4: MedAbstain (`Jzzzc-zj`) | Real six-option examples converted to the common format. | Connect the loader or export its examples to JSONL; use `options` rather than raw `choices`. Check six-option scoring with real data. |
| #11: validation subsets | Agreed `hp_dev`, `calib`, `thresh`, and `test` split IDs. | Confirm which held-out subset to use for the ~100-question check and keep final test data separate from tuning. |

These inputs have not been integrated or verified in this branch. That does
not mean the owners have not started their work. Once they are available,
complete the shared-prompt/tokenizer checks, ~100-question accuracy check, and
six-option GPU check before marking #5 fully complete.

## How others can use this code

### Input and shared prompt

Run from the repository root. Each example uses this common format:

```python
example = {
    "id": "example-1",
    "question": "What is 2 + 2?",
    "options": {"A": "3", "B": "4", "C": "5", "D": "6"},
    "answer": "B",
}
```

Scoring needs `question` and `options`; validation also needs `answer`.
Option keys must be single uppercase letters. The required `prompt_fn` takes
one example and returns chat messages such as
`[{"role": "user", "content": "question + lettered options + answer instruction"}]`.
Use the shared formatter from #3 so evaluation and training use the same prompt.
The wrapper applies the tokenizer's chat template itself.

### Single-question and batch scoring

```python
from src.models.qwen import QwenWrapper

# shared_prompt_fn is the function from #3; its import path is still to be confirmed.
wrapper = QwenWrapper(
    model_path="/shared/models/Qwen3-4B-Instruct-2507",
    prompt_fn=shared_prompt_fn,
)

result = wrapper.score(example)
probabilities = result["probabilities"]  # {"A": ..., "B": ..., "C": ..., "D": ...}
predicted_letter = result["prediction"]

results = wrapper.score_batch([example], batch_size=4)
```

Batch results are returned in input order. Probabilities sum to approximately
one over the available answer letters; they are not calibrated confidence
scores. The caller keeps the example IDs and correct answers alongside results.

- **#12 evaluation** directly uses `score_batch` to compute accuracy, NLL, and
  ECE, and saves per-question predictions for later calibration/abstention work.
- **#7 training** can reuse the model-loading setup and must implement the
  adapter hook so saved LoRA checkpoints can be loaded for evaluation.
  `load_adapter(path)` and `adapter_path` currently raise `NotImplementedError`.
  Scoring methods use inference mode; training needs its own loss and training setup.

Keep the class/method names and result keys consistent with these callers.

### Tests and VM validation

```sh
python3 -m pytest -q tests
QWEN_TOKENIZER_PATH=/shared/models/Qwen3-4B-Instruct-2507 python3 -m pytest -s tests/test_qwen_tokenizer.py
```

For VM validation, replace the uppercase placeholders below with the real
shared formatter and common-format JSONL paths. Each JSONL line is one example.

```sh
python3 scripts/validate_qwen.py \
  --prompt-function YOUR_SHARED_MODULE:YOUR_SHARED_FUNCTION \
  --data /shared/data/YOUR_MEDMCQA_VALIDATION.jsonl \
  --six-option-data /shared/data/YOUR_MEDABSTAIN.jsonl \
  --limit 100 --batch-size 4
```

The script prints the sample count, accuracy, GPU type, peak memory, and the
six-option result. It fails if probabilities are not normalized or accuracy is
at or below 25%. For acceptance, check that accuracy is clearly above that
baseline and memory fits the L4; record the measured results after the VM run.
