# Issue #5: Qwen model wrapper

## What this does

Load Qwen3-4B-Instruct-2507 locally in 4-bit NF4 with double quantization and
bf16 compute, shared with #7 training. Score answer letters in one forward pass
per batch and return normalized option probabilities plus the predicted letter.
Four-, five-, and six-option examples use the same interface.

## Files

| File | Purpose |
|---|---|
| `src/models/qwen.py` | Shared `load_base_model`, single/batch scoring, shared prompt, local PEFT checkpoint loading, and training-mode restoration. |
| `scripts/validate_qwen.py` | CUDA check using 100 MedMCQA `hp_dev` examples and a real six-option MedAbstain example; prints accuracy and peak GPU memory. Independent of #12. |
| `tests/test_qwen.py` | CPU tests for normalized scores, batching, invalid inputs and token boundaries. |
| `tests/test_qwen_integration.py` | Shared prompt, no answer leakage, base-loader contract, adapter initialization and training-mode restoration checks. |
| `tests/test_qwen_peft.py` | Small random Qwen3 batch/single comparison and real PEFT save/load round trip on CPU. |
| `tests/test_qwen_tokenizer.py` | Optional real local tokenizer checks with the shared prompt. |

## Dependencies and remaining validation

Runtime: the team's pinned `torch`, `transformers`, `accelerate`, `bitsandbytes`,
`peft`, and `wandb`; tests also use `pytest`. Project pins are unchanged.

| Team work | Interface used / remaining integration |
|---|---|
| #1 shared paths | `src.utils.paths.MODELS_DIR`. Set `FME_SHARED_DIR` before starting Python. Override `model_path` if the model folder has another name. |
| #2 GPU setup | CUDA with bf16 support and local model/tokenizer files. GPU loading and memory validation still need the real machine. |
| #3 shared prompt/data | Default `build_messages(example)` excludes the gold answer. `load_splits("medmcqa")["hp_dev"]` selects the fixed subset, rather than reading the mixed processed JSONL directly. |
| #4 MedAbstain | `load_medabstain("A", data_dir=...)` supplies a six-option smoke example. This verifies scoring, not an abstention policy. |
| #11 data subsets | Use frozen `hp_dev` for tuning, `calib` for calibration, `thresh` for thresholds, and `test` for final reporting. Use the team-approved data build. |

The real Qwen tokenizer passes the shared-prompt A–F boundary check. CUDA
4-bit loading, ~100-question medical accuracy clearly above 25%, six-option
real-data scoring, and real trained checkpoint evaluation remain pending. CPU tests do not establish these results.

## How others use it

```python
from src.models.qwen import QwenWrapper, load_base_model

wrapper = QwenWrapper()  # MODELS_DIR / "Qwen3-4B-Instruct-2507"
# example: {"id": ..., "question": ..., "options": {"A": ..., "B": ...}, "answer": ...}
result = wrapper.score(example)
results = wrapper.score_batch(examples, batch_size=4)
# {"probabilities": {"A": ..., "B": ...}, "prediction": "A"}
```

Results preserve input order. Option keys are uppercase single letters. Scores
sum to approximately one over the allowed letters, not the full vocabulary.
Ties select the first letter alphabetically. They are not calibrated confidence.
The wrapper applies the chat template once and rejects assistant answer turns.

**#7 training:** `model, tokenizer = load_base_model(model_path, device_map=...)`
is the same entry point imported by the current training branch. Training owns
LoRA preparation and loss computation. For evaluation of the current in-memory
model, use `QwenWrapper(model=training_model, tokenizer=tokenizer)`; scoring
restores its original training flag even on errors. Final-token-only Qwen3
logits avoid storing logits for every prompt position on the L4.

**#12 evaluation:** call `score_batch` and consume `probabilities` / `prediction`.
Keep IDs and correct answers with the caller. No W&B run is started by scoring.

**Saved adapters:** `QwenWrapper(adapter_path="/path/to/checkpoint")` loads local
PEFT weights after base initialization, for evaluation. Use a fresh wrapper for
each saved adapter. Do not pass `adapter_path` for an already attached training
model; injected PEFT models can be scored directly.

```sh
python -m pytest -q
QWEN_TOKENIZER_PATH=/path/to/local/model python -m pytest -s tests/test_qwen_tokenizer.py
WANDB_MODE=offline python scripts/validate_qwen.py --limit 100 --batch-size 4
```

The validation script uses the team's existing W&B helper. On non-L4 hardware,
pass the appropriate `--gpu-hourly-usd`. Real weights and data stay outside git.

## Verification on this Mac

Full local suite: **91 passed, 19 skipped**. Real shared-prompt Qwen tokenizer,
tiny random Qwen3 mixed-length batching, and actual PEFT save/load passed.
Skipped tests require real dataset builds. Local test environment: Python 3.11,
torch 2.6.0, transformers 4.57.6, peft 0.18.1, accelerate 1.4.0. This is a
compatibility check, not validation of the team's pinned CUDA stack.
