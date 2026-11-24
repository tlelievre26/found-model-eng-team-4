# Evaluation harness (issue #12)

## What this does

Score a shared data split through the Qwen wrapper, calculate accuracy, natural-log
NLL and ECE, and optionally save per-question results for later calibration and
threshold selection. The standalone command logs through the shared W&B helper.

## Files

| File | Purpose |
|---|---|
| `src/eval/metrics.py` | Pure `compute_metrics(probabilities, answers, n_bins=15)` function for variable-option predictions. |
| `src/eval/evaluate.py` | `evaluate(model, examples, batch_size=16, ...)`: batch scoring, metric aggregation and optional JSONL/metadata output. |
| `src/eval/__init__.py` | Exports `evaluate` and `compute_metrics`. |
| `scripts/evaluate.py` | Select dataset, split, base model or adapter, and output; start/finish a standalone W&B evaluation run. |
| `src/utils/paths.py` | Adds `EVAL_DIR` under the configurable shared root. |
| `tests/test_eval.py` | Hand-worked metrics, ECE edges, output/reload checks, malformed output handling, and real shared-loader/W&B integration with synthetic examples. |

## Dependencies and exact interfaces

- Requires the updated #5 wrapper (PR #14), plus the team's existing pinned environment.
- Woohyeon's `build_messages(ex, include_answer=False)` is called by the wrapper;
  the evaluator does not create a second prompt or include gold answers in inputs.
- Woohyeon's `load_splits(name, data_dir=None, check_config=True)` supplies
  `hp_dev`, `calib`, `thresh`, `test`, `train` for MedMCQA and `val`, `test`,
  `train` for MedQA. Keep configuration checking enabled and use the approved,
  frozen data build. The CLI does not rebuild or resplit data.
- Uses `paths.DATA_DIR` / `paths.EVAL_DIR`; set `FME_SHARED_DIR` before Python.
- Uses the existing `wandb_logging.start_run`, `log_eval`, and context-manager
  cleanup. Reuses `DEFAULT_GPU_HOURLY_USD`; pass the agreed rate on non-L4 GPUs.
- #7's current training branch imports the temporary `src.eval.stub.evaluate`.
  **When integrating #12, replace that line with `from src.eval import evaluate`.**
  No other call-site change is required for its existing `run_eval()` function.

## How others use it

```python
from src.eval import evaluate
from src.models.qwen import QwenWrapper
from src.utils import wandb_logging as wl

# Within an already active training run:
wrapper = QwenWrapper(model=training_model, tokenizer=tokenizer)
metrics = evaluate(wrapper, hp_dev_examples, batch_size=16)
wl.log_eval(metrics)  # training keeps ownership of its W&B run
# {"accuracy": ..., "nll": ..., "ece": ..., "n": ...}
```

Use batch size 16 (the default, and the training config's `eval_batch_size`) for every
reported number: padding changes the 4-bit/bf16 arithmetic, so scores shift slightly
with batch size (on 100 hp_dev questions, batch 16 vs 1 differed by 0.013 per letter
on average, with identical predictions). Evaluation is forward-only, so 16 fits easily
(about 4 GiB peak on an A6000).

Training ranks sweep trials by `nll` (lower is better). The wrapper restores
training mode after scoring. Evaluation itself starts no W&B run. Optional
arguments are `n_bins=15`, `output_path`, and `metadata`:

```python
metrics = evaluate(wrapper, examples, batch_size=16,
                   output_path=output_file,
                   metadata={"dataset": "medmcqa", "split": "hp_dev"})
```

JSONL records contain `id`, `source`, `probabilities`, `prediction`, `answer`,
`confidence`, and `correct`; they omit question and option text. A `.meta.json`
sidecar records metadata, bin count, and sample count. Existing files are never
overwritten. The CLI records prompt version and checksums of split IDs and build
statistics. Keep these local outputs in shared storage, not git/W&B artifacts.

ECE uses 15 equal-width right-closed bins by default, matching the current
training stub: `(i/B, (i+1)/B]`, with confidence 0 in the first bin. Confidence
1 is included. Ties choose the first letter alphabetically. The gold
probability is clipped at 1e-12 before the log (`NLL_MIN_PROB`), so a probability
that underflows to 0 adds at most about 27.6 to the sum instead of making the mean
NLL infinite, which would break the sweep's ranking.

## Commands

```sh
python scripts/evaluate.py --dataset medmcqa --split hp_dev
python scripts/evaluate.py --dataset medmcqa --split test
python scripts/evaluate.py --dataset medqa --split test
python scripts/evaluate.py --dataset medmcqa --split hp_dev --adapter /path/to/checkpoint
```

Use `--limit 10` for a smoke test; omit it for reported baselines. Offline logging
is supported with `WANDB_MODE=offline`; tests use `WANDB_MODE=disabled`.
Never select training settings on `test`. Calibration and thresholds use
`calib` and `thresh`, separately.

## Verification and remaining work

Local suite on Mac CPU: **103 passed, 19 skipped** (real dataset checks skipped).
Tests include actual shared-loader reads from synthetic on-disk splits, a
standalone CLI W&B run in disabled mode, and reuse of an existing training run.
Real Qwen tokenizer, tiny Qwen3 batching and PEFT save/load also pass.
Local stack: Python 3.11, torch 2.6.0, transformers 4.57.6, peft 0.18.1,
accelerate 1.4.0. Team dependency pins are unchanged; pinned CUDA validation is pending.

In a temporary copy of the actual `training-pipeline` branch, replacing the stub
import with this evaluator and overlaying the updated wrapper passed all **11
training tests**, including checkpointing and exact resume. The teammate's remote
training branch was not modified.

Still pending: pretrained 4B GPU runs, full baseline results for MedMCQA hp_dev/test
and MedQA test, real checkpoint evaluation on team hardware, and team-agreed
numerical accuracy/ECE/selective-accuracy targets before fine-tuning. Merging code
does not complete these real-run requirements or close #12.
