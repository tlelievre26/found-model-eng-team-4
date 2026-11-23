# W&B metrics logged by training and the sweep

Everything `scripts/train.py` and `scripts/run_sweep.py` send to W&B (entity `tjl2152-columbia-university`,
project `FME-project`), through `src/utils/wandb_logging.py`. Question text is never logged, only metrics and
the run config.

**Axes.** `train/*`, `eval/*` and `perf/*` are plotted against `train/step` (optimizer steps). `train_vs_tokens/*`
and `eval_vs_tokens/*` hold the same values plotted against `train/tokens_seen`, the cumulative count of
non-padding input tokens (the project's compute measure). Use the `_vs_tokens` panels to compare runs by compute.

**When.** Training metrics are logged every `log_every_steps` optimizer steps (10 by default, and at the last
step). Each value is an average over the steps since the previous log point, unless stated otherwise.

## Training (`train/<name>` and `train_vs_tokens/<name>`)

| Name | Meaning | How it is computed | What to look for |
|---|---|---|---|
| `loss` | Training loss | Cross-entropy of the answer-letter token (full vocabulary), averaged over the questions in each optimizer step, then over the steps in the interval | Should fall quickly and then flatten. A rise after warmup means the learning rate is too high |
| `lr` | Learning rate | The rate used for the last step in the interval | A linear ramp over the first 100 steps, then flat |
| `grad_norm` | Gradient norm **before** clipping | Global L2 norm of the LoRA gradients returned by `clip_grad_norm_`, averaged | Spikes point to unstable batches. A steady rise can come before divergence |
| `clip_frac` | Fraction of steps clipped | Steps with `grad_norm > max_grad_norm` (1.0) ÷ steps in the interval | Mostly 0 after the first steps. Near 1 all the time means the rate is too high, or the clip too low |
| `answer_acc` | Accuracy on the training batches | argmax over the question's option letters equals the gold letter, ÷ questions in the interval | Training-set accuracy, measured before each update. Compare with `eval/accuracy`: a growing gap means overfitting, from the second epoch on |
| `p_correct` | Mean probability of the correct letter | Softmax over the **full vocabulary** at the answer position, the gold letter's probability, averaged | Rises with training. It is `exp(-loss)`-like but averaged in probability space |
| `letter_mass` | Probability mass on the option letters | Sum of full-vocabulary probabilities of the question's option letters (A–D for MedMCQA), averaged | Should stay close to 1 (the base model already answers with a letter). **A falling letter mass means the model is drifting off answering with a letter**, which the letter-only scores used for evaluation would hide |
| `pred_frac_A` … `pred_frac_D` | How often each letter is predicted | Share of questions in the interval whose argmax option is that letter | Roughly matches the gold-letter distribution. One letter taking over means a positional bias or collapse |
| `pad_frac` | Padding in each micro-batch | `1 − real tokens / (batch × padded length)`, averaged over micro-batches | About 0.01 with length grouping. A large value means length grouping isn't working |
| `epoch` | Epochs completed (float) | `tokens_seen / tokens in one full epoch` | The sweep trials end at 0.25 |
| `resume_count` | Times this run was resumed | Counter stored in the resume state | 0 for an uninterrupted run. Training points shortly before a resume may appear twice |

**Checkpoint events** (logged once at each kept checkpoint, against `train/step`):

| Name | Meaning |
|---|---|
| `train/checkpoint_index` | Number of kept checkpoints saved so far, this one included |
| `train/checkpoint_gpu_hours` | GPU-hours used up to this checkpoint (summed over resumed sessions) |

The same step, `tokens_seen` and GPU-hours are in the checkpoint's `metadata.json`.

## Evaluation (`eval/<name>` and `eval_vs_tokens/<name>`)

Logged at each eval point (every `eval_every_epochs`, and at the end) on `eval_split` (`medmcqa:hp_dev` for the
sweep). They are computed from the wrapper's letter probabilities (`QwenWrapper.score_batch`, which normalizes
over the option letters only) by `src/eval/stub.py`, a stub until the #12 harness lands.

| Name | Meaning | How it is computed | What to look for |
|---|---|---|---|
| `nll` | Mean negative log-likelihood of the correct letter | `mean(−log p(gold))` over questions, with p normalized over the options | **The sweep's ranking metric** (lower is better). It rewards being right *and* confident in the right answer |
| `accuracy` | Accuracy | argmax letter equals the gold letter | Compare with the baseline run (`job_type` eval, `<group>-baseline-*`) |
| `ece` | Expected calibration error | Confidence = max letter probability. 15 equal-width bins, `Σ (n_bin / n) × |accuracy_bin − confidence_bin|` (Guo et al. 2017) | Lower is better. SFT often lowers NLL while raising ECE (overconfidence) |
| `n` | Number of questions evaluated | | 2,000 for hp_dev; fewer if `eval_limit` is set (smoke runs) |

The baseline run of a sweep logs the same `eval/*` values at `train/step` 0. The values are also in the run
folder's `evals.jsonl`.

## Performance (`perf/<name>`, against `train/step`)

Logged by `wandb_logging.log_train` with every training log point.

| Name | Meaning | Notes |
|---|---|---|
| `tokens_per_sec` | Non-padding tokens processed per second since the previous log point | Includes any evaluation or checkpoint saving in that interval, so it dips at eval points. For the pure training throughput use `train_tokens_per_sec` in the run summary |
| `gpu_mem_peak_gb` | Peak GPU memory allocated by PyTorch since the run (session) started | **The number to check against the L4's 24 GB** (it does not include the CUDA context, about 0.5 GB) |
| `gpu_mem_allocated_gb` / `gpu_mem_reserved_gb` | Memory currently allocated / held by PyTorch's caching allocator | Reserved is closer to what `nvidia-smi` shows, and is what actually has to fit. The scripts set `expandable_segments` to keep it close to allocated (17.7 GiB at rank 32 on the A6000) |
| `elapsed_hours` | Hours since this session's run started | Restarts at 0 after a resume. GPU-hours over all sessions are in checkpoint metadata and `train/checkpoint_gpu_hours` |
| `est_cost_usd` | `elapsed_hours × gpu_hourly_usd` | 0 on Insomnia (`gpu_hourly_usd=0`). Not used to compare runs; see `tuning_setup.md` |

W&B's own hardware panels (`system/*`) are recorded by W&B itself. The pipeline never logs under `system/`.

## Run summary

| Name | Meaning |
|---|---|
| `train_tokens_per_sec` | Tokens ÷ time spent in training steps only, for this session. **The throughput to quote**, with the GPU name |
| `train_hours` | Hours spent in training steps, over all sessions |
| `resume_count` | Times the run was resumed |
| `gpu_name`, `wall_clock_hours`, `peak_gpu_mem_gb`, `est_cost_usd`, `final_step`, `total_tokens_seen` | Written by `wandb_logging.finish_run` for the last session |

## Run config (provenance)

Each run's config holds the resolved training config (every key of `configs/base.yaml` after overrides) and:

- `provenance`: `git_commit`, `git_dirty`, `data_manifest_sha256` (SHA-256 of `$FME_SHARED_DIR/data/manifest.json`),
  `model_revision` (read from the downloaded model), `prompt_version` (`PROMPT_VERSION`), `dedupe_scheme`
  (`DEDUPE_SCHEME`)
- `gpu_tag` (e.g. `A6000`; also a run tag and the end of the run name), `epoch_tokens`, `n_train_examples`,
  `total_steps`, `plan_sha256` (fingerprint of the data plan), `n_eval_examples`
- added by `wandb_logging`: `git_commit`, `git_dirty`, `gpu_name`, `gpu_hourly_usd`

Run names are `<run_name>-<gpu_tag>`. Sweep trials are in the group `sweep-<name>` with the tag `sweep`, and smoke
runs are in the group `smoke` with the tag `smoke`.
