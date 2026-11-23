# CLAUDE.md

Guidance for coding agents working in this repo. Human setup steps and team rules are in [README.md](README.md). This file covers what the project is trying to show and the design decisions code must respect.

## Project

Foundation Model Engineering course, Team 4, topic #22: **a trustworthy small model for medical question answering.**

**Goal:** fine-tune a ~4B-parameter model, using only public data, on a single GPU with a limited compute budget, so that it is both accurate on medical multiple-choice QA and *trustworthy*: its confidence is well calibrated and it can abstain when it is likely to be wrong. The motivation is that hospitals could then self-host a small model (keeping control over data and tuning, without sending PHI to outside vendors) instead of relying on vendor models.

**Research question:** how much does SFT on one GPU improve a small model, and how much compute does it take to reach target thresholds for accuracy, calibration, and willingness to abstain?

**Hypotheses:**
- Each metric plateaus beyond some compute budget, but they will not all peak at the same point.
- Answer-only SFT improves accuracy, but calibration and selective accuracy may not improve at the same checkpoints.

**Scope limit:** these benchmarks do not establish clinical safety, and they do not verify that generated explanations are correct. Never claim otherwise in docs, results, or plots.

## Approach

- **Base model:** `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0). The untuned model is the main baseline.
- **QLoRA:** base weights frozen in 4-bit, only LoRA adapters trained. Checkpoints save adapter weights only.
- **Answer-only SFT first:** the target is the answer letter, with loss computed on the answer only, not the question.
- **Staged, compute-centric experiment:** train once for a fixed compute budget and save checkpoints *evenly spaced in compute* (training tokens or GPU-time), not just in steps. Every checkpoint records step, tokens seen, and GPU-hours so far, and every checkpoint is benchmarked. This shows how much compute each metric needs.
- **Abstention:** for each checkpoint, pick a confidence threshold on a small validation subset and abstain below it. Compare with and without temperature scaling (one temperature per checkpoint).
- **If answer-only SFT doesn't help:** first inspect data quality and training settings. Only then consider explanation-based SFT using the explanations MedMCQA already provides (motivated by Hsieh et al. 2023 and Meerkat 2025), and filter out empty or uninformative explanations. Other optional extensions: an abstention-aware SFT variant and retrieval over PubMed abstracts. Optional work is dropped first if time runs short.

## Data

| Dataset | License | Role |
|---|---|---|
| MedMCQA (`openlifescienceai/medmcqa`) | Apache-2.0 (selected release) | Training data, and held-out evaluation for accuracy |
| MedQA (English test set) | MIT | Cross-dataset transfer |
| MedAbstain | **CC-BY-NC 4.0** | Calibration and abstention benchmark (abstention and perturbed-abstention tasks) |

- **MedAbstain must never be re-uploaded or redistributed.** It stays in the shared data directory: not in git, not on the HF Hub, not in W&B artifacts.
- **Split discipline is critical.** Hyperparameter tuning, temperature fitting, threshold selection, and final evaluation each use *disjoint* questions. The official MedMCQA validation set is our `test` set, used only for final reported numbers. Never tune, fit, or select anything on it. MedMCQA `val` is divided into `hp_dev`, `calib`, and `thresh` (issue #11). Split assignments are deterministic (fixed seed, sorted IDs) and frozen before any tuning run uses them.

## Evaluation

For the baseline and every checkpoint:
- **Accuracy**, **mean NLL** of the correct letter (used to rank sweep trials), and **ECE**: confidence is the max letter probability, with equal-width bins (15 by default), per Guo et al. 2017.
- **Calibration before and after temperature scaling.**
- **Selective accuracy** (accuracy on questions the model doesn't abstain on) and **risk–coverage curves** (Geifman & El-Yaniv 2017), comparing models at the same coverage.
- **MedAbstain:** abstention rate, and accuracy among answered questions.
- **Memorization:** planted-canary tests.
- **Serving:** single-GPU latency, peak memory, and cost per 1,000 questions.

Define "trustworthy" numerically (target accuracy, ECE, and selective accuracy at a chosen coverage) after the baseline and *before* fine-tuning, so "how much compute to reach the thresholds" has a clear answer (issue #12).

## Compute and budget

- **Target hardware:** a single NVIDIA L4 (GCP `g2-standard-4`, 24 GB, about $0.71/hr on-demand). Everything must fit in 24 GB. Check peak memory at the largest LoRA rank you plan to use.
- **Budget:** team credits are $500, with a hard cap of 80% ($400). The plan is about $101 total: 85 L4-hours plus 3 A100-hours of GPU (about $71), plus storage and a CPU VM. Each member's runs stay under $25.
- **Planned GPU-hours (L4):** setup/smoke tests 6, baseline eval + hyperparameter search 8, main answer-only QLoRA run 20, checkpoint evaluation 8, canary memorization 10. Optional: explanation SFT 12, abstention-aware SFT 4, retrieval 8, reserve 11. Optional: A100 serving reference 3 A100-hours.
- **Long runs may use spot VMs:** checkpoint at least every 30 minutes, and make re-running the same command resume from the latest checkpoint.
- **Every kept run must log GPU type, wall-clock time, peak memory, and dollar cost.** `src/utils/wandb_logging.py` does this automatically.
- **Insomnia** (Columbia HPC, available through Taylor's lab) can be used for heavier experiments (see Environments).

## W&B logging

All training and eval runs log through `src/utils/wandb_logging.py`. Usage is in the module docstring and the README.

- API: `start_run(name, config, ...)` (works as a context manager), `log_train(step, tokens_seen, **metrics)`, `log_eval(metrics)`, `finish_run()`. Eval-only scripts pass `job_type="eval"`.
- Defaults are entity `tjl2152-columbia-university` and project `FME-project`. Override with `WANDB_ENTITY` / `WANDB_PROJECT` or the `entity=` / `project=` arguments.
- **`tokens_seen` is the project's measure of training compute:** a cumulative count of *non-padding* tokens (e.g. the sum of `attention_mask`), not `batch_size × seq_len`. Use this same definition everywhere compute is measured (checkpoint spacing, the cost estimator in #13, plots), so the numbers line up.
- Metric namespaces: `train/*`, `eval/*`, and `perf/*` are plotted against `train/step`. `train_vs_tokens/*` and `eval_vs_tokens/*` are plotted against `train/tokens_seen`. Don't log under `system/`, because W&B uses it for its own hardware metrics.
- The GPU price comes from `DEFAULT_GPU_HOURLY_USD`. Reuse it rather than redefining it, and pass `gpu_hourly_usd=` when running on non-L4 hardware so cost estimates stay honest.
- Tests and quick local runs use `WANDB_MODE=disabled`. Everything must still work in that mode.
- `scripts/wandb_smoke_test.py` is a dummy run for checking the setup.

## Environments

**Team VM (GCP L4):** large files live under `/shared` (`models/`, `data/`, `checkpoints/`), with `HF_HOME=/shared/models` so the base model isn't re-downloaded.

**Insomnia (Columbia HPC, Slurm):**
- Login nodes have no GPU, so run GPU work through Slurm. The `pmg` account can use partition `pmg1` (A6000 nodes, up to 7 days). The `short` (12 h) and `burst` (14 days) partitions also have GPU nodes (A6000, L40/L40S, H100, and others). Example:
  ```bash
  srun -A pmg -p pmg1 --gres=gpu:1 -c 4 --mem=16G -t 15 \
    bash -c 'PYTHONPATH=. python scripts/wandb_smoke_test.py'
  ```
- `/shared` does not exist there. Set `FME_SHARED_DIR` to a directory with the same layout.
- The default conda Python has torch 2.8 and wandb 0.22, *not* the pinned stack, and the repo isn't pip-installed into it. That's fine for smoke tests with `PYTHONPATH=.`. Use an environment with the pinned versions for real runs.
- The login node can reach W&B. Internet access from compute nodes hasn't been verified, so if it fails, use `WANDB_MODE=offline` and `wandb sync` from the login node afterwards.
- Throughput, memory, and cost numbers from non-L4 GPUs aren't directly comparable to the L4 plan. Tag such runs and set `gpu_hourly_usd` accordingly.

## Code conventions

- Package is imported as `src` (e.g. `from src.utils import paths`). Install with `pip install -e ".[dev]"`. Python ≥ 3.11.
- **All shared paths come from `src/utils/paths.py`.** Never hard-code `/shared/...`. Add new locations to `paths.py`. The root is overridable with `FME_SHARED_DIR`.
- Dependencies are pinned exactly in `pyproject.toml`. Change them only in a dedicated PR.
- Large files (data, weights, checkpoints, `wandb/` run dirs) never go in git.
- Layout: `src/data` (loading, splits, prompt formatting), `src/models` (model wrapper), `src/training`, `src/eval`, `src/utils`, `configs/` (YAML run configs), `scripts/` (CLI entry points), `tests/` (pytest).
- Don't name modules after standard-library modules (which is why the module is `wandb_logging.py`, not `logging.py`).
- Run settings belong in config files and must be overridable per run. Don't hard-code hyperparameters in scripts.
- Work on a branch and merge through a PR. Don't push to `main`.

## Design in progress

Agreed interfaces and requirements for in-flight work are in GitHub issues. Read the relevant issue (`gh issue view <n>`) before implementing:
#3 MedMCQA/MedQA loading and deduplicated splits · #4 MedAbstain loader · #5 Qwen3-4B wrapper · #6 W&B logging · #7 training script, compute-spaced checkpoints, and hyperparameter sweep · #11 `hp_dev` / `calib` / `thresh` subsets · #12 evaluation harness (`evaluate(model, examples, ...)`) and baseline · #13 compute cost estimator.

## References

- Hsieh et al. 2023, distilling step-by-step: https://aclanthology.org/2023.findings-acl.507/
- Meerkat 2025, explanations for medical QA: https://www.nature.com/articles/s41746-025-01653-8
- Machcha et al. 2026, MedAbstain: https://aclanthology.org/2026.eacl-long.291/
- Guo et al. 2017, calibration and temperature scaling: https://arxiv.org/abs/1706.04599
- Geifman & El-Yaniv 2017, selective classification and risk–coverage: https://arxiv.org/abs/1705.08500
- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- MedMCQA: https://huggingface.co/datasets/openlifescienceai/medmcqa
- MedQA: https://github.com/jind11/MedQA
- MedAbstain: https://github.com/sravanthi6m/MedAbstain
