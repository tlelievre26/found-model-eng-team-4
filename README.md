# found-model-eng-team-4
This is the repo for our small medical question answering model we are building as a part of the Foundation Model Engineering course

## Shared directory

All large files — data, model weights, and checkpoints — live in one shared folder on the team VM, **never in git**:

```
/shared/
  models/        # base model weights (HF_HOME points here so nobody re-downloads)
  data/          # raw downloads + processed splits
  checkpoints/   # training run outputs
```

### Rules for `/shared`
- **Don't delete or overwrite files you didn't create.** Ask the owner on Slack first.
- **Never re-upload MedAbstain anywhere.** Its license is CC-BY-NC; it stays in `/shared/data` only.
- **Point `HF_HOME` at `/shared/models`** so Hugging Face loads the shared copy of the base model (`Qwen/Qwen3-4B-Instruct-2507`) instead of downloading it again. Run this once on the VM:
  ```bash
  echo 'export HF_HOME=/shared/models' >> ~/.bashrc && source ~/.bashrc
  ```

### Using `paths.py`
[`src/utils/paths.py`](src/utils/paths.py) is the **only** place shared paths are defined. Never hard-code `/shared/...` anywhere else — import from `paths` instead:

```python
from src.utils import paths

paths.SHARED_DIR       # /shared
paths.MODELS_DIR       # /shared/models
paths.DATA_DIR         # /shared/data
paths.CHECKPOINTS_DIR  # /shared/checkpoints
```

The shared root comes from the `FME_SHARED_DIR` environment variable and defaults to `/shared`. On the VM you don't need to set anything. To run the code on another machine (e.g. your laptop), point it at a local folder with the same layout:

```bash
export FME_SHARED_DIR=~/fme_shared
```

If you need a new shared location (e.g. a new subfolder), add it to `paths.py` in a PR rather than building the path inline.

## Getting started

### 1. SSH into the VM
VM setup is tracked in #2. Once the VM is running:

```bash
gcloud compute ssh <vm-name> --zone <zone>
```

Check the VM is up and the GPU is visible with `nvidia-smi`.

### 2. Clone the repo into your home directory
Everyone works in their **own** clone in their home directory — never edit someone else's checkout.

```bash
cd ~
git clone git@github.com:tlelievre26/found-model-eng-team-4.git
cd found-model-eng-team-4
```

### 3. Create an environment and install the package
Python 3.11+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

`-e` installs the package in editable mode, so your code changes take effect without reinstalling. Dependencies are pinned to exact versions in [`pyproject.toml`](pyproject.toml) — change them only through a PR so everyone stays on the same stack.

### 4. Check it works

```bash
python -c "from src.utils import paths; print(paths.MODELS_DIR, paths.DATA_DIR, paths.CHECKPOINTS_DIR)"
# /shared/models /shared/data /shared/checkpoints
pytest
```

### 5. Set up your API keys
Keys live in your own `~/.bashrc` — **never commit them**. Run each command below and paste your key when prompted. The key isn't shown on screen or saved to your shell history.

```bash
read -rsp "W&B API key: " key && echo "export WANDB_API_KEY=$key" >> ~/.bashrc && unset key && echo
```

```bash
read -rsp "Hugging Face token: " key && echo "export HF_TOKEN=$key" >> ~/.bashrc && unset key && echo
```

Then reload your shell:

```bash
source ~/.bashrc
```

Get your keys from https://wandb.ai/authorize and https://huggingface.co/settings/tokens.

## Building the datasets

[`scripts/build_all_datasets.py`](scripts/build_all_datasets.py) builds every dataset in one go: it downloads MedAbstain, builds the MedMCQA and MedQA splits (excluding the MedAbstain test questions), validates the result and records it in `/shared/data/manifest.json` (git commit, library versions, dataset revisions, settings, file checksums, split sizes).

- **It is run once, on the team VM, from a clean `main`, into `/shared/data`.** Everyone else uses that build. **Don't run it yourself**: one build means everyone works with the same splits.
  ```bash
  python -m scripts.build_all_datasets
  ```
- **Other machines** (e.g. Insomnia) copy `/shared/data` into `$FME_SHARED_DIR/data` and check the copy against the manifest. A non-zero exit means the copy differs:
  ```bash
  FME_SHARED_DIR=/path/to/shared python -m scripts.build_all_datasets --verify-only
  ```
- **Rebuild only when a build setting changes** (the settings block in `src/data/medqa_datasets/data.py`, or the pinned dataset revisions), and **never once tuning has started**: a rebuild can change which questions are in each split. The script refuses to overwrite an existing build unless `--force` is given.

Details are in [`src/data/medqa_datasets/README.md`](src/data/medqa_datasets/README.md).

## Logging runs to W&B

Every training and eval run should log to the shared W&B project through [`src/utils/wandb_logging.py`](src/utils/wandb_logging.py). It records the full config, git commit, GPU type, wall-clock time, peak GPU memory, and estimated cost (hours × $0.71/hr for the L4 VM).

```python
from src.utils import wandb_logging as wl

with wl.start_run("lora-r16-lr2e-4", config):          # config: dict or dataclass
    tokens_seen = 0
    for step, batch in enumerate(loader, start=1):
        loss = train_step(batch)
        tokens_seen += int(batch["attention_mask"].sum())   # real tokens, not padding
        if step % 10 == 0:
            wl.log_train(step, tokens_seen, loss=loss, lr=scheduler.get_last_lr()[0])
        if step % 500 == 0:
            wl.log_eval({"accuracy": evaluate(model)})
# leaving the block logs the time / memory / cost summary, even if training crashes
```

For an eval-only script, use `wl.start_run(name, config, job_type="eval")` and call `wl.log_eval(...)`. You can also call `wl.start_run(...)` and `wl.finish_run()` yourself instead of using `with`.

What shows up in W&B:
- `train/*` and `eval/*`: your metrics plotted against optimizer step.
- `train_vs_tokens/*` and `eval_vs_tokens/*`: the same metrics plotted against tokens processed, so runs with different batch sizes or sequence lengths can be compared by compute.
- `perf/*`: GPU memory (allocated / reserved / peak), tokens per second, elapsed hours, running cost.
- Run summary: `gpu_name`, `wall_clock_hours`, `peak_gpu_mem_gb`, `est_cost_usd`, `final_step`, `total_tokens_seen`.

To check your setup, run the dummy script:

```bash
python scripts/wandb_smoke_test.py                       # logs a run to W&B
WANDB_MODE=disabled python scripts/wandb_smoke_test.py   # W&B off, prints the summary only
```

Use `WANDB_MODE=disabled` for quick local tests. Runs go to the `FME-project` project in the `tjl2152-columbia-university` team by default. To log somewhere else (e.g. a personal sandbox), set `WANDB_ENTITY` and/or `WANDB_PROJECT`, or pass `entity=` / `project=` to `start_run`.

## Team usage rules
- **Work on a branch.** Clone into your own home directory, create a branch for your work, and merge into `main` through a pull request.
- **Large files go in `/shared`, never in git.** This includes data, model weights, and checkpoints. MedAbstain's license (CC-BY-NC) means we must not re-upload it.
- **There is one GPU — share it.**
  - Check `nvidia-smi` before starting anything.
  - Run long jobs inside `tmux` so they survive disconnects (`tmux new -s <name>`, detach with `Ctrl-b d`, reattach with `tmux attach -t <name>`).
  - If the GPU is unavailable, please message in the Slack to coordinate.
- **Don't delete or overwrite files in `/shared` that you didn't create.**
- **API keys stay out of the repo.** Keep W&B and Hugging Face keys in your own environment or a git-ignored `.env`.
- **Stop the VM when you're done** and nobody else is using it — it costs money while running.

## Repo layout

```
src/              # importable package (import as `src`, e.g. `from src.utils import paths`)
  data/           # dataset loading + prompt formatting
  models/         # model wrapper
  training/       # training script(s)
  eval/           # evaluation + metrics
  utils/          # W&B logging, paths, misc helpers
configs/          # YAML config files for runs
scripts/          # entry points you run from the command line
tests/            # quick checks (pytest)
```
