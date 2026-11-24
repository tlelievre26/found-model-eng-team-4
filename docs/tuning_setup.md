# Training and hyperparameter tuning setup

How answer-only QLoRA training (`scripts/train.py`) and the hyperparameter sweep (`scripts/run_sweep.py`) work,
which settings are fixed and why, and how to launch them on Insomnia. Every metric they log is described in
[`wandb_metrics.md`](wandb_metrics.md).

These runs measure accuracy and calibration on MedMCQA-style multiple-choice questions. They do not establish
clinical safety, and they say nothing about whether any generated explanation is correct.

## The sweep

| | |
|---|---|
| Grid | `learning_rate` ∈ {5e-5, 1e-4, 2e-4} × `lora_rank` ∈ {8, 16, 32}: 9 trials |
| Trial length | 0.25 epoch of MedMCQA train: 2,965,496 tokens, 39,523 questions, 1,236 optimizer steps |
| Evaluation | MedMCQA `hp_dev` (2,000 questions) at 1/8 epoch (the midpoint) and at 1/4 epoch (the end) |
| Baseline | the untuned base model, evaluated on `hp_dev` before the first trial |
| Ranking | final `eval/nll` (mean negative log-likelihood of the correct letter, lower is better) |
| Selection | **by the team, by hand.** The sweep prints and saves a ranking, but it does not pick a winner |

NLL is the ranking metric because it moves smoothly with both accuracy and confidence. Accuracy on 2,000
questions is coarse (one question is 0.05 points). Look at accuracy and ECE too, and at how the midpoint and
end evaluations differ: a trial whose NLL is still falling at the end may simply need more compute.

Trials run one after another in a single process and a single Slurm job, with the same `train()` function as
every other run. Trial *k* writes to `$CHECKPOINTS_DIR/sweep-<name>/<trial>/` (e.g. `sweep-v1/lr0.0001_r16/`)
and logs to W&B group `sweep-<name>`. This deliberately differs from #7's plan to use a W&B sweep controller: a
fixed loop with fixed output folders makes resuming reliable (see [Resume](#resume)). Each trial saves one
adapter, at its end. Trials don't need compute-spaced checkpoints.

`configs/sweep.yaml` holds the base config, the shared trial overrides (`epochs: 0.25`,
`eval_split: medmcqa:hp_dev`, `eval_every_epochs: 0.125`), the grid and the sweep name. Results are in
`$CHECKPOINTS_DIR/sweep-<name>/summary.json` and `baseline_eval.json`, and in W&B (`eval/nll` per run in
the group).

## Fixed settings (`configs/base.yaml`)

| Setting | Value | Why |
|---|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554` | Training checks the downloaded copy's revision against `model_revision` and stops on a mismatch |
| Quantization | 4-bit NF4, double quantization, **bf16 compute** | QLoRA. bf16 has float32's exponent range, so no loss scaling is needed. Training stops if the GPU lacks bf16 (L4, A6000, A100 all have it) |
| LoRA targets | all linear layers: `q, k, v, o, gate, up, down` | Adapting every linear layer matches full fine-tuning more closely than attention-only LoRA |
| LoRA alpha | `2 × rank` | The LoRA update is scaled by alpha / rank. Tying alpha to rank keeps that scale fixed, so changing the rank does not also change the effective learning rate |
| LoRA dropout | 0.05 | |
| Optimizer | paged 8-bit AdamW (bitsandbytes), weight decay 0 | Small optimizer state; paging avoids out-of-memory spikes |
| Gradient clipping | 1.0 (global norm) | |
| Learning rate schedule | linear warmup for `warmup_steps: 100`, then **constant** | Warmup in *steps*, not a ratio: with a ratio the 1,236-step trials and the ~30,000-step main run would warm up for very different lengths, and the best learning rate might not carry over. A constant rate after warmup means a checkpoint midway through the main run is trained the same way as a trial |
| Batch | 4 per micro-batch × 8 accumulation steps = **32 questions per optimizer step** | Chosen for simplicity and a guaranteed memory fit over throughput: even the worst possible micro-batch (4 × 512 tokens) fits in the L4's 24 GB without gradient checkpointing (measured below). Batch 16 × 2 ran about twice as fast, but its long-tail batches needed more than 47 GiB |
| Max length | 512 tokens | The longest MedMCQA train example is 472 tokens with the chat template. Nothing is truncated: a longer example stops training with an error |
| Loss | cross-entropy on the **answer-letter token only** | Every other label is -100. The supervised token is the same one the wrapper (`src/models/qwen.py`) reads at evaluation |
| Gradient checkpointing | off (`gradient_checkpointing: true` to turn on) | The alternative if memory is ever short: at 4 × 512 tokens it peaks at 3.8 GiB instead of 20.4 GiB, but each step takes 1.5× as long (measured below) |
| Seed | 42 | Sets the data order, the LoRA initialization and dropout |

The letter-only loss: the training text is the chat template of `build_messages(ex, include_answer=True)`,
which is exactly the scoring prompt (`build_messages(ex)` with `add_generation_prompt=True`) followed by the answer
letter and `<|im_end|>\n`. Training checks this for every example and fails if it does not hold. The letters
A–F are single tokens (ids 32–37). The model is trained to put probability on the gold letter at the answer
position, which is what the evaluation reads. The trailing `<|im_end|>\n` tokens are part of the input but
unsupervised, and the causal mask means they cannot affect the answer position.

Inputs are left-padded (as the wrapper pads), and only the last few positions' logits are computed. This keeps
the 151,936-entry vocabulary from being materialized for every position.

## Data selection and the budget

- **Training data:** `train_datasets: [medmcqa]`, from `load_splits(name)["train"]` (158,146 questions,
  11,861,451 tokens per epoch, 75 tokens per question on average). MedQA train can be added later by listing it.
- **Order:** examples are sorted by id, then shuffled with the seed. Epoch *k* is reshuffled with `seed + k`.
- **Budget in epochs (a float, `epochs`):** each whole epoch uses every example. A fractional part takes the
  **shortest prefix of that epoch's shuffled order whose non-padding token count reaches
  `fraction × (tokens in one full epoch)`**. For the sweep, `epochs: 0.25` is the first 39,523 questions of
  epoch 0's order, 2,965,496 tokens. All of this is decided up front, before training starts.
- **Length grouping:** the selected examples, in shuffled order, are cut into chunks of 64 micro-batches
  (256 questions). Each chunk is sorted by length and cut into micro-batches of 4, and then the order of all
  micro-batches in the epoch is shuffled deterministically. Padding is per batch (dynamic), and the average
  padding is 0.9% of tokens. An optimizer step never crosses an epoch boundary, so the last step of an epoch may
  have fewer than 32 questions.
- **Compute measure:** `tokens_seen` is the cumulative number of non-padding input tokens (the sum of
  `attention_mask`, the prompt included). It is the definition in `src/utils/wandb_logging.py`, used for
  checkpoint spacing, eval spacing, the budget, and the x-axis of `*_vs_tokens` plots. `epoch` (a float) is
  `tokens_seen / tokens per epoch`.

## Checkpoints

- **Kept (evaluated) checkpoints:** `checkpoints_per_epoch` of them per epoch, evenly spaced in tokens
  (the first optimizer step at or past each `j / checkpoints_per_epoch` epoch of tokens), plus one at the end of
  the run. With the default of 1, the main run keeps one checkpoint at the end of each epoch. **Revisit this for
  the main run:** to see where each metric plateaus, 2–4 per epoch (12–24 checkpoints over 6 epochs) is probably
  more useful. Each costs a few hundred MB and an evaluation.
- Each is a folder `step-XXXXXX/` holding the LoRA adapter (`adapter_model.safetensors`, `adapter_config.json`)
  and `metadata.json`: step, epoch (float), `tokens_seen`, examples seen, GPU-hours, training-only hours,
  wall-clock hours since the run started, GPU name, W&B run id, the resolved config, git commit and dirty flag,
  the data manifest's SHA-256, the model revision, `PROMPT_VERSION` and `DEDUPE_SCHEME`.
- Evaluate one with `scripts/evaluate.py --adapter .../step-001236 --dataset medmcqa --split <split>`, or load it
  in Python with `QwenWrapper(adapter_path=".../step-001236")`.
- **GPU-hours** are the time the training process ran (model loading, training, evaluation and saving), summed
  over sessions, on one GPU. Time lost between the last resume state and a crash is not counted.

## Resume

- A rolling resume state, `resume.pt`, is overwritten every `resume_every_minutes` (20), after every kept
  checkpoint or evaluation, and when Slurm signals the job (`SIGTERM`, sent 5 minutes before the time limit).
  It holds the adapter, the optimizer and scheduler states, the RNG states, the step reached (which fixes the
  position in the data plan), `tokens_seen`, the accumulated GPU-hours, and the W&B run id.
- **Re-running the same command continues the run:** same step, same data order, same W&B run (resumed by id).
  A CPU test checks that a crashed and resumed run ends with bit-identical adapters and evaluations.
- Resuming refuses to start if the config or the data plan differs from the saved run. Only `resume_every_minutes`,
  `log_every_steps`, `eval_batch_size` and `wandb_tags` may change between sessions.
- A completed run writes `DONE` (a JSON summary), deletes `resume.pt`, and refuses to train again. Use a new
  `run_name` to retrain.
- The sweep skips trials with `DONE`, resumes a partial one, and evaluates the baseline only once
  (`baseline_eval.json`). Re-submitting the sweep job continues where it stopped.
- Steps between the last resume state and an interruption are redone, so W&B may show a few training points twice.

## Launching (Insomnia)

From the repo root on the login node. Both scripts request one A6000 on `pmg1` (`-A pmg -p pmg1
--gres=gpu:A6000:1`), activate `$FME_PROJECT/envs/fme`, set `FME_SHARED_DIR` and `HF_HOME`, and write logs to
`$FME_PROJECT/fme_scratch_train/logs/`. Re-submitting the same command resumes.

```bash
# smoke test (a few minutes): 1,024 questions, 1.5 epochs, 3 kept checkpoints, small hp_dev evals
sbatch scripts/slurm/train.sbatch configs/smoke.yaml
# the same at the largest rank, for peak memory and throughput
sbatch scripts/slurm/train.sbatch configs/smoke.yaml run_name=smoke-r32 lora_rank=32
# the sweep (baseline + 9 trials)
sbatch scripts/slurm/sweep.sbatch configs/sweep.yaml
# the main run, after the team has chosen the hyperparameters and the budget
sbatch scripts/slurm/train.sbatch configs/main.yaml learning_rate=1e-4 lora_rank=16
```

Without Slurm (e.g. on the L4 VM): `python scripts/train.py --config configs/smoke.yaml [key=value ...]` and
`python scripts/run_sweep.py --sweep configs/sweep.yaml [key=value ...]`.

`gpu_hourly_usd` is the only setting that differs between the L4 VM and Insomnia. `null` (the default) means
`wandb_logging.DEFAULT_GPU_HOURLY_USD`, the L4 VM's price, and is refused on any other GPU. The Insomnia scripts
pass `gpu_hourly_usd=0`, so `perf/est_cost_usd` is 0 there: the A6000 time is not billed to the team, and the L4
price would misstate it. Dollar figures are not used to compare runs (see below).

## Hardware: these runs use A6000s, not the L4

The sweep and the smoke runs run on Insomnia's RTX A6000s (48 GB), not the project's target L4 (24 GB). The run
name, the W&B tags and the run config (`gpu_tag`, `provenance`, `gpu_name`) all record the GPU type.

**Measured memory and throughput** (RTX A6000, torch 2.14.1+cu126, the real config: batch 4 × 8, length
grouping, no gradient checkpointing, `expandable_segments`). Each throughput run trained on 0.05 epoch of the full
MedMCQA train set (593,121 tokens, 248 optimizer steps), so the batch shapes match a real run. Memory is in GiB, as
PyTorch reports it. W&B has these as `perf/gpu_mem_peak_gb` and `perf/gpu_mem_reserved_gb`, plus
`train_tokens_per_sec` in the run summary.

| LoRA rank | Training throughput | Step time (32 questions) | Peak allocated | Peak reserved | W&B run |
|---|---|---|---|---|---|
| 8 | 1,027 tokens/s | 2.33 s | 13.0 GiB | 13.4 GiB | `tput-b4-r8-A6000` |
| 32 | 952 tokens/s | 2.51 s | 13.4 GiB | 13.9 GiB | `tput-b4-r32-A6000` |

**Worst case** (one micro-batch of 4 at the full 512 tokens, rank 32, train mode, forward + backward + optimizer
step): **20.4 GiB allocated, 20.9 GiB reserved**, 0.88 s. The L4 has about 22 GiB usable, so even this fits, with
about 1 GiB to spare. The longest real example is 472 tokens, so real batches stay below this. Smaller widths:
4 × 64 is 5.0 GiB, 4 × 128 is 7.2 GiB, 4 × 256 is 11.6 GiB and 4 × 384 is 16.0 GiB. With gradient checkpointing,
4 × 512 is 3.8 GiB at 1.33 s.

- Batch 4 × 8 was chosen for simplicity and a guaranteed fit, not for speed. The earlier batch 16 × 2 ran at
  about 1,890 tokens/s (1.27 s per step), but its long-tail micro-batches (16 × 384 or wider) needed more than
  47 GiB. It would have required splitting wide batches or turning gradient checkpointing on.
- The rank matters little: the frozen 4-bit base dominates both memory and time.
- The scripts set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, unless it is already set, so reserved memory
  stays close to allocated with variable-width batches.
- The smoke config (1,024 questions, 48 steps) at rank 16 ran at 1,050 tokens/s with a 14.4 GiB peak
  (`smoke-b4-A6000`).
- A full hp_dev evaluation (2,000 questions, wrapper batch 16) takes 34 s and peaks at 5.1 GiB. Loading the 4-bit
  model takes about 15 s, and tokenizing the training set about 25 s.
- Peak memory on the L4 itself still has to be confirmed there.

**Projected sweep time (A6000):** a trial is 2,965,496 tokens. At 950–1,030 tokens/s that is 48–52 minutes of
training. Add two hp_dev evaluations (about 1 minute) and startup (about 1 minute), for about 50–54 minutes per
trial. The 9 trials and the baseline evaluation take **about 7.5–8 hours** in one job, within `sweep.sbatch`'s
2-day limit.

**Projected main run (A6000):** 6 epochs are 71.2M tokens, so **about 19–21 hours** of training at 950–1,030
tokens/s, plus about 4 minutes for six hp_dev evaluations. That is within `train.sbatch`'s 3-day limit, and a
resubmit resumes it if needed.

## L4-equivalent cost

Tokens are the hardware-independent measure of training compute. GPU-hours and dollars depend on the GPU, so
A6000 GPU-hours must not be reported as if they were L4 GPU-hours:

- The A6000 has about 768 GB/s of memory bandwidth against the L4's 300 GB/s, more compute, and twice the
  memory. 4-bit QLoRA at these short lengths is largely bound by memory bandwidth (every step dequantizes all
  the weights), so the A6000 is likely to be much faster per token. The exact ratio depends on the batch shape.
- The software stack (driver, CUDA, kernels) differs between Insomnia and the VM, which also changes throughput.

To report L4 GPU-hours and cost:

1. On the team VM's L4, with the real config (the same batch size, max length and length grouping), run the
   smoke config once per LoRA rank of interest. Read `train_tokens_per_sec` from the run summary (training
   steps only, without evaluation and checkpoint time).
2. Convert any run's compute: `L4-hours = tokens_seen / L4_tokens_per_sec / 3600`, and
   `L4 cost = L4-hours × $0.71` (`DEFAULT_GPU_HOURLY_USD`). Add the evaluation time separately, measured on the
   L4 too.

Every throughput, memory and GPU-hour number stays tied to the hardware it was measured on. Always state the GPU
next to it.
