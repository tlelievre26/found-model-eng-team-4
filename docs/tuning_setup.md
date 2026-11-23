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
| Batch | 16 per micro-batch × 2 accumulation steps = **32 questions per optimizer step** | |
| Max length | 512 tokens | The longest MedMCQA train example is 472 tokens with the chat template. Nothing is truncated: a longer example stops training with an error |
| Loss | cross-entropy on the **answer-letter token only** | Every other label is -100. The supervised token is the same one the wrapper (`src/models/qwen.py`) reads at evaluation |
| Gradient checkpointing | off (`gradient_checkpointing: true` to turn on) | It caps memory at 3.5–6.5 GB at any length, but each step takes 1.4–1.7× as long (measured, see below). The memory guard below bounds memory instead |
| Memory guard | `max_tokens_per_forward: 1536` | Without checkpointing, activations cost about 9 MB per padded token at rank 32, and length grouping pads a micro-batch to its longest row (up to 472). A micro-batch whose padded size (rows × longest row) exceeds 1,536 tokens is run as consecutive row chunks, each a separate forward/backward pass. The loss is summed per question over the optimizer step, so the gradient is unchanged (a CPU test checks this). `train/forwards_per_step` shows how often it happens |
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
  (1,024 questions). Each chunk is sorted by length and cut into micro-batches of 16, and then the order of all
  micro-batches in the epoch is shuffled deterministically. Padding is per batch (dynamic), and the average
  padding is 1.3% of tokens. An optimizer step never crosses an epoch boundary, so the last step of an epoch may
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
- Load one with `QwenWrapper(prompt_fn=build_messages, adapter_path=".../step-001236")`.
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

**Measured memory and throughput** (RTX A6000, torch 2.14.1+cu126, the real config: batch 16 × 2, length
grouping, `max_tokens_per_forward: 1536`, no gradient checkpointing). Each run trained on 0.05 epoch of the full
MedMCQA train set (593,121 tokens, 248 steps), so the batch shapes match a real run. Memory is in GiB, as PyTorch
reports it. W&B has these as `perf/gpu_mem_peak_gb` and `perf/gpu_mem_reserved_gb`, plus `train_tokens_per_sec`
in the run summary.

| LoRA rank | Peak allocated | Peak reserved | Training throughput | W&B run |
|---|---|---|---|---|
| 8 | 16.6 GiB | 20.7 GiB (before expandable segments) | 1,895 tokens/s | `tput-r8-A6000` |
| 32 | 17.1 GiB | 21.8 GiB (before expandable segments) | 1,882 tokens/s | `tput-r32-A6000` |
| 32, expandable segments | 16.9 GiB | **17.7 GiB** | 1,885 tokens/s | `tput-r32-expseg-A6000` |

- The rank hardly matters: the frozen 4-bit base dominates both memory and time.
- The scripts set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, unless it is already set. With
  variable-width batches the default allocator held about 4 GiB more than it used. With expandable segments,
  peak reserved memory at rank 32 is 17.7 GiB. That leaves about 4 GiB of headroom on the L4 (about 22 GiB
  usable), which still has to be confirmed there.
- The smoke runs (1,024 questions) gave 16.7 GiB / 1,845 tokens/s at rank 16 and 16.9 GiB / 1,448 tokens/s at
  rank 32. The second is probably a slow node: the longer rank-32 run above is the representative one.
- A full hp_dev evaluation (2,000 questions, wrapper batch 16) takes 34 s and peaks at 5.1 GiB. Loading the 4-bit
  model takes about 15 s, and tokenizing the training set about 25 s.

**Why the memory guard and not gradient checkpointing.** Each row below is one micro-batch of 16 at rank 32, in
train mode, with a full forward, backward and optimizer step:

| Padded width | No checkpointing: peak / step time | Gradient checkpointing: peak / step time |
|---|---|---|
| 64 | 11.7 GiB / 0.52 s | 3.5 GiB / 0.88 s |
| 128 | 20.5 GiB / 0.90 s | 3.8 GiB / 1.26 s |
| 256 | 38.0 GiB / 1.66 s | 4.8 GiB / 2.36 s |
| 384, 472 | out of memory (over 47 GiB) | 5.9 / 6.5 GiB, 3.4 / 4.2 s |

The median micro-batch is 69 tokens wide, but 1% are 251 or wider, and the longest is 472. Without a guard the
main run would run out of memory on its widest micro-batches even on the A6000. Checkpointing would fix this at
1.4–1.7× the step time on every batch. The guard (`max_tokens_per_forward`) splits only the wide ones, and costs
2.26 forward passes per step on average instead of 2. To use checkpointing instead, set
`gradient_checkpointing=true max_tokens_per_forward=null`.

**Projected sweep time (A6000):** a trial is 2,965,496 tokens. At about 1,890 tokens/s that is 26 minutes of
training. Add two hp_dev evaluations (about 1 minute) and startup (model load and tokenization, about 1 minute),
for about 28–29 minutes per trial. The 9 trials and the baseline evaluation take **about 4.3–4.5 hours** in one
job, well within `sweep.sbatch`'s 2-day limit. For comparison, the main run (6 epochs, 71.2M tokens) would be
about 10.5 hours of training on an A6000.

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
