"""Answer-only QLoRA fine-tuning with compute-spaced checkpoints and exact resume.

One function, train(cfg), used by scripts/train.py and by every trial of scripts/run_sweep.py.

Outputs in output_dir ($CHECKPOINTS_DIR/<run_name>/ by default):
  step-XXXXXX/   kept checkpoints: LoRA adapter (loadable with QwenWrapper(adapter_path=...)) + metadata.json
  resume.pt      rolling resume state, overwritten every resume_every_minutes, after each event and on SIGTERM
  evals.jsonl    eval metrics (no question text) at each eval point
  config.json    the resolved config
  DONE           written when the run completes; train() then refuses to run again
"""

from __future__ import annotations

import datetime
import gc
import hashlib
import json
import os
import random
import shutil
import signal
import threading
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.medqa_datasets import PROMPT_VERSION, build_messages
from src.data.medqa_datasets.data import DEDUPE_SCHEME
from src.eval import evaluate
from src.models.qwen import DEFAULT_MODEL_PATH, QwenWrapper, load_base_model
from src.training import data as td
from src.utils import paths
from src.utils import wandb_logging as wl

DONE_FILE = "DONE"
RESUME_FILE = "resume.pt"
# Config keys that may change between the sessions of one run (they don't affect what is trained).
RESUMABLE_CHANGES = {"resume_every_minutes", "log_every_steps", "eval_batch_size", "wandb_tags"}


class RunAlreadyComplete(RuntimeError):
    """The output folder has a DONE marker."""


def output_dir(cfg: dict) -> Path:
    if cfg.get("output_dir"):
        return Path(cfg["output_dir"])
    if not cfg.get("run_name"):
        raise ValueError("Set run_name (outputs go to $CHECKPOINTS_DIR/<run_name>)")
    return paths.CHECKPOINTS_DIR / cfg["run_name"]


def is_done(cfg: dict) -> bool:
    return (output_dir(cfg) / DONE_FILE).exists()


def gpu_tag(gpu_name: str) -> str:
    """Short hardware name for run names and tags: "NVIDIA RTX A6000" -> "A6000", "NVIDIA L4" -> "L4"."""
    return gpu_name.split()[-1] if gpu_name else "cpu"


def resolve_gpu_hourly_usd(cfg: dict, gpu_name: str) -> float:
    """The configured price; null means the L4 VM price and is only allowed on an L4."""
    if cfg["gpu_hourly_usd"] is not None:
        return float(cfg["gpu_hourly_usd"])
    if "L4" not in gpu_name.split():
        raise ValueError(f"gpu_hourly_usd is null (the L4 VM price) but this GPU is {gpu_name!r}. "
                         "Set gpu_hourly_usd explicitly (the Insomnia Slurm scripts pass gpu_hourly_usd=0).")
    return wl.DEFAULT_GPU_HOURLY_USD


def downloaded_revision(model_path: Path) -> str | None:
    """Commit of a model downloaded with `hf download --local-dir` / snapshot_download(local_dir=...)."""
    meta = Path(model_path) / ".cache" / "huggingface" / "download" / "config.json.metadata"
    return meta.read_text().split()[0] if meta.is_file() else None


def provenance(cfg: dict, model_path: Path | None) -> dict:
    found = downloaded_revision(model_path) if model_path else None
    if model_path and cfg["model_revision"] and found != cfg["model_revision"]:
        raise RuntimeError(f"Model at {model_path} is revision {found}, the config expects {cfg['model_revision']}")
    manifest = paths.DATA_DIR / "manifest.json"
    return {
        "git_commit": wl._git_commit(),
        "git_dirty": wl._git_dirty(),
        "data_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
        "model_revision": found,
        "prompt_version": PROMPT_VERSION,
        "dedupe_scheme": DEDUPE_SCHEME,
    }


def answer_logits(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits that predict the answer letter (B, vocab) and the letter targets (B,).

    Inputs are left-padded, so the answer sits near the end of every row; only the last few positions'
    logits are computed (logits_to_keep), which avoids materializing (B, seq, vocab).
    """
    labels = batch["labels"]
    rows = torch.arange(labels.shape[0], device=labels.device)
    width = labels.shape[1]
    pos = (labels != td.IGNORE_INDEX).float().argmax(dim=1)  # the single supervised position per row
    keep = width - int(pos.min()) + 1
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                logits_to_keep=keep, use_cache=False)
    return out.logits[rows, pos - 1 - (width - keep)], labels[rows, pos]


def batch_stats(logits: torch.Tensor, batch: dict, targets: torch.Tensor) -> dict:
    """Answer accuracy (argmax over the options), p(correct letter) and letter mass over the full vocabulary."""
    probs = logits.float().softmax(dim=-1)
    rows = torch.arange(len(targets), device=targets.device)
    option_probs = probs.gather(1, batch["option_ids"]) * batch["option_valid"]
    pred = option_probs.masked_fill(~batch["option_valid"], -1).argmax(dim=1)
    return {
        "correct": int((pred == batch["answer_index"]).sum()),
        "p_correct": float(probs[rows, targets].sum()),
        "letter_mass": float(option_probs.sum()),
        "preds": [letters[i] for letters, i in zip(batch["letters"], pred.tolist())],
    }


class _Interval:
    """Training metrics accumulated between two log points."""

    def __init__(self):
        self.steps = self.loss = self.grad_norm = self.clipped = 0.0
        self.n = self.correct = self.p_correct = self.letter_mass = 0.0
        self.pad_frac = self.micro_batches = 0.0
        self.preds = Counter()

    def metrics(self) -> dict:
        total = sum(self.preds.values())
        return {
            "loss": self.loss / self.steps,
            "grad_norm": self.grad_norm / self.steps,
            "clip_frac": self.clipped / self.steps,
            "answer_acc": self.correct / self.n,
            "p_correct": self.p_correct / self.n,
            "letter_mass": self.letter_mass / self.n,
            "pad_frac": self.pad_frac / self.micro_batches,
            **{f"pred_frac_{k}": v / total for k, v in sorted(self.preds.items())},
        }


def _rng_state() -> dict:
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def run_eval(model, tokenizer, examples: list[dict], batch_size: int) -> dict:
    """Evaluate the in-memory model through the wrapper (no reload).

    The wrapper switches the model to eval mode for scoring and restores train mode afterwards.
    """
    wrapper = QwenWrapper(prompt_fn=build_messages, model=model, tokenizer=tokenizer)
    return evaluate(wrapper, examples, batch_size=batch_size)


def train(cfg: dict, *, model=None, tokenizer=None, train_examples=None, eval_examples=None,
          optimizer_cls=None) -> dict:
    """Train (or resume) one run. Returns a summary dict.

    model / tokenizer / examples / optimizer_cls can be injected for CPU tests; by default the 4-bit base model,
    the configured train and eval splits, and bitsandbytes' paged 8-bit AdamW are used.
    """
    from peft import LoraConfig, get_peft_model
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import get_constant_schedule_with_warmup

    session_start = time.time()
    out = output_dir(cfg)
    if (out / DONE_FILE).exists():
        raise RunAlreadyComplete(f"{out} is complete ({DONE_FILE} exists); use a new run_name to train again")

    if model is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Training needs a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{torch.cuda.get_device_name()} does not support bf16, which training requires "
                               "(4-bit NF4 with bf16 compute). Use an Ampere-or-newer GPU (L4, A6000, A100, ...).")
    gpu_name = wl._gpu_name()
    tag = gpu_tag(gpu_name)
    hourly = resolve_gpu_hourly_usd(cfg, gpu_name)
    model_path = Path(cfg["model_path"] or DEFAULT_MODEL_PATH)
    prov = provenance(cfg, model_path if model is None else None)

    # ---- data: tokenize everything, then fix the whole step-by-step plan up front
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if model is None:
        model, tokenizer = load_base_model(model_path, device_map={"": torch.cuda.current_device()})
    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if train_examples is None:
        train_examples = td.load_train_examples(cfg["train_datasets"], cfg["train_limit"])
    items = td.tokenize_examples(train_examples, tokenizer, cfg["max_seq_len"])
    lengths = [len(x["input_ids"]) for x in items]
    epoch_tokens = sum(lengths)
    plan = td.build_plan(lengths, cfg)
    plan_sha = td.plan_fingerprint(plan)
    per_epoch = cfg["checkpoints_per_epoch"]
    ckpt_targets = td.event_targets(1 / per_epoch if per_epoch else None, cfg["epochs"], epoch_tokens)
    if eval_examples is None and cfg["eval_split"]:
        eval_examples = td.load_eval_examples(cfg["eval_split"], cfg["eval_limit"])
    eval_targets = td.event_targets(cfg["eval_every_epochs"], cfg["epochs"], epoch_tokens) if eval_examples else []

    # ---- model: frozen 4-bit base + LoRA on every linear layer
    model.config.use_cache = False
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha_ratio"] * cfg["lora_rank"], lora_dropout=cfg["lora_dropout"],
        target_modules=list(cfg["lora_target_modules"]), bias="none", task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_cls is None:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.PagedAdamW8bit
    optimizer = optimizer_cls(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = get_constant_schedule_with_warmup(optimizer, cfg["warmup_steps"])
    device = next(p.device for p in params)

    state = {"step": 0, "tokens_seen": 0, "examples_seen": 0, "gpu_hours": 0.0, "train_hours": 0.0,
             "wandb_run_id": None, "resume_count": 0,
             "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    resume_path = out / RESUME_FILE
    if resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if saved["plan_sha256"] != plan_sha:
            raise RuntimeError(f"{resume_path} was written for a different data plan; refusing to resume")
        changed = {k for k in cfg.keys() | saved["config"].keys() if cfg.get(k) != saved["config"].get(k)}
        if changed - RESUMABLE_CHANGES:
            raise RuntimeError(f"Config differs from the run being resumed in {sorted(changed - RESUMABLE_CHANGES)}")
        set_peft_model_state_dict(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        _set_rng_state(saved["rng"])
        state = saved["state"]
        state["resume_count"] += 1
        print(f"[train] resuming {out} at step {state['step']}/{len(plan)} "
              f"({state['tokens_seen']:,} tokens, resume #{state['resume_count']})")
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "config.json", cfg)
    gpu_hours_before, session_tokens_start = state["gpu_hours"], state["tokens_seen"]
    session_train_hours = 0.0

    def gpu_hours() -> float:
        return gpu_hours_before + (time.time() - session_start) / 3600

    def save_resume() -> None:
        state["gpu_hours"] = gpu_hours()
        _atomic_save({"adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "rng": _rng_state(), "state": state,
                      "config": cfg, "plan_sha256": plan_sha}, resume_path)

    def metadata(final: bool) -> dict:
        started = datetime.datetime.fromisoformat(state["started_utc"])
        now = datetime.datetime.now(datetime.timezone.utc)
        return {
            "step": state["step"], "total_steps": len(plan), "final": final,
            "epoch": state["tokens_seen"] / epoch_tokens, "tokens_seen": state["tokens_seen"],
            "examples_seen": state["examples_seen"], "gpu_hours": gpu_hours(), "train_hours": state["train_hours"],
            "wall_clock_hours": (now - started).total_seconds() / 3600, "saved_utc": now.isoformat(timespec="seconds"),
            "gpu_name": gpu_name, "wandb_run_id": state["wandb_run_id"], "resume_count": state["resume_count"],
            "epoch_tokens": epoch_tokens, "plan_sha256": plan_sha, **prov, "config": cfg,
        }

    # Slurm sends SIGTERM before the time limit (see scripts/slurm): finish the step, save, and exit.
    stop = {"signal": None}
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            old_handlers[sig] = signal.signal(sig, lambda s, _f: stop.update(signal=s))

    run_config = {**cfg, "provenance": prov, "gpu_tag": tag, "epoch_tokens": epoch_tokens,
                  "n_train_examples": len(items), "total_steps": len(plan), "plan_sha256": plan_sha,
                  "n_eval_examples": len(eval_examples or [])}
    try:
        with wl.start_run(f"{output_name(cfg)}-{tag}", run_config, group=cfg["wandb_group"],
                          tags=[*cfg["wandb_tags"], tag], job_type="train", gpu_hourly_usd=hourly,
                          run_id=state["wandb_run_id"], step=state["step"],
                          tokens_seen=state["tokens_seen"]) as logger:
            state["wandb_run_id"] = logger.run.id
            ci = sum(t <= state["tokens_seen"] for t in ckpt_targets)  # events before the resume point are done
            ei = sum(t <= state["tokens_seen"] for t in eval_targets)
            interval, last_resume_save = _Interval(), time.time()
            result = None
            model.train()
            for step_index in range(state["step"], len(plan)):
                t0 = time.time()
                step = plan[step_index]
                n_examples = sum(len(mb) for mb in step["micro_batches"])
                lr = scheduler.get_last_lr()[0]
                step_loss = 0.0
                for mb in step["micro_batches"]:
                    batch = td.collate([items[i] for i in mb], pad_id)
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    logits, targets = answer_logits(model, batch)
                    loss = F.cross_entropy(logits.float(), targets, reduction="sum") / n_examples
                    loss.backward()
                    step_loss += float(loss.detach())
                    with torch.no_grad():
                        s = batch_stats(logits.detach(), batch, targets)
                    real = int(batch["attention_mask"].sum())
                    state["tokens_seen"] += real
                    interval.pad_frac += 1 - real / batch["attention_mask"].numel()
                    interval.micro_batches += 1
                    interval.correct += s["correct"]
                    interval.p_correct += s["p_correct"]
                    interval.letter_mass += s["letter_mass"]
                    interval.preds.update(s["preds"])
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state["step"] = step_index + 1
                state["examples_seen"] += n_examples
                dt = (time.time() - t0) / 3600
                state["train_hours"] += dt
                session_train_hours += dt
                interval.steps += 1
                interval.n += n_examples
                interval.loss += step_loss
                interval.grad_norm += grad_norm
                interval.clipped += grad_norm > cfg["max_grad_norm"]

                final = state["step"] == len(plan)
                tokens = state["tokens_seen"]
                if state["step"] % cfg["log_every_steps"] == 0 or final:
                    wl.log_train(state["step"], tokens, **interval.metrics(), lr=lr, epoch=tokens / epoch_tokens,
                                 resume_count=state["resume_count"])
                    interval = _Interval()

                ckpt_due = final or (ci < len(ckpt_targets) and tokens >= ckpt_targets[ci])
                eval_due = bool(eval_examples) and (final or (ei < len(eval_targets) and tokens >= eval_targets[ei]))
                while ci < len(ckpt_targets) and tokens >= ckpt_targets[ci]:
                    ci += 1
                while ei < len(eval_targets) and tokens >= eval_targets[ei]:
                    ei += 1
                if eval_due:
                    result = run_eval(model, tokenizer, eval_examples, cfg["eval_batch_size"])
                    wl.log_eval(result, step=state["step"], tokens_seen=tokens)
                    with open(out / "evals.jsonl", "a") as f:
                        f.write(json.dumps({"step": state["step"], "tokens_seen": tokens,
                                            "epoch": tokens / epoch_tokens, "gpu_hours": gpu_hours(),
                                            "split": cfg["eval_split"], **result}) + "\n")
                    print(f"[train] step {state['step']} eval {result}")
                if ckpt_due:
                    folder = out / f"step-{state['step']:06d}"
                    tmp = out / f"{folder.name}.tmp"
                    shutil.rmtree(tmp, ignore_errors=True)
                    model.save_pretrained(tmp)
                    _write_json(tmp / "metadata.json", metadata(final))
                    shutil.rmtree(folder, ignore_errors=True)
                    os.replace(tmp, folder)
                    index = len(list(out.glob("step-??????")))
                    logger.run.log({"train/step": state["step"], "train/tokens_seen": tokens,
                                    "train/checkpoint_index": index, "train/checkpoint_gpu_hours": gpu_hours()})
                    print(f"[train] kept checkpoint {folder} (step {state['step']}, {tokens:,} tokens, "
                          f"{gpu_hours():.3f} GPU-h)")
                if (ckpt_due or eval_due or stop["signal"]
                        or time.time() - last_resume_save >= 60 * cfg["resume_every_minutes"]):
                    save_resume()
                    last_resume_save = time.time()
                if stop["signal"] and not final:
                    raise SystemExit(f"[train] stopped by signal {stop['signal']} at step {state['step']}; "
                                     "resume state saved, re-run the same command to continue")

            session_tokens = state["tokens_seen"] - session_tokens_start
            summary = {
                "run": output_name(cfg), "output_dir": str(out), "steps": state["step"],
                "tokens_seen": state["tokens_seen"], "epochs": state["tokens_seen"] / epoch_tokens,
                "gpu_hours": gpu_hours(), "train_hours": state["train_hours"], "gpu_name": gpu_name,
                "train_tokens_per_sec": session_tokens / (session_train_hours * 3600) if session_train_hours else None,
                "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
                "resume_count": state["resume_count"], "wandb_run_id": state["wandb_run_id"],
                "final_eval": result,
            }
            for key in ("train_tokens_per_sec", "train_hours", "resume_count"):
                logger.run.summary[key] = summary[key]
        _write_json(out / DONE_FILE, summary)
        resume_path.unlink(missing_ok=True)  # the run is complete; the optimizer state is no longer needed
        print(f"[train] done: {json.dumps(summary)}")
        return summary
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        del model, optimizer, scheduler, params
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def output_name(cfg: dict) -> str:
    """Run name for W&B (the GPU type is appended): run_name, or the output folder's name."""
    return cfg.get("run_name") or output_dir(cfg).name
