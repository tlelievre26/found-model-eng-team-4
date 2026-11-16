"""Weights & Biases logging helpers for training and evaluation runs.

Every run reports to the shared W&B project with its config, git commit, GPU
type, wall-clock time, peak GPU memory, and estimated dollar cost.

Typical use from a training script::

    from src.utils import wandb_logging as wl

    with wl.start_run("lora-r16-lr2e-4", config):
        tokens_seen = 0
        for step, batch in enumerate(loader, start=1):
            loss = train_step(batch)
            tokens_seen += int(batch["attention_mask"].sum())
            if step % 10 == 0:
                wl.log_train(step, tokens_seen, loss=loss.item(), lr=sched.get_last_lr()[0])
            if step % 500 == 0:
                wl.log_eval({"accuracy": evaluate(model)})

Leaving the ``with`` block calls :func:`finish_run`, which writes the final
time / memory / cost summary, even if training crashes. You can also call
``start_run`` and ``finish_run`` yourself instead of using ``with``.

Every training metric is plotted twice in W&B:

* ``train/<metric>`` against the optimizer step (``train/step``)
* ``train_vs_tokens/<metric>`` against tokens processed (``train/tokens_seen``)

so runs with different batch sizes or sequence lengths can be compared by
compute. Eval metrics are split the same way (``eval/`` and ``eval_vs_tokens/``).
GPU memory, throughput, elapsed time, and running cost go under ``perf/``.

Set ``WANDB_MODE=disabled`` to turn W&B off for quick local tests: every
function still works and :func:`finish_run` still prints the summary.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import wandb

# Shared W&B project. Override with the WANDB_PROJECT / WANDB_ENTITY env vars.
DEFAULT_PROJECT = "found-model-eng-team-4"
# Hourly price of the team VM (L4 GPU), used to estimate run cost.
DEFAULT_GPU_HOURLY_USD = 0.71

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BYTES_PER_GB = 1024**3

_active: RunLogger | None = None


class RunLogger:
    """Tracks one W&B run. Usually created with :func:`start_run`."""

    def __init__(
        self,
        name: str,
        config: Mapping[str, Any] | Any,
        *,
        project: str | None = None,
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        job_type: str = "train",
        gpu_hourly_usd: float = DEFAULT_GPU_HOURLY_USD,
    ) -> None:
        self.gpu_hourly_usd = gpu_hourly_usd
        self.gpu_name = _gpu_name()
        self.step = 0
        self.tokens_seen = 0
        self._finished = False

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        full_config = {
            **_config_to_dict(config),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "gpu_name": self.gpu_name,
            "gpu_hourly_usd": gpu_hourly_usd,
        }
        self.run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
            entity=entity or os.environ.get("WANDB_ENTITY"),
            name=name,
            group=group,
            tags=tags,
            job_type=job_type,
            config=full_config,
        )
        self._define_axes()

        self._start_time = time.perf_counter()
        self._last_log_time = self._start_time
        self._last_log_tokens = 0

    def _define_axes(self) -> None:
        """Tell W&B which x-axis each metric family is plotted against."""
        self.run.define_metric("train/step")
        self.run.define_metric("train/tokens_seen")
        self.run.define_metric("train/*", step_metric="train/step")
        self.run.define_metric("perf/*", step_metric="train/step")
        self.run.define_metric("eval/*", step_metric="train/step")
        self.run.define_metric("train_vs_tokens/*", step_metric="train/tokens_seen")
        self.run.define_metric("eval_vs_tokens/*", step_metric="train/tokens_seen")

    # ------------------------------------------------------------------ logging

    def log_train(self, step: int, tokens_seen: int, **metrics: float) -> None:
        """Log training metrics (loss, lr, grad norm, ...) for one step.

        Args:
            step: Optimizer step (counting from 1 is fine; must not decrease).
            tokens_seen: Total tokens processed so far in this run (cumulative,
                not per batch). Count real tokens, not padding, e.g. the sum of
                ``attention_mask``.
            **metrics: Scalar metrics, e.g. ``loss=..., lr=...``.
        """
        now = time.perf_counter()
        dt = now - self._last_log_time
        d_tokens = tokens_seen - self._last_log_tokens
        self.step, self.tokens_seen = step, tokens_seen

        data: dict[str, Any] = {"train/step": step, "train/tokens_seen": tokens_seen}
        for key, value in metrics.items():
            value = _to_scalar(value)
            data[f"train/{key}"] = value
            data[f"train_vs_tokens/{key}"] = value

        data.update(self._perf_metrics(now))
        if dt > 0 and d_tokens >= 0:
            data["perf/tokens_per_sec"] = d_tokens / dt

        self.run.log(data)
        self._last_log_time = now
        self._last_log_tokens = tokens_seen

    def log_eval(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
        tokens_seen: int | None = None,
    ) -> None:
        """Log evaluation metrics (accuracy, abstention rate, ...).

        ``step`` and ``tokens_seen`` default to the values from the most recent
        :meth:`log_train` call, so evals run mid-training line up with training
        curves. For a standalone eval run they are simply 0.
        """
        if step is not None:
            self.step = step
        if tokens_seen is not None:
            self.tokens_seen = tokens_seen

        data: dict[str, Any] = {"train/step": self.step, "train/tokens_seen": self.tokens_seen}
        for key, value in metrics.items():
            value = _to_scalar(value)
            data[f"eval/{key}"] = value
            data[f"eval_vs_tokens/{key}"] = value
        self.run.log(data)

    def summary(self) -> dict[str, Any]:
        """Time / memory / cost totals for the run so far."""
        hours = self.elapsed_hours()
        return {
            "gpu_name": self.gpu_name,
            "wall_clock_hours": hours,
            "peak_gpu_mem_gb": _peak_mem_gb(),
            "est_cost_usd": hours * self.gpu_hourly_usd,
            "final_step": self.step,
            "total_tokens_seen": self.tokens_seen,
        }

    def finish(self, exit_code: int = 0) -> dict[str, Any]:
        """Write the final summary to W&B and close the run. Safe to call twice."""
        summary = self.summary()
        if self._finished:
            return summary
        self._finished = True
        for key, value in summary.items():
            self.run.summary[key] = value
        self.run.finish(exit_code=exit_code)
        _print_summary(summary)
        return summary

    def elapsed_hours(self) -> float:
        return (time.perf_counter() - self._start_time) / 3600

    def _perf_metrics(self, now: float) -> dict[str, float]:
        hours = (now - self._start_time) / 3600
        data = {
            "perf/elapsed_hours": hours,
            "perf/est_cost_usd": hours * self.gpu_hourly_usd,
        }
        if torch.cuda.is_available():
            data["perf/gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / _BYTES_PER_GB
            data["perf/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / _BYTES_PER_GB
            data["perf/gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / _BYTES_PER_GB
        return data

    # ------------------------------------------------------- context manager

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _active
        if _active is self:
            _active = None
        self.finish(exit_code=0 if exc_type is None else 1)


# --------------------------------------------------------------------------
# Module-level API: one active run per process.


def start_run(
    name: str,
    config: Mapping[str, Any] | Any,
    **kwargs: Any,
) -> RunLogger:
    """Start a W&B run that records ``config`` plus git commit and GPU info.

    Args:
        name: Run name shown in W&B, e.g. ``"lora-r16-lr2e-4"``.
        config: Run settings, as a dict or a dataclass.
        **kwargs: Optional ``project``, ``entity``, ``group``, ``tags``,
            ``job_type`` (``"train"`` or ``"eval"``), ``gpu_hourly_usd``.

    Returns:
        The :class:`RunLogger`, which can be used as a context manager.
    """
    global _active
    if _active is not None and not _active._finished:
        raise RuntimeError("A W&B run is already active; call finish_run() first.")
    _active = RunLogger(name, config, **kwargs)
    return _active


def log_train(step: int, tokens_seen: int, **metrics: float) -> None:
    """Log training metrics for the active run. See :meth:`RunLogger.log_train`."""
    _require_active().log_train(step, tokens_seen, **metrics)


def log_eval(
    metrics: Mapping[str, float],
    step: int | None = None,
    tokens_seen: int | None = None,
) -> None:
    """Log evaluation metrics for the active run. See :meth:`RunLogger.log_eval`."""
    _require_active().log_eval(metrics, step=step, tokens_seen=tokens_seen)


def finish_run(exit_code: int = 0) -> dict[str, Any]:
    """Log the final time / memory / cost summary and close the active run."""
    global _active
    run = _require_active()
    _active = None
    return run.finish(exit_code=exit_code)


def _require_active() -> RunLogger:
    if _active is None:
        raise RuntimeError("No active W&B run; call start_run() first.")
    return _active


# --------------------------------------------------------------------------
# Helpers


def _config_to_dict(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if config is None:
        return {}
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError(f"config must be a dict or dataclass, got {type(config).__name__}")


def _to_scalar(value: Any) -> Any:
    """Turn 0-d tensors into Python numbers so they log cleanly."""
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return value


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _git_commit() -> str | None:
    return _git("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


def _gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return "cpu"


def _peak_mem_gb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / _BYTES_PER_GB
    return None


def _print_summary(summary: Mapping[str, Any]) -> None:
    peak = summary["peak_gpu_mem_gb"]
    peak_str = "n/a" if peak is None else f"{peak:.2f} GB"
    print(
        f"[wandb_logging] {summary['gpu_name']} | "
        f"{summary['wall_clock_hours'] * 60:.1f} min | "
        f"peak mem {peak_str} | "
        f"est. cost ${summary['est_cost_usd']:.2f} | "
        f"{summary['total_tokens_seen']:,} tokens"
    )
