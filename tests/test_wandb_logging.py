from dataclasses import dataclass

import pytest
import torch

from src.utils import wandb_logging as wl


@pytest.fixture(autouse=True)
def wandb_disabled(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    yield
    # Don't leak an active run into the next test.
    if wl._active is not None:
        wl.finish_run()


@dataclass
class Cfg:
    lr: float = 1e-4
    lora_r: int = 16


def test_run_without_wandb():
    run = wl.start_run("test", {"lr": 1e-4}, gpu_hourly_usd=1.0)
    assert run.run.config["lr"] == 1e-4
    assert "git_commit" in run.run.config
    assert run.run.config["gpu_hourly_usd"] == 1.0

    wl.log_train(1, 100, loss=torch.tensor(2.5), lr=1e-4)
    wl.log_train(2, 250, loss=2.0)
    wl.log_eval({"accuracy": 0.5})

    summary = wl.finish_run()
    assert summary["final_step"] == 2
    assert summary["total_tokens_seen"] == 250
    assert summary["est_cost_usd"] == pytest.approx(summary["wall_clock_hours"] * 1.0)
    assert wl._active is None


def test_context_manager_and_dataclass_config():
    with wl.start_run("test", Cfg()) as run:
        assert run.run.config["lora_r"] == 16
        wl.log_train(5, 1000, loss=1.0)
    assert wl._active is None
    assert run._finished


def test_context_manager_finishes_on_error():
    with pytest.raises(ValueError):
        with wl.start_run("test", {}) as run:
            raise ValueError("boom")
    assert wl._active is None
    assert run._finished


def test_eval_uses_latest_train_position():
    run = wl.start_run("test", {})
    wl.log_train(10, 5000, loss=1.0)
    wl.log_eval({"accuracy": 0.7})
    assert (run.step, run.tokens_seen) == (10, 5000)
    wl.finish_run()


def test_errors_without_active_run():
    with pytest.raises(RuntimeError):
        wl.log_train(1, 1, loss=1.0)
    with pytest.raises(RuntimeError):
        wl.finish_run()


def test_only_one_active_run():
    wl.start_run("a", {})
    with pytest.raises(RuntimeError):
        wl.start_run("b", {})
    wl.finish_run()


def test_resume_passes_run_id_and_position(monkeypatch):
    seen = {}
    real_init = wl.wandb.init

    def init(**kwargs):
        seen.update(kwargs)
        return real_init(**{k: v for k, v in kwargs.items() if k not in ("id", "resume")})
    monkeypatch.setattr(wl.wandb, "init", init)
    run = wl.start_run("test", {}, run_id="abc123", step=7, tokens_seen=900)
    assert seen["id"] == "abc123" and seen["resume"] == "allow"
    assert (run.step, run.tokens_seen, run._last_log_tokens) == (7, 900, 900)
    wl.finish_run()
