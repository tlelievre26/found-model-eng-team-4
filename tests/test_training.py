"""CPU tests for the training pipeline: data plan, label masking, configs, schedules, and an exact resume."""
import json
import random
from pathlib import Path

import pytest
import torch

from src.training import data as td
from src.training.config import apply_overrides, load_config, parse_overrides

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class CharTokenizer:
    """One token per character, with a chat template shaped like Qwen's (prompt, then answer, then end tokens)."""
    pad_token_id = 0
    eos_token_id = 1
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}|\n" for m in messages)
        return text + "<assistant>" if add_generation_prompt else text

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, texts, add_special_tokens=False, padding=False, return_tensors=None):
        rows = [self.encode(t) for t in texts]
        if not padding:
            return {"input_ids": rows}
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([[0] * (width - len(r)) + r for r in rows]),
                "attention_mask": torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows])}


def make_examples(n, seed=0):
    rng = random.Random(seed)
    return [{"id": f"ex-{i:03d}", "question": "q" * rng.randint(1, 30),
             "options": {k: f"opt {k}" for k in "ABCD"}, "answer": rng.choice("ABCD")} for i in range(n)]


# ---------------------------------------------------------------- budget and order

def test_budget_prefix_is_deterministic_shortest_prefix():
    lengths = [random.Random(i).randint(20, 200) for i in range(500)]
    plan = td.select_budget(lengths, 1.3, seed=7)
    assert plan == td.select_budget(lengths, 1.3, seed=7)
    assert len(plan) == 2 and sorted(plan[0]) == list(range(500))
    order = td.epoch_order(500, 7, 1)
    prefix = plan[1]
    assert prefix == order[:len(prefix)]  # a prefix of epoch 1's shuffled order
    target = 0.3 * sum(lengths)
    assert sum(lengths[i] for i in prefix) >= target > sum(lengths[i] for i in prefix[:-1])


def test_fraction_only_and_whole_epochs():
    lengths = [10] * 100
    assert [len(e) for e in td.select_budget(lengths, 0.25, 0)] == [25]
    assert [len(e) for e in td.select_budget(lengths, 2.0, 0)] == [100, 100]


def test_each_epoch_reshuffles_with_seed_plus_epoch():
    plan = td.select_budget([5] * 50, 3.0, seed=1)
    assert plan[0] != plan[1] != plan[2]
    assert plan[2] == td.epoch_order(50, 1, 2) == td.epoch_order(50, 3, 0)


def test_length_grouping_is_deterministic_and_covers_each_example_once():
    lengths = [random.Random(i).randint(1, 100) for i in range(203)]
    indices = td.epoch_order(203, 0, 0)
    a = td.group_by_length(indices, lengths, 4, 8, random.Random("x"))
    assert a == td.group_by_length(indices, lengths, 4, 8, random.Random("x"))
    assert sorted(i for b in a for i in b) == list(range(203))
    assert all(len(b) <= 4 for b in a)
    # Within a chunk of 8 micro-batches the examples are sorted by length, so each batch spans a narrow range.
    first_chunk = sorted(indices[:32], key=lambda i: lengths[i])
    assert first_chunk[:4] in a


def test_plan_steps_respect_accumulation_and_epochs():
    cfg = {"epochs": 1.5, "seed": 0, "micro_batch_size": 4, "grad_accum_steps": 2, "group_chunk_batches": 2}
    plan = td.build_plan([10] * 30, cfg)
    assert all(len(s["micro_batches"]) <= 2 for s in plan)
    assert sum(len(mb) for s in plan for mb in s["micro_batches"]) == 30 + 15
    assert [s["epoch"] for s in plan] == sorted(s["epoch"] for s in plan)


# ---------------------------------------------------------------- tokenization and labels

def test_only_the_answer_letter_is_supervised():
    tok = CharTokenizer()
    examples = make_examples(5)
    items = td.tokenize_examples(examples, tok, max_seq_len=512)
    batch = td.collate(items, tok.pad_token_id)
    labels = batch["labels"]
    assert ((labels != td.IGNORE_INDEX).sum(dim=1) == 1).all()
    for row, ex in enumerate(examples):
        pos = int((labels[row] != td.IGNORE_INDEX).nonzero())
        assert int(labels[row, pos]) == ord(ex["answer"]) == int(batch["input_ids"][row, pos])
        # The tokens before the answer are exactly the scoring prompt (left-padded).
        prompt = tok.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True)
        assert prompt.endswith("<assistant>")
        assert bytes(batch["input_ids"][row, pos - len("<assistant>"):pos].tolist()).decode() == "<assistant>"
    assert (batch["attention_mask"].sum(1) == torch.tensor([len(x["input_ids"]) for x in items])).all()


def test_too_long_examples_fail_instead_of_truncating():
    with pytest.raises(ValueError, match="longer than max_seq_len"):
        td.tokenize_examples(make_examples(3), CharTokenizer(), max_seq_len=20)


# ---------------------------------------------------------------- configs and schedules

def test_config_overrides():
    cfg = load_config(CONFIGS / "smoke.yaml", ["lora_rank=32", "learning_rate=5e-5", "eval_split=null",
                                               "train_datasets=[medmcqa, medqa]", "epochs=2"])
    assert cfg["lora_rank"] == 32 and cfg["learning_rate"] == 5e-5 and isinstance(cfg["learning_rate"], float)
    assert cfg["eval_split"] is None and cfg["train_datasets"] == ["medmcqa", "medqa"]
    assert cfg["epochs"] == 2.0 and cfg["run_name"] == "smoke" and cfg["max_seq_len"] == 512  # from base.yaml
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["lora_rnak=8"])
    with pytest.raises(ValueError):
        parse_overrides(["lora_rank"])
    for name in ("main.yaml", "base.yaml"):
        assert load_config(CONFIGS / name).keys() == cfg.keys()


def test_event_schedule():
    t = 1000
    assert td.event_targets(1 / 2, 1.5, t) == [500, 1000, 1500]  # checkpoints_per_epoch=2
    assert td.event_targets(1.0, 6.0, t) == [1000 * j for j in range(1, 7)]
    assert td.event_targets(0.125, 0.25, t) == [125, 250]  # sweep: hp_dev at the midpoint and the end
    assert td.event_targets(None, 1.0, t) == []
    assert len(td.event_targets(1 / 3, 1.0, t)) == 3


# ---------------------------------------------------------------- end to end on a tiny random Qwen3

class CrashingAdamW(torch.optim.AdamW):
    crash_at = None
    calls = 0

    def step(self, *args, **kwargs):
        type(self).calls += 1
        if type(self).calls == self.crash_at:
            raise RuntimeError("simulated crash")
        return super().step(*args, **kwargs)


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                        tie_word_embeddings=True))


def run(cfg, crash_at=None):
    from src.training.train import train
    CrashingAdamW.calls, CrashingAdamW.crash_at = 0, crash_at
    return train(cfg, model=tiny_model(), tokenizer=CharTokenizer(), train_examples=make_examples(40),
                 eval_examples=make_examples(8, seed=1), optimizer_cls=CrashingAdamW)


def test_train_end_to_end_and_exact_resume(tmp_path, monkeypatch):
    from safetensors.torch import load_file
    from src.training.train import RunAlreadyComplete
    monkeypatch.setenv("WANDB_MODE", "disabled")
    cfg = load_config(CONFIGS / "base.yaml", {
        "run_name": "t", "epochs": 1.5, "micro_batch_size": 4, "grad_accum_steps": 2, "group_chunk_batches": 2,
        "warmup_steps": 2, "learning_rate": 1e-2, "lora_rank": 4, "checkpoints_per_epoch": 2,
        "eval_split": "fake:hp_dev", "eval_every_epochs": 0.5, "resume_every_minutes": 0, "log_every_steps": 1,
        "gpu_hourly_usd": 0, "eval_batch_size": 4})

    straight = run({**cfg, "output_dir": str(tmp_path / "a")})
    with pytest.raises(RuntimeError, match="simulated crash"):
        run({**cfg, "output_dir": str(tmp_path / "b")}, crash_at=5)
    assert (tmp_path / "b" / "resume.pt").exists() and not (tmp_path / "b" / "DONE").exists()
    resumed = run({**cfg, "output_dir": str(tmp_path / "b")})
    assert resumed["resume_count"] == 1 and resumed["steps"] == straight["steps"] == 8
    assert resumed["tokens_seen"] == straight["tokens_seen"]

    ckpts = sorted(p.name for p in (tmp_path / "a").glob("step-*"))
    assert len(ckpts) == 3 and ckpts == sorted(p.name for p in (tmp_path / "b").glob("step-*"))
    for name in ckpts:
        a, b = (load_file(tmp_path / d / name / "adapter_model.safetensors") for d in "ab")
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        meta = json.loads((tmp_path / "a" / name / "metadata.json").read_text())
        for key in ("step", "epoch", "tokens_seen", "gpu_hours", "wall_clock_hours", "gpu_name", "wandb_run_id",
                    "config", "git_commit", "git_dirty", "data_manifest_sha256", "model_revision", "prompt_version"):
            assert key in meta
    evals = [[json.loads(x)["nll"] for x in (tmp_path / d / "evals.jsonl").read_text().splitlines()] for d in "ab"]
    assert len(evals[0]) == 3 and evals[0] == evals[1]
    with pytest.raises(RunAlreadyComplete):
        run({**cfg, "output_dir": str(tmp_path / "a")})


def test_sweep_trials_order_and_folders():
    from scripts.run_sweep import sweep_trials, trial_name
    from src.utils import paths
    assert trial_name({"learning_rate": 5e-5, "lora_rank": 8}) == "lr5e-05_r8"
    sweep, sweep_dir, trials = sweep_trials(CONFIGS / "sweep.yaml", ["gpu_hourly_usd=0"])
    assert sweep_dir == paths.CHECKPOINTS_DIR / "sweep-v1"
    assert [n for n, _ in trials][:4] == ["lr5e-05_r8", "lr5e-05_r16", "lr5e-05_r32", "lr0.0001_r8"]
    assert len(trials) == 9
    _, cfg = trials[0]
    assert cfg["epochs"] == 0.25 and cfg["eval_split"] == "medmcqa:hp_dev" and cfg["eval_every_epochs"] == 0.125
    assert cfg["output_dir"] == str(sweep_dir / "lr5e-05_r8") and cfg["wandb_group"] == "sweep-v1"
    assert cfg["gpu_hourly_usd"] == 0 and cfg["warmup_steps"] == 100 and "sweep" in cfg["wandb_tags"]
