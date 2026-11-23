"""Training data: tokenization with an answer-letter-only label, the epoch budget, length grouping, batching.

Everything here is deterministic given the config, so the full step-by-step data plan is rebuilt identically
when a run resumes. Token counts are non-padding tokens (the project's compute measure, see CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import random

import torch

from src.data.medqa_datasets import build_messages, load_splits

IGNORE_INDEX = -100


def load_train_examples(names: list[str], limit: int | None = None) -> list[dict]:
    """Train splits of the named datasets, sorted by id (the order every shuffle starts from)."""
    rows = sorted((ex for name in names for ex in load_splits(name)["train"]), key=lambda ex: ex["id"])
    return rows[:limit] if limit else rows


def load_eval_examples(spec: str, limit: int | None = None) -> list[dict]:
    """Examples for a "dataset:split" spec such as "medmcqa:hp_dev", sorted by id, optionally the first `limit`."""
    name, split = spec.split(":")
    if split == "test":
        raise ValueError("The test split is for final reported numbers only; never evaluate on it during training")
    rows = sorted(load_splits(name)[split], key=lambda ex: ex["id"])
    return rows[:limit] if limit else rows


def letter_token_ids(tokenizer, letters: str = "ABCDEFGHIJ") -> dict[str, int]:
    """Token id of each bare answer letter; each must be a single token."""
    ids = {}
    for letter in letters:
        tok = tokenizer.encode(letter, add_special_tokens=False)
        if len(tok) != 1:
            raise ValueError(f"Answer letter {letter!r} is not a single token: {tok}")
        ids[letter] = tok[0]
    return ids


def tokenize_examples(examples: list[dict], tokenizer, max_seq_len: int) -> list[dict]:
    """Render each example with its answer and record where the answer letter is.

    The training text is the chat template of build_messages(ex, include_answer=True). It must start with the
    exact scoring prompt (build_messages(ex) + add_generation_prompt=True, what the wrapper scores), followed by
    the answer letter token, so the supervised token is the one the wrapper reads. Fails if anything would be
    truncated, rather than truncating.
    """
    def render(ex, answer):
        return tokenizer.apply_chat_template(build_messages(ex, include_answer=answer), tokenize=False,
                                             add_generation_prompt=not answer)

    prompts = tokenizer([render(ex, False) for ex in examples], add_special_tokens=False)["input_ids"]
    fulls = tokenizer([render(ex, True) for ex in examples], add_special_tokens=False)["input_ids"]
    letters = letter_token_ids(tokenizer)
    items, too_long = [], 0
    for ex, prompt, full in zip(examples, prompts, fulls):
        pos = len(prompt)
        if full[:pos] != prompt or len(full) <= pos or full[pos] != letters[ex["answer"]]:
            raise ValueError(f"{ex['id']}: training text is not the scoring prompt followed by the answer letter")
        too_long += len(full) > max_seq_len
        options = sorted(ex["options"])
        items.append({"input_ids": full, "label_pos": pos, "option_ids": [letters[x] for x in options],
                      "letters": options, "answer_index": options.index(ex["answer"])})
    if too_long:
        raise ValueError(f"{too_long} training examples are longer than max_seq_len={max_seq_len}; "
                         "raise max_seq_len (nothing is truncated)")
    return items


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    """Shuffled order of n examples for one epoch (epoch k uses seed + k)."""
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    return order


def select_budget(lengths: list[int], epochs: float, seed: int) -> list[list[int]]:
    """Example indices for each epoch of an `epochs` budget (a float).

    Whole epochs use every example. A fractional part takes the shortest prefix of that epoch's shuffled order
    whose token count reaches fraction x (tokens in one full epoch).
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    whole = math.floor(epochs)
    plan = [epoch_order(len(lengths), seed, k) for k in range(whole)]
    fraction = epochs - whole
    if fraction > 1e-9:
        order, target, total = epoch_order(len(lengths), seed, whole), fraction * sum(lengths), 0
        for i, idx in enumerate(order):
            total += lengths[idx]
            if total >= target:
                break
        plan.append(order[:i + 1])
    return plan


def group_by_length(indices: list[int], lengths: list[int], micro_batch_size: int, chunk_batches: int,
                    rng: random.Random) -> list[list[int]]:
    """Micro-batches of similar length: sort chunks of `chunk_batches` micro-batches (kept in the shuffled order)
    by length, cut them into micro-batches, then shuffle the micro-batch order."""
    chunk = micro_batch_size * chunk_batches
    batches = []
    for start in range(0, len(indices), chunk):
        part = sorted(indices[start:start + chunk], key=lambda i: lengths[i])
        batches += [part[j:j + micro_batch_size] for j in range(0, len(part), micro_batch_size)]
    rng.shuffle(batches)
    return batches


def build_plan(lengths: list[int], cfg: dict) -> list[dict]:
    """Every optimizer step of the run: {"epoch": k, "micro_batches": [[example indices], ...]}.

    Steps never cross an epoch boundary, so an epoch's last step may have fewer micro-batches.
    """
    steps = []
    for k, indices in enumerate(select_budget(lengths, cfg["epochs"], cfg["seed"])):
        batches = group_by_length(indices, lengths, cfg["micro_batch_size"], cfg["group_chunk_batches"],
                                  random.Random(f"batches-{cfg['seed']}-{k}"))
        for j in range(0, len(batches), cfg["grad_accum_steps"]):
            steps.append({"epoch": k, "micro_batches": batches[j:j + cfg["grad_accum_steps"]]})
    return steps


def split_by_tokens(batch: list[int], lengths: list[int], max_tokens: int | None) -> list[list[int]]:
    """Consecutive row chunks of a micro-batch whose padded size (rows x longest row) stays within max_tokens.

    Bounds activation memory: a wide micro-batch is run as several forward/backward passes. The gradient is
    unchanged, because the loss is summed per example and divided by the step's example count. A single row
    longer than max_tokens is its own chunk.
    """
    if not max_tokens:
        return [batch]
    chunks, current, width = [], [], 0
    for i in batch:
        if current and (len(current) + 1) * max(width, lengths[i]) > max_tokens:
            chunks.append(current)
            current, width = [], 0
        current.append(i)
        width = max(width, lengths[i])
    return chunks + [current] if current else chunks


def plan_fingerprint(plan: list[dict]) -> str:
    """SHA-256 of the data plan, stored with resume state to make sure a resumed run sees the same data."""
    return hashlib.sha256(json.dumps(plan, separators=(",", ":")).encode()).hexdigest()


def event_targets(every_epochs: float | None, total_epochs: float, epoch_tokens: int) -> list[float]:
    """Token counts at which a periodic event (kept checkpoint, eval) is due: every `every_epochs` of an epoch's
    tokens, up to the budget. The end of the run is always an event as well (handled by the training loop)."""
    if not every_epochs:
        return []
    n = math.floor(total_epochs / every_epochs + 1e-9)
    return [j * every_epochs * epoch_tokens for j in range(1, n + 1)]


def collate(items: list[dict], pad_id: int) -> dict:
    """Left-padded batch (as the wrapper scores). Labels are IGNORE_INDEX except at the answer letter."""
    width = max(len(x["input_ids"]) for x in items)
    input_ids, mask, labels = [], [], []
    for x in items:
        pad = width - len(x["input_ids"])
        input_ids.append([pad_id] * pad + x["input_ids"])
        mask.append([0] * pad + [1] * len(x["input_ids"]))
        row = [IGNORE_INDEX] * width
        row[pad + x["label_pos"]] = x["input_ids"][x["label_pos"]]
        labels.append(row)
    k = max(len(x["option_ids"]) for x in items)
    option_ids = [x["option_ids"] + [x["option_ids"][0]] * (k - len(x["option_ids"])) for x in items]
    option_valid = [[True] * len(x["option_ids"]) + [False] * (k - len(x["option_ids"])) for x in items]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(mask),
        "labels": torch.tensor(labels),
        "option_ids": torch.tensor(option_ids),
        "option_valid": torch.tensor(option_valid),
        "answer_index": torch.tensor([x["answer_index"] for x in items]),
        "letters": [x["letters"] for x in items],
    }
