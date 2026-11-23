"""STUB evaluation, to be replaced by the #12 harness.

It has #12's signature, ``evaluate(model, examples, batch_size=...) -> {"accuracy", "nll", "ece", "n"}``,
so training (src/training/train.py) and the sweep (scripts/run_sweep.py) can switch to the real harness
by changing one import. It does not save per-question outputs.
"""

from __future__ import annotations

import math


def evaluate(model, examples, batch_size: int = 16, n_bins: int = 15) -> dict:
    """Accuracy, mean NLL of the correct letter, and ECE from ``model.score_batch`` (a QwenWrapper)."""
    if not examples:
        raise ValueError("evaluate needs at least one example")
    results = model.score_batch(list(examples), batch_size=batch_size)
    probs = [r["probabilities"] for r in results]
    answers = [ex["answer"] for ex in examples]
    return {**metrics(probs, answers, n_bins=n_bins), "n": len(examples)}


def metrics(probs: list[dict], answers: list[str], n_bins: int = 15) -> dict:
    """Accuracy, mean NLL and ECE (equal-width confidence bins, Guo et al. 2017) from letter probabilities."""
    correct = [max(p, key=p.get) == a for p, a in zip(probs, answers)]
    nll = sum(-math.log(max(p[a], 1e-12)) for p, a in zip(probs, answers)) / len(probs)
    conf = [max(p.values()) for p in probs]
    bins: dict[int, list[int]] = {}
    for i, c in enumerate(conf):
        # Bins are (k/n, (k+1)/n]; a confidence of exactly 0 goes into the first bin.
        bins.setdefault(min(max(math.ceil(c * n_bins) - 1, 0), n_bins - 1), []).append(i)
    ece = sum(len(idx) / len(probs) * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
              for idx in bins.values())
    return {"accuracy": sum(correct) / len(correct), "nll": nll, "ece": ece}
