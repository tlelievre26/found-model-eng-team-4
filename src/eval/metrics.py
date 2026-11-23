"""Metrics over variable-option probability dictionaries; no model required."""
import math


def compute_metrics(probabilities, answers, n_bins=15):
    """Return accuracy, natural-log NLL and top-label equal-width ECE.

    Bins are (i/B, (i+1)/B], with confidence 0 in the first bin.
    Zero gold probability gives infinite NLL, rather than silently clipping it.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if len(probabilities) != len(answers) or not answers:
        raise ValueError("Require equal, non-empty probabilities and answers")
    counts, hits, confidence_sums = [0]*n_bins, [0]*n_bins, [0.0]*n_bins
    correct, losses = 0, []
    for probs, answer in zip(probabilities, answers):
        if not probs or answer not in probs:
            raise ValueError("Gold answer must be in the supplied options")
        if any(not math.isfinite(p) or p < 0 or p > 1 for p in probs.values()):
            raise ValueError("Probabilities must be finite and in [0, 1]")
        if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5):
            raise ValueError("Option probabilities must sum to one")
        prediction = max(sorted(probs), key=probs.get)
        confidence = probs[prediction]
        hit = int(prediction == answer)
        correct += hit
        losses.append(-math.log(probs[answer]) if probs[answer] > 0 else math.inf)
        i = min(max(math.ceil(confidence*n_bins) - 1, 0), n_bins-1)
        counts[i] += 1
        hits[i] += hit
        confidence_sums[i] += confidence
    n = len(answers)
    ece = sum(abs(hits[i]-confidence_sums[i]) for i in range(n_bins)) / n
    return {"accuracy": correct/n, "nll": sum(losses)/n, "ece": ece, "n": n}
