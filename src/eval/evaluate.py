"""Evaluation callable shared by the training sweep and standalone CLI."""
import json
from pathlib import Path

from .metrics import compute_metrics


def evaluate(model, examples, batch_size=8, *, n_bins=15, output_path=None,
             metadata=None):
    """Score any common-format split; caller owns W&B run lifetime.

    Output JSONL contains predictions, not question text. A metadata sidecar
    identifies the model/data build. Existing outputs are never overwritten.
    """
    examples = list(examples)
    if not examples:
        raise ValueError("Cannot evaluate an empty split")
    path = Path(output_path) if output_path is not None else None
    sidecar = path.with_suffix(path.suffix + ".meta.json") if path else None
    if path and (path.exists() or sidecar.exists()):
        raise FileExistsError(f"Evaluation output already exists: {path}")
    results = model.score_batch(examples, batch_size=batch_size)
    if len(results) != len(examples):
        raise ValueError("Wrapper returned the wrong number of predictions")
    probabilities, answers, records = [], [], []
    for ex, result in zip(examples, results):
        probs = result["probabilities"]
        if set(probs) != set(ex["options"]):
            raise ValueError("Wrapper options do not match the example")
        prediction = max(sorted(probs), key=probs.get)
        if result["prediction"] != prediction:
            raise ValueError("Wrapper prediction does not match probabilities")
        probabilities.append(probs)
        answers.append(ex["answer"])
        records.append({"id": ex["id"], "source": ex.get("source"),
                        "probabilities": probs, "prediction": prediction,
                        "answer": ex["answer"], "confidence": probs[prediction],
                        "correct": prediction == ex["answer"]})
    metrics = compute_metrics(probabilities, answers, n_bins=n_bins)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = "".join(json.dumps(record, allow_nan=False) + "\n" for record in records)
        info = json.dumps({"metadata": metadata or {}, "n_bins": n_bins,
                           "n": metrics["n"]}, indent=2, allow_nan=False)
        created = []
        try:
            for dest, content in ((path, rows), (sidecar, info)):
                with dest.open("x", encoding="utf-8") as stream:
                    created.append(dest)
                    stream.write(content)
        except Exception:
            # Remove only files this call created; preserve existing output.
            for dest in created:
                dest.unlink(missing_ok=True)
            raise
    return metrics
