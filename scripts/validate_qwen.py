"""VM validation using local common-format JSONL and Issue #3's prompt function."""
import argparse
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.qwen import DEFAULT_MODEL_PATH, QwenWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="MedMCQA common-format validation JSONL")
    parser.add_argument("--prompt-function", required=True, help="Actual shared formatter: module:function")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--six-option-data", required=True, help="Common-format MedAbstain JSONL")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    module, name = args.prompt_function.split(":", 1)
    prompt_fn = getattr(importlib.import_module(module), name)
    examples = read_examples(args.data)[:args.limit]
    if not examples or any(x.get("answer") not in x["options"] for x in examples):
        raise ValueError("Validation examples require valid answer letters")
    six = next((x for x in read_examples(args.six_option_data) if len(x["options"]) == 6), None)
    if six is None:
        raise ValueError("Provide at least one real six-option MedAbstain question")
    if not torch.cuda.is_available():
        raise RuntimeError("Run VM validation on CUDA")
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    wrapper = QwenWrapper(args.model_path, prompt_fn=prompt_fn)
    results = wrapper.score_batch(examples, args.batch_size)
    six_result = wrapper.score(six)
    for result in results + [six_result]:
        if abs(sum(result["probabilities"].values()) - 1) > 1e-5:
            raise AssertionError("Probabilities do not sum to one")
    accuracy = sum(r["prediction"] == x["answer"] for r, x in zip(results, examples)) / len(examples)
    memory = []
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
        memory.append({"device": torch.cuda.get_device_name(device),
                       "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30})
    print(json.dumps({"n": len(examples), "accuracy": accuracy, "memory": memory,
                      "six_option_result": six_result}, indent=2))
    if accuracy <= .25:
        raise AssertionError("Accuracy is not above the 25% random baseline")


def read_examples(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


if __name__ == "__main__":
    main()
