"""Smoke-test the shared prompt/loaders and real Qwen on a CUDA machine."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.data.medqa_datasets import load_splits
from src.data.medabstain.data import load_medabstain
from src.models.qwen import DEFAULT_MODEL_PATH, QwenWrapper
from src.utils import paths, wandb_logging as wl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data-dir", type=Path, default=paths.DATA_DIR)
    parser.add_argument("--medabstain-dir", type=Path, default=paths.MEDABSTAIN_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--gpu-hourly-usd", type=float, default=wl.DEFAULT_GPU_HOURLY_USD)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Run model validation on CUDA")
    examples = load_splits("medmcqa", data_dir=args.data_dir)["hp_dev"][:args.limit]
    six = next(x for x in load_medabstain("A", data_dir=args.medabstain_dir)
               if len(x["options"]) == 6)
    if not examples:
        raise ValueError("The hp_dev split is empty")
    torch.cuda.init()  # reset_peak_memory_stats fails before CUDA is initialized
    for device in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device)
    with wl.start_run("qwen-wrapper-smoke", vars(args) | {
        "data_dir": str(args.data_dir), "medabstain_dir": str(args.medabstain_dir)
    }, job_type="eval", gpu_hourly_usd=args.gpu_hourly_usd):
        wrapper = QwenWrapper(args.model_path)
        results = wrapper.score_batch(examples, batch_size=args.batch_size)
        for result in results:
            probs = result["probabilities"]
            if abs(sum(probs.values()) - 1) > 1e-5:
                raise AssertionError("Probabilities do not sum to one")
        metrics = {"accuracy": sum(r["prediction"] == ex["answer"]
                                  for r, ex in zip(results, examples)) / len(examples),
                   "n": len(examples)}
        single = wrapper.score(examples[0])
        batched = wrapper.score_batch([examples[0], six], batch_size=2)
        for letter, p in single["probabilities"].items():
            if abs(p-batched[0]["probabilities"][letter]) > 1e-3:
                raise AssertionError("Single/batch probabilities differ by more than 1e-3")
        six_probs = batched[1]["probabilities"]
        if set(six_probs) != set(six["options"]) or abs(sum(six_probs.values())-1) > 1e-5:
            raise AssertionError("Invalid six-option scoring")
        memory = []
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
            memory.append({"gpu": torch.cuda.get_device_name(device),
                           "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                           "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30})
        print(json.dumps({"memory": memory}, indent=2))
        wl.log_eval(metrics)
        print(json.dumps({"metrics": metrics, "six_option_result": batched[1]}, indent=2))
        if metrics["accuracy"] <= .25:
            raise AssertionError("Investigate scoring: accuracy is not above chance")


if __name__ == "__main__":
    main()
