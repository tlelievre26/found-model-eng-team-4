"""Evaluate the base model or one local LoRA checkpoint using fixed splits."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.medqa_datasets import load_splits, PROMPT_VERSION
from src.eval import evaluate
from src.models.qwen import QwenWrapper, DEFAULT_MODEL_PATH, MODEL_ID
from src.utils import paths, wandb_logging as wl


def model_revision(model_path):
    """Hub commit of a model downloaded with `hf download --local-dir`, or None if unknown."""
    meta = Path(model_path) / ".cache" / "huggingface" / "download" / "config.json.metadata"
    return meta.read_text().split()[0] if meta.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["medmcqa", "medqa"], default="medmcqa")
    parser.add_argument("--split", default="hp_dev")
    parser.add_argument("--adapter", default=None, help="Local adapter folder; omit for base")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data-dir", type=Path, default=paths.DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Keep 16 for reported numbers; scores shift slightly with batch size")
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--limit", type=int, help="Smoke test only; omit for full baseline")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-hourly-usd", type=float, default=wl.DEFAULT_GPU_HOURLY_USD)
    args = parser.parse_args()
    if args.batch_size < 1 or args.n_bins < 1:
        parser.error("--batch-size and --n-bins must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    splits = load_splits(args.dataset, data_dir=args.data_dir)
    if args.split not in splits:
        parser.error(f"Unknown split; choose from {list(splits)}")
    examples = splits[args.split]
    if args.limit is not None:
        examples = examples[:args.limit]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{args.dataset}-{args.split}-{'adapter' if args.adapter else 'base'}-{stamp}"
    output = args.output or paths.EVAL_DIR / (name + ".jsonl")
    split_file = args.data_dir / "splits" / f"{args.dataset}.json"
    stats_file = args.data_dir / "splits" / "stats.json"
    config = {"dataset": args.dataset, "split": args.split, "model_id": MODEL_ID,
              "model_path": str(args.model_path), "model_revision": model_revision(args.model_path),
              "adapter_path": args.adapter,
              "prompt_version": PROMPT_VERSION, "batch_size": args.batch_size,
              "n_bins": args.n_bins, "limit": args.limit, "output": str(output),
              "split_ids_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
              "stats_sha256": hashlib.sha256(stats_file.read_bytes()).hexdigest()}
    with wl.start_run(name, config, job_type="eval", gpu_hourly_usd=args.gpu_hourly_usd):
        wrapper = QwenWrapper(model_path=args.model_path, adapter_path=args.adapter)
        metrics = evaluate(wrapper, examples, args.batch_size, n_bins=args.n_bins,
                           output_path=output, metadata=config)
        wl.log_eval(metrics)
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()
