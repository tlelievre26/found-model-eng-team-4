"""Train (or resume) one answer-only QLoRA run.

    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/smoke.yaml run_name=smoke-r32 lora_rank=32

Overrides are key=value pairs, parsed with yaml.safe_load. Re-running the same command resumes the run from its
latest resume state; a completed run (DONE marker) is not trained again. See docs/tuning_setup.md.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Less allocator fragmentation: variable-width batches otherwise leave GBs reserved but unused (matters on the L4).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.training.config import load_config  # noqa: E402
from src.training.train import RunAlreadyComplete, train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/smoke.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. lora_rank=32")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    try:
        train(cfg)
    except RunAlreadyComplete as exc:
        sys.exit(f"Refusing to retrain: {exc}")


if __name__ == "__main__":
    main()
