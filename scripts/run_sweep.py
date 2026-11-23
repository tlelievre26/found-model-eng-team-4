"""Run a hyperparameter grid sequentially in this process, with the same train() as every other run.

    python scripts/run_sweep.py --sweep configs/sweep.yaml

First the untuned base model is evaluated on the trials' eval_split (a W&B run with job_type "eval" in the
sweep's group). Then every grid point is trained in a fixed order into $CHECKPOINTS_DIR/sweep-<name>/<trial>/.
Re-running the command skips finished trials (DONE marker) and resumes a partial one. No W&B sweep controller is
used, so resuming is reliable. Trials are listed by final eval/nll at the end; the team picks the winner by hand.
"""

import argparse
import gc
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Less allocator fragmentation: variable-width batches otherwise leave GBs reserved but unused (matters on the L4).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import yaml  # noqa: E402

from src.data.medqa_datasets import build_messages  # noqa: E402
from src.eval.stub import evaluate  # noqa: E402  STUB until #12's harness lands
from src.models.qwen import DEFAULT_MODEL_PATH, QwenWrapper  # noqa: E402
from src.training import data as td  # noqa: E402
from src.training.config import apply_overrides, load_config  # noqa: E402
from src.training.train import gpu_tag, is_done, output_dir, provenance, resolve_gpu_hourly_usd, train  # noqa: E402
from src.utils import paths  # noqa: E402
from src.utils import wandb_logging as wl  # noqa: E402

SHORT_NAMES = {"learning_rate": "lr", "lora_rank": "r"}


def trial_name(params: dict) -> str:
    """{"learning_rate": 5e-5, "lora_rank": 8} -> "lr5e-05_r8"."""
    return "_".join(f"{SHORT_NAMES.get(k, k)}{v:g}" if isinstance(v, (int, float)) else f"{SHORT_NAMES.get(k, k)}{v}"
                    for k, v in params.items())


def sweep_trials(sweep_path: Path, extra_overrides: list[str] = ()) -> tuple[dict, Path, list[tuple[str, dict]]]:
    """(sweep spec, sweep folder, [(trial name, resolved config), ...]) in run order."""
    sweep = yaml.safe_load(sweep_path.read_text())
    base = sweep_path.parent / sweep["base_config"]
    sweep_dir = paths.CHECKPOINTS_DIR / f"sweep-{sweep['name']}"
    group = f"sweep-{sweep['name']}"
    keys = list(sweep["grid"])
    trials = []
    for values in itertools.product(*(sweep["grid"][k] for k in keys)):
        params = dict(zip(keys, values))
        name = trial_name(params)
        cfg = load_config(base, {**sweep.get("overrides", {}), **params})
        cfg.update(run_name=f"{group}-{name}", output_dir=str(sweep_dir / name), wandb_group=group,
                   wandb_tags=[*cfg["wandb_tags"], "sweep"])
        if extra_overrides:
            cfg = apply_overrides(cfg, extra_overrides)
        trials.append((name, cfg))
    return sweep, sweep_dir, trials


def run_baseline(cfg: dict, sweep_dir: Path, group: str) -> dict:
    """Evaluate the untuned base model on the trials' eval split; saved to sweep_dir/baseline_eval.json."""
    examples = td.load_eval_examples(cfg["eval_split"], cfg["eval_limit"])
    gpu_name = wl._gpu_name()
    model_path = Path(cfg["model_path"] or DEFAULT_MODEL_PATH)
    config = {key: cfg[key] for key in ("eval_split", "eval_limit", "eval_batch_size", "model_path", "model_revision")}
    config.update(provenance=provenance(cfg, model_path), gpu_tag=gpu_tag(gpu_name), adapter=None,
                  n_eval_examples=len(examples))
    with wl.start_run(f"{group}-baseline-{gpu_tag(gpu_name)}", config, group=group, job_type="eval",
                      tags=[*cfg["wandb_tags"], "baseline", gpu_tag(gpu_name)],
                      gpu_hourly_usd=resolve_gpu_hourly_usd(cfg, gpu_name)):
        wrapper = QwenWrapper(model_path, prompt_fn=build_messages)
        result = evaluate(wrapper, examples, batch_size=cfg["eval_batch_size"])
        wl.log_eval(result, step=0, tokens_seen=0)
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "baseline_eval.json").write_text(json.dumps({"split": cfg["eval_split"], **result}, indent=2))
    print(f"[sweep] baseline {cfg['eval_split']}: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="sweep YAML, e.g. configs/sweep.yaml")
    parser.add_argument("overrides", nargs="*",
                        help="key=value applied to every trial (e.g. gpu_hourly_usd=0); not part of the trial name")
    args = parser.parse_args()
    sweep, sweep_dir, trials = sweep_trials(Path(args.sweep), args.overrides)
    group = f"sweep-{sweep['name']}"
    print(f"[sweep] {group}: {len(trials)} trials -> {sweep_dir}")

    baseline_file = sweep_dir / "baseline_eval.json"
    if baseline_file.exists():
        print(f"[sweep] baseline already evaluated ({baseline_file})")
    else:
        run_baseline(trials[0][1], sweep_dir, group)

    for name, cfg in trials:
        if is_done(cfg):
            print(f"[sweep] {name}: done, skipping")
            continue
        print(f"[sweep] {name}: training into {output_dir(cfg)}")
        train(cfg)

    rows = []
    for name, cfg in trials:
        evals = output_dir(cfg) / "evals.jsonl"
        last = json.loads(evals.read_text().splitlines()[-1]) if evals.exists() else {}
        rows.append({"trial": name, **{k: last.get(k) for k in ("nll", "accuracy", "ece", "n", "step")}})
    rows.sort(key=lambda r: float("inf") if r["nll"] is None else r["nll"])
    baseline = json.loads(baseline_file.read_text())
    (sweep_dir / "summary.json").write_text(json.dumps({"baseline": baseline, "trials": rows}, indent=2))
    print(f"[sweep] final {trials[0][1]['eval_split']} metrics, by eval/nll (lower is better); "
          "the team picks the final hyperparameters:")
    print(f"  {'baseline':<16} nll={baseline['nll']:.4f} acc={baseline['accuracy']:.4f} ece={baseline['ece']:.4f}")
    for r in rows:
        if r["nll"] is not None:
            print(f"  {r['trial']:<16} nll={r['nll']:.4f} acc={r['accuracy']:.4f} ece={r['ece']:.4f}")


if __name__ == "__main__":
    main()
