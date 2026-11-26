import json
import math
from types import SimpleNamespace
import sys

import pytest
import torch

from src.eval import compute_metrics, evaluate
from src.models.qwen import QwenWrapper
from test_qwen import Model, Tokenizer, example


def test_metrics_by_hand():
    probs = [{"A": .8, "B": .2}, {"A": .8, "B": .2}]
    got = compute_metrics(probs, ["A", "B"])
    assert got == pytest.approx({"accuracy": .5, "nll": -math.log(.16)/2,
                                "ece": .3, "n": 2})
    assert compute_metrics([{"A": 1., "B": 0.}], ["A"])["ece"] == 0
    assert compute_metrics([{"A": 1., "B": 0.}], ["B"])["nll"] == pytest.approx(-math.log(1e-12))


def test_calibrated_and_bin_edges():
    probs = [{"A": .5, "B": .5}, {"A": .5, "B": .5}, {"A": 1., "B": 0.}]
    assert compute_metrics(probs, ["A", "B", "A"], n_bins=2)["ece"] == 0


@pytest.mark.parametrize("p,a,b", [([], [], 15), ([{"A": .2}], ["A"], 15),
    ([{"A": float('nan')}], ["A"], 15), ([{"A": 1}], ["B"], 15),
    ([{"A": 1}], ["A"], 0)])
def test_invalid_metrics(p, a, b):
    with pytest.raises(ValueError):
        compute_metrics(p, a, b)


class FakeWrapper:
    def score_batch(self, examples, batch_size):
        return [{"probabilities": {"A": .75, "B": .25}, "prediction": "A"}
                for _ in examples]


def test_evaluate_saves_reloadable_predictions(tmp_path):
    examples = [{"id": str(i), "options": {"A": "a", "B": "b"}, "answer": ans}
                for i, ans in enumerate(["A", "B"])]
    path = tmp_path / "run" / "predictions.jsonl"
    metrics = evaluate(FakeWrapper(), examples, output_path=path,
                       metadata={"split": "hp_dev"})
    assert metrics["accuracy"] == .5
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["0", "1"]
    assert [r["correct"] for r in rows] == [True, False]
    assert json.loads(path.with_suffix('.jsonl.meta.json').read_text())["metadata"]["split"] == "hp_dev"
    assert "question" not in rows[0]
    with pytest.raises(FileExistsError):
        evaluate(FakeWrapper(), examples, output_path=path)


def test_invalid_metadata_leaves_no_partial_outputs(tmp_path):
    path = tmp_path / 'results.jsonl'
    ex = {'id': '1', 'options': {'A': 'a', 'B': 'b'}, 'answer': 'A'}
    with pytest.raises(TypeError):
        evaluate(FakeWrapper(), [ex], output_path=path, metadata={'bad': object()})
    assert not path.exists()
    assert not path.with_suffix('.jsonl.meta.json').exists()


def test_wrong_prediction_count_rejected():
    class Missing:
        def score_batch(self, examples, batch_size):
            return []
    with pytest.raises(ValueError, match='number'):
        evaluate(Missing(), [{'id': '1', 'options': {'A': 'a'}, 'answer': 'A'}])


def _cli_data(tmp_path):
    from src.data.medqa_datasets import data
    root = tmp_path / 'data'
    (root / 'splits').mkdir(parents=True)
    ex = {'id': 'medmcqa-1', 'source': 'medmcqa', 'question': 'synthetic',
          'options': {'A': 'a', 'B': 'b'}, 'answer': 'A'}
    (root / 'processed').mkdir()
    (root / 'processed' / 'medmcqa.jsonl').write_text(json.dumps(ex)+'\n')
    (root / 'splits' / 'medmcqa.json').write_text(json.dumps(
        {key: [ex['id']] if key == 'hp_dev' else [] for key in data.split_names('medmcqa')}))
    (root / 'splits' / 'stats.json').write_text(json.dumps({'config': data.current_config()}))
    return root, ex


def test_cli_shared_loaders_and_existing_logging(monkeypatch, tmp_path):
    import runpy
    from src.utils import wandb_logging as wl
    from src.eval import evaluate as real_evaluate
    root, ex = _cli_data(tmp_path)
    from src.models import qwen
    seen = []
    class CLIWrapper(FakeWrapper):
        def __init__(self, **kwargs):
            seen.append(kwargs)
    monkeypatch.setattr(qwen, 'QwenWrapper', CLIWrapper)
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    out = tmp_path / 'predictions.jsonl'
    monkeypatch.setattr(sys, 'argv', ['scripts/evaluate.py', '--data-dir', str(root),
        '--dataset', 'medmcqa', '--split', 'hp_dev', '--output', str(out)])
    runpy.run_path('scripts/evaluate.py', run_name='__main__')
    assert json.loads(out.read_text())['id'] == ex['id']
    assert seen[0]['adapter_path'] is None
    # In-training API must neither start nor finish the caller's W&B run.
    with wl.start_run('training-test', {}, job_type='train') as logger:
        assert real_evaluate(FakeWrapper(), [ex])['accuracy'] == 1
        wl.log_eval({'accuracy': 1.0})
        assert not logger._finished


def _run_cli_with_compute(monkeypatch, tmp_path, adapter=None, metadata=None):
    """Run the CLI on synthetic data; return (log_eval kwargs, saved config, run name, summary)."""
    import runpy
    from src.models import qwen
    from src.utils import wandb_logging as wl
    root, _ = _cli_data(tmp_path)
    model = tmp_path / 'model'
    (model / '.cache' / 'huggingface' / 'download').mkdir(parents=True)
    (model / '.cache' / 'huggingface' / 'download' / 'config.json.metadata').write_text('rev-a\netag\n')
    if metadata is not None:
        adapter.mkdir()
        (adapter / 'metadata.json').write_text(json.dumps(metadata))
    logged, runs = [], []
    real_log_eval, real_start_run = wl.log_eval, wl.start_run
    def log_eval(metrics, **kwargs):
        logged.append(kwargs)
        real_log_eval(metrics, **kwargs)
    def start_run(name, config, **kwargs):
        runs.append(real_start_run(name, config, **kwargs))
        return runs[-1]
    monkeypatch.setattr(wl, 'log_eval', log_eval)
    monkeypatch.setattr(wl, 'start_run', start_run)
    monkeypatch.setattr(qwen, 'QwenWrapper', lambda **kwargs: FakeWrapper())
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    out = tmp_path / 'predictions.jsonl'
    argv = ['scripts/evaluate.py', '--data-dir', str(root), '--model-path', str(model),
            '--output', str(out)]
    monkeypatch.setattr(sys, 'argv', argv + (['--adapter', str(adapter)] if adapter else []))
    runpy.run_path('scripts/evaluate.py', run_name='__main__')
    config = json.loads(out.with_suffix('.jsonl.meta.json').read_text())['metadata']
    return logged[0], config, runs[0].run.name, runs[0].run.summary


def _checkpoint_metadata(**changes):
    from src.data.medqa_datasets import PROMPT_VERSION
    meta = {'step': 248, 'total_steps': 248, 'final': True, 'epoch': 0.05, 'tokens_seen': 593121,
            'examples_seen': 7921, 'gpu_hours': 0.108, 'train_hours': 0.087, 'wall_clock_hours': 0.09,
            'gpu_name': 'NVIDIA RTX A6000', 'wandb_run_id': 'abc123', 'resume_count': 0,
            'git_commit': 'deadbeef', 'git_dirty': False, 'model_revision': 'rev-a',
            'prompt_version': PROMPT_VERSION, 'dedupe_scheme': 'two-tier-v3',
            'config': {'run_name': 'r32', 'lora_rank': 32, 'learning_rate': 2e-4, 'seed': 42}}
    return {**meta, **changes}


def test_cli_logs_checkpoint_compute(monkeypatch, tmp_path):
    adapter = tmp_path / 'step-000248'
    logged, config, name, summary = _run_cli_with_compute(
        monkeypatch, tmp_path, adapter, _checkpoint_metadata())
    assert logged == {'step': 248, 'tokens_seen': 593121}
    ckpt = config['checkpoint']
    assert ckpt['tokens_seen'] == 593121 and ckpt['gpu_hours'] == 0.108
    assert ckpt['run_name'] == 'r32' and ckpt['lora_rank'] == 32 and ckpt['learning_rate'] == 2e-4
    assert ckpt['gpu_name'] == 'NVIDIA RTX A6000' and ckpt['wandb_run_id'] == 'abc123'
    assert 'seed' not in ckpt and config['model_revision'] == 'rev-a'
    assert name.startswith('medmcqa-hp_dev-r32-step000248-')
    assert summary['checkpoint/gpu_hours'] == 0.108


def test_cli_base_model_is_zero_compute(monkeypatch, tmp_path):
    logged, config, name, summary = _run_cli_with_compute(monkeypatch, tmp_path)
    assert logged == {'step': 0, 'tokens_seen': 0}
    assert config['checkpoint'] is None
    assert name.startswith('medmcqa-hp_dev-base-')
    assert summary['checkpoint/gpu_hours'] == 0


@pytest.mark.parametrize('metadata, error', [
    (None, 'metadata.json'),
    (_checkpoint_metadata(model_revision='rev-b'), 'model revision'),
    (_checkpoint_metadata(prompt_version='v0'), 'prompt version'),
])
def test_cli_rejects_bad_checkpoint(monkeypatch, tmp_path, metadata, error):
    adapter = tmp_path / 'adapter'
    if metadata is None:
        adapter.mkdir()
    with pytest.raises((FileNotFoundError, ValueError), match=error):
        _run_cli_with_compute(monkeypatch, tmp_path, adapter, metadata)


def test_ece_matches_training_stub_boundary_convention():
    # Confidence .5 belongs to the first of two right-closed bins.
    p = [{"A": .5, "B": .5}, {"A": .75, "B": .25}]
    assert compute_metrics(p, ["A", "B"], n_bins=2)["ece"] == pytest.approx(.625)
