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


def test_cli_shared_loaders_and_existing_logging(monkeypatch, tmp_path):
    import runpy
    from src.data.medqa_datasets import data
    from src.utils import wandb_logging as wl
    from src.eval import evaluate as real_evaluate
    root = tmp_path / 'data'
    (root / 'splits').mkdir(parents=True)
    ex = {'id': 'medmcqa-1', 'source': 'medmcqa', 'question': 'synthetic',
          'options': {'A': 'a', 'B': 'b'}, 'answer': 'A'}
    (root / 'processed').mkdir()
    (root / 'processed' / 'medmcqa.jsonl').write_text(json.dumps(ex)+'\n')
    (root / 'splits' / 'medmcqa.json').write_text(json.dumps(
        {key: [ex['id']] if key == 'hp_dev' else [] for key in data.split_names('medmcqa')}))
    (root / 'splits' / 'stats.json').write_text(json.dumps({'config': data.current_config()}))
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


def test_ece_matches_training_stub_boundary_convention():
    # Confidence .5 belongs to the first of two right-closed bins.
    p = [{"A": .5, "B": .5}, {"A": .75, "B": .25}]
    assert compute_metrics(p, ["A", "B"], n_bins=2)["ece"] == pytest.approx(.625)
