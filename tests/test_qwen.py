"""CPU tests with deterministic model logits; no weights or network needed."""
from types import SimpleNamespace

import pytest
import torch

from src.models.qwen import QwenWrapper


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\nassistant:\n"

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def __call__(self, prompts, **kwargs):
        rows = [self.encode(p) for p in prompts]
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([[0] * (width-len(r)) + r for r in rows]),
                "attention_mask": torch.tensor([[0] * (width-len(r)) + [1]*len(r) for r in rows])}


class Model:
    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def get_input_embeddings(self):
        return SimpleNamespace(weight=torch.zeros(1))

    def __call__(self, input_ids, attention_mask, **kwargs):
        self.calls += 1
        logits = torch.zeros(*input_ids.shape, 128)
        # Different question lengths produce different winners. Padding must not
        # change the result when the same question is scored in a mixed batch.
        for i, mask in enumerate(attention_mask):
            winner = ord("A") + int(mask.sum()) % 4
            logits[i, :, winner] = 4
        return SimpleNamespace(logits=logits)


def example(n=4, question="Question"):
    return {"question": question, "options": {chr(65+i): str(i) for i in range(n)}}


def wrapper(tokenizer=None):
    return QwenWrapper(prompt_fn=lambda x: [{"role": "user", "content": x["question"]}],
                       model=Model(), tokenizer=tokenizer or Tokenizer())


@pytest.mark.parametrize("n", [4, 6])
def test_normalized_scores(n):
    result = wrapper().score(example(n))
    assert len(result["probabilities"]) == n
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["prediction"] == max(result["probabilities"], key=result["probabilities"].get)


def test_batch_matches_single_and_uses_one_pass_per_batch():
    w = wrapper()
    questions = [example(4, "x"), example(6, "long question"), example(4, "abc")]
    batched = w.score_batch(questions, batch_size=2)
    assert w.model.calls == 2
    assert batched == [w.score(x) for x in questions]
    assert w.score_batch([]) == []


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_batch_size(size):
    with pytest.raises(ValueError):
        wrapper().score_batch([], size)


def test_adapter_hook():
    with pytest.raises(NotImplementedError):
        wrapper().load_adapter("adapter")


def test_bad_options():
    with pytest.raises(ValueError):
        wrapper().score({"options": {"AA": "text", "B": "text"}})


def test_context_token_split_is_rejected():
    class SplitTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            return ids + [2] if text.endswith("A") else ids
    with pytest.raises(ValueError, match="one next token"):
        wrapper(SplitTokenizer()).score(example())
