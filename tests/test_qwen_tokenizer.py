"""Checks against the real local Qwen tokenizer, without loading weights; skipped if no tokenizer is found."""
import os

import pytest

from src.models.qwen import DEFAULT_MODEL_PATH, QwenWrapper
from test_qwen import Model, example


def test_real_tokenizer_letter_boundaries():
    # Falls back to the shared model folder (paths.MODELS_DIR), so it runs on the VM without extra setup.
    path = os.environ.get("QWEN_TOKENIZER_PATH")
    if not path and DEFAULT_MODEL_PATH.is_dir():
        path = str(DEFAULT_MODEL_PATH)
    if not path:
        pytest.skip("Set QWEN_TOKENIZER_PATH or put the model under paths.MODELS_DIR")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    from src.data.medqa_datasets import build_messages
    w = QwenWrapper(model=Model(), tokenizer=tokenizer, prompt_fn=build_messages)
    prompt, letters, ids = w._prepare(example(6))
    assert len(set(ids)) == 6
    for letter in letters:
        bare = tokenizer.encode(letter, add_special_tokens=False)
        spaced = tokenizer.encode(" " + letter, add_special_tokens=False)
        print(f"{letter}: bare={bare}, spaced={spaced}, chat={ids[letters.index(letter)]}")
    assert prompt
