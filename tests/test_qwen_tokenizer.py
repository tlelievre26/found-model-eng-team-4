"""Opt-in checks against the real local Qwen tokenizer, without loading weights."""
import os

import pytest

from src.models.qwen import QwenWrapper
from test_qwen import Model, example


def test_real_tokenizer_letter_boundaries():
    path = os.environ.get("QWEN_TOKENIZER_PATH")
    if not path:
        pytest.skip("Set QWEN_TOKENIZER_PATH to run the real tokenizer check")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    w = QwenWrapper(model=Model(), tokenizer=tokenizer, prompt_fn=lambda x: [
        {"role": "user", "content": x["question"] + "\nAnswer with only the letter."}])
    prompt, letters, ids = w._prepare(example(6))
    assert len(set(ids)) == 6
    for letter in letters:
        bare = tokenizer.encode(letter, add_special_tokens=False)
        spaced = tokenizer.encode(" " + letter, add_special_tokens=False)
        print(f"{letter}: bare={bare}, spaced={spaced}, chat={ids[letters.index(letter)]}")
    assert prompt
