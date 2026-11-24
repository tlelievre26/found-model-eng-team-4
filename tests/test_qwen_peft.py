"""Real small random Qwen/PEFT checks on CPU; no pretrained weights or GPU."""
import torch
import pytest

from src.models.qwen import QwenWrapper
from test_qwen import Tokenizer, example


def tiny_model():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(vocab_size=128, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=2048)
    return transformers.Qwen3ForCausalLM(config)


def test_real_qwen_padding_matches_single():
    torch.manual_seed(42)
    w = QwenWrapper(model=tiny_model(), tokenizer=Tokenizer())
    examples = [example(4, "x"), example(6, "a longer question")]
    batched = w.score_batch(examples)
    singles = [w.score(x) for x in examples]
    for a, b in zip(batched, singles):
        assert a["probabilities"] == pytest.approx(b["probabilities"], abs=1e-6)


def test_real_peft_checkpoint_roundtrip(tmp_path):
    peft = pytest.importorskip("peft")
    torch.manual_seed(42)
    base = tiny_model()
    state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    trained = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4,
        target_modules=["q_proj"], task_type="CAUSAL_LM"))
    # Make the adapter non-zero so the test verifies actual adapter effects.
    with torch.no_grad():
        for name, p in trained.named_parameters():
            if "lora_B" in name:
                p.fill_(0.3)
    ex = example()
    active = QwenWrapper(model=trained, tokenizer=Tokenizer())
    expected = active.score(ex)
    trained.save_pretrained(tmp_path)
    restored_base = tiny_model()
    restored_base.load_state_dict(state)
    restored = QwenWrapper(model=restored_base, tokenizer=Tokenizer(), adapter_path=tmp_path)
    assert restored.score(ex)["probabilities"] == pytest.approx(expected["probabilities"], abs=1e-6)
