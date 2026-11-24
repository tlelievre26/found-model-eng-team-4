from types import SimpleNamespace
import sys
import pytest
from src.models.qwen import QwenWrapper
from test_qwen import Model, Tokenizer, example


def test_shared_prompt_default_and_no_gold_answer():
    w = QwenWrapper(model=Model(), tokenizer=Tokenizer())
    ex = example()
    ex["answer"] = "C"
    prompt, _, _ = w._prepare(ex)
    assert "A. 0" in prompt and "Answer with only the letter (A, B, C, D)." in prompt
    assert w.score(ex) == w.score(dict(ex, answer="D"))
    w.prompt_fn = lambda ex: [{"role": "assistant", "content": ex["answer"]}]
    with pytest.raises(ValueError, match="assistant"):
        w.score(ex)


def test_training_mode_restored_on_success_and_error():
    class StatefulModel(Model):
        training = True
        def train(self, state=True):
            self.training = state
            return self
        def eval(self):
            return self.train(False)
        def __call__(self, **kwargs):
            assert not self.training
            return super().__call__(**kwargs)
    model = StatefulModel()
    w = QwenWrapper(model=model, tokenizer=Tokenizer())
    w.score(example())
    assert model.training
    with pytest.raises(ValueError):
        w.score({"options": {}})
    assert model.training


def test_adapter_loads_after_base_initialization(monkeypatch, tmp_path):
    (tmp_path / "adapter_config.json").write_text('{}')
    model = Model()
    called = []
    class FakePeft:
        @staticmethod
        def from_pretrained(base, path, **kwargs):
            called.append((base, path, kwargs))
            return base
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=FakePeft))
    w = QwenWrapper(model=model, tokenizer=Tokenizer(), adapter_path=tmp_path)
    assert called[0][0] is model
    assert called[0][2]["is_trainable"] is False
    assert called[0][2]["local_files_only"] is True
    with pytest.raises(RuntimeError):
        w.load_adapter(tmp_path)


def test_shared_base_loader_contract(monkeypatch, tmp_path):
    import torch
    from src.models.qwen import load_base_model
    seen = {}
    class Factory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen['model'] = (path, kwargs)
            return 'model'
    class TokenizerFactory:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen['tokenizer'] = (path, kwargs)
            return 'tokenizer'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=Factory, AutoTokenizer=TokenizerFactory,
        BitsAndBytesConfig=lambda **kwargs: kwargs))
    assert load_base_model(tmp_path, device_map={'': 0}) == ('model', 'tokenizer')
    kwargs = seen['model'][1]
    assert kwargs['device_map'] == {'': 0}
    assert kwargs['local_files_only']
    assert kwargs['dtype'] == torch.bfloat16
    assert kwargs['quantization_config'] == {
        'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': torch.bfloat16}
