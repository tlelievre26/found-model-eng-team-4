"""Next-token multiple-choice scoring for Issue #5."""
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from src.utils.paths import MODELS_DIR


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MODEL_PATH = MODELS_DIR / "Qwen3-4B-Instruct-2507"


def load_base_model(model_path=DEFAULT_MODEL_PATH, device_map="auto"):
    """Load the local NF4 base and tokenizer for scoring or QLoRA training.

    Matches #7's shared entry point: returns (model, tokenizer), uses double
    quantization and bf16 compute. The caller attaches LoRA and prepares training.
    """
    model_path = Path(model_path).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory missing: {model_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit loading requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The shared QLoRA configuration requires bf16 support")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, device_map=device_map, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
    )
    return model, tokenizer


class QwenWrapper:
    """Return probabilities conditional on the allowed answer letters.

    Pass Issue #3's shared chat-message formatter as ``prompt_fn``. No local
    alternative prompt is used, so training and evaluation share one format.
    Inject both model and tokenizer for CPU unit tests without downloading weights.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, *, prompt_fn: Callable | None = None,
                 adapter_path=None, model=None, tokenizer=None):
        if prompt_fn is None:
            from src.data.medqa_datasets.prompts import build_messages
            prompt_fn = build_messages
        if not callable(prompt_fn):
            raise TypeError("prompt_fn must be the shared chat-message formatter")
        if (model is None) != (tokenizer is None):
            raise ValueError("Inject both model and tokenizer, or neither")
        if model is None:
            model, tokenizer = load_base_model(model_path)
        self.model = model
        self.adapter_path = None
        self.tokenizer = tokenizer
        self.prompt_fn = prompt_fn
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a padding or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if adapter_path is not None:
            self.load_adapter(adapter_path)

    def load_adapter(self, path):
        """Load one local saved PEFT adapter for evaluation.

        Use a fresh wrapper for another checkpoint; never stack evaluation
        checkpoints or replace adapters on an actively trained model.
        """
        if self.adapter_path is not None or hasattr(self.model, "peft_config"):
            raise RuntimeError("Use a fresh base-model wrapper for each saved adapter")
        path = Path(path).expanduser()
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Local PEFT adapter config missing: {path}")
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.model, str(path), is_trainable=False, local_files_only=True,
        ).eval()
        self.adapter_path = str(path)

    def _prepare(self, example):
        options = example.get("options")
        if not isinstance(options, Mapping) or len(options) < 2:
            raise ValueError("options must map at least two letters to answer texts")
        letters = list(options)
        if any(not isinstance(x, str) or len(x) != 1 or not ("A" <= x <= "Z")
               for x in letters):
            raise ValueError("Option keys must be single uppercase letters A-Z")
        letters.sort()
        messages = self.prompt_fn(example)
        if any(message.get("role") == "assistant" for message in messages):
            raise ValueError("Scoring prompts must not contain assistant answer turns")
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prefix_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        token_ids = []
        for letter in letters:
            # Check the actual rendered context, not just isolated letter encoding.
            combined = self.tokenizer.encode(prompt + letter, add_special_tokens=False)
            if combined[:len(prefix_ids)] != prefix_ids or len(combined) != len(prefix_ids) + 1:
                raise ValueError(f"{letter!r} is not one next token in this chat context")
            token_ids.append(combined[-1])
        if len(set(token_ids)) != len(letters):
            raise ValueError("Different answer letters map to the same token")
        return prompt, letters, token_ids

    def score(self, example):
        """Score one common-format question; return probabilities and prediction."""
        return self.score_batch([example], batch_size=1)[0]

    def score_batch(self, examples: Sequence[Mapping], batch_size=8):
        """Score questions in order with one forward pass per batch.

        Scores are softmax-normalized over the supplied letters only. They are
        not calibrated probabilities of medical correctness or full-vocabulary mass.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        was_training = getattr(self.model, "training", None)
        self.model.eval()
        try:
            return self._score_batches(examples, batch_size)
        finally:
            if was_training is not None:
                self.model.train(was_training)

    def _score_batches(self, examples, batch_size):
        results = []
        for start in range(0, len(examples), batch_size):
            prepared = [self._prepare(x) for x in examples[start:start + batch_size]]
            inputs = self.tokenizer([x[0] for x in prepared], padding=True,
                                    add_special_tokens=False, return_tensors="pt")
            device = self.model.get_input_embeddings().weight.device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            # Forward calls need positions independent of left-padding width.
            mask = inputs["attention_mask"]
            inputs["position_ids"] = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
            with torch.inference_mode():
                # Qwen3 supports computing only the final-token logits. Avoid
                # allocating a batch x sequence x vocabulary tensor on the L4.
                kwargs = {}
                if getattr(getattr(self.model, "config", None), "model_type", None) == "qwen3":
                    kwargs["logits_to_keep"] = 1
                logits = self.model(**inputs, use_cache=False, **kwargs).logits
                # Locate the final non-padding token, including for unequal lengths.
                mask = inputs["attention_mask"]
                positions = torch.arange(mask.shape[1], device=device).expand_as(mask)
                last = positions.masked_fill(mask == 0, -1).max(dim=1).values
                next_logits = (logits[:, -1] if kwargs else
                               logits[torch.arange(len(prepared), device=device), last])
                for row, (_, letters, ids) in zip(next_logits, prepared):
                    probs = torch.softmax(row[ids].float(), dim=-1)
                    if not torch.isfinite(probs).all():
                        raise ValueError("Model produced non-finite option probabilities")
                    probabilities = dict(zip(letters, probs.cpu().tolist()))
                    results.append({"probabilities": probabilities,
                                    "prediction": max(probabilities, key=probabilities.get)})
        return results
