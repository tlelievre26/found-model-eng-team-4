"""Next-token multiple-choice scoring for Issue #5."""
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MODEL_PATH = "/shared/models/Qwen3-4B-Instruct-2507"


class QwenWrapper:
    """Return probabilities conditional on the allowed answer letters.

    Pass Issue #3's shared chat-message formatter as ``prompt_fn``. No local
    alternative prompt is used, so training and evaluation share one format.
    Inject both model and tokenizer for CPU unit tests without downloading weights.
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, *, prompt_fn: Callable,
                 adapter_path=None, model=None, tokenizer=None):
        if not callable(prompt_fn):
            raise TypeError("prompt_fn must be the shared chat-message formatter")
        if adapter_path is not None:
            self.load_adapter(adapter_path)
        if (model is None) != (tokenizer is None):
            raise ValueError("Inject both model and tokenizer, or neither")
        if model is None:
            if not Path(model_path).is_dir():
                raise FileNotFoundError(f"Local model directory missing: {model_path}")
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit loading requires the project's CUDA VM")
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, local_files_only=True, device_map="auto",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16),
            )
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.prompt_fn = prompt_fn
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a padding or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_adapter(self, path):
        """Future Issue #7 hook: attach a LoRA adapter to the base model.

        Deliberately unsupported until adapter loading is implemented and tested.
        """
        raise NotImplementedError("LoRA adapter loading is pending Issue #7")

    def _prepare(self, example):
        options = example.get("options")
        if not isinstance(options, Mapping) or len(options) < 2:
            raise ValueError("options must map at least two letters to answer texts")
        letters = list(options)
        if any(not isinstance(x, str) or len(x) != 1 or not ("A" <= x <= "Z")
               for x in letters):
            raise ValueError("Option keys must be single uppercase letters A-Z")
        messages = self.prompt_fn(example)
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
        results = []
        for start in range(0, len(examples), batch_size):
            prepared = [self._prepare(x) for x in examples[start:start + batch_size]]
            inputs = self.tokenizer([x[0] for x in prepared], padding=True,
                                    add_special_tokens=False, return_tensors="pt")
            device = self.model.get_input_embeddings().weight.device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits
                # Locate the final non-padding token, including for unequal lengths.
                mask = inputs["attention_mask"]
                positions = torch.arange(mask.shape[1], device=device).expand_as(mask)
                last = positions.masked_fill(mask == 0, -1).max(dim=1).values
                next_logits = logits[torch.arange(len(prepared), device=device), last]
                for row, (_, letters, ids) in zip(next_logits, prepared):
                    probs = torch.softmax(row[ids].float(), dim=-1)
                    if not torch.isfinite(probs).all():
                        raise ValueError("Model produced non-finite option probabilities")
                    probabilities = dict(zip(letters, probs.cpu().tolist()))
                    results.append({"probabilities": probabilities,
                                    "prediction": max(probabilities, key=probabilities.get)})
        return results
