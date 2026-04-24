from __future__ import annotations


class LocalTextClient:
    """Shared local Hugging Face causal-LM client for both text agents."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 160,
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _torch_dtype(self):
        if self.dtype == "auto":
            return "auto"
        import torch

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if self.dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return dtype_map[self.dtype]

    def load(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self._torch_dtype(),
            device_map=self.device_map,
            trust_remote_code=True,
        )
        return self._tokenizer, self._model

    def generate(self, prompt: str) -> str:
        tokenizer, model = self.load()
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


_TEXT_CLIENTS: dict[tuple[str, str, str, int], LocalTextClient] = {}


def get_local_text_client(
    model_name: str,
    dtype: str = "auto",
    device_map: str = "auto",
    max_new_tokens: int = 160,
) -> LocalTextClient:
    key = (model_name, dtype, device_map, max_new_tokens)
    if key not in _TEXT_CLIENTS:
        _TEXT_CLIENTS[key] = LocalTextClient(
            model_name=model_name,
            dtype=dtype,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
        )
    return _TEXT_CLIENTS[key]
