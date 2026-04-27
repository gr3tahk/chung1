from __future__ import annotations


class LocalVisionClient:
    """Shared local Hugging Face image-text client for VLM agents."""

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
        self._processor = None
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
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=self._torch_dtype(),
            device_map=self.device_map,
            trust_remote_code=True,
        )
        return self._processor, self._model

    def generate(self, prompt: str, image) -> str:
        processor, model = self.load()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        generated = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, outputs, strict=True)
        ]
        decoded = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()


_VISION_CLIENTS: dict[tuple[str, str, str, int], LocalVisionClient] = {}


def get_local_vision_client(
    model_name: str,
    dtype: str = "auto",
    device_map: str = "auto",
    max_new_tokens: int = 160,
) -> LocalVisionClient:
    key = (model_name, dtype, device_map, max_new_tokens)
    if key not in _VISION_CLIENTS:
        _VISION_CLIENTS[key] = LocalVisionClient(
            model_name=model_name,
            dtype=dtype,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
        )
    return _VISION_CLIENTS[key]
