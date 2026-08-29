from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class GenerationSettings:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.05


class NexoRuntime:
    """Thread-safe loader and text-generation interface for a trained Nexo model."""

    def __init__(self, model_path: str, device: str | None = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
        ).to(self.device)
        self.model.eval()
        self._lock = threading.Lock()

    def generate(self, prompt: str, settings: GenerationSettings | None = None) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        settings = settings or GenerationSettings()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._lock, torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=settings.temperature > 0,
                temperature=max(settings.temperature, 1e-5),
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
