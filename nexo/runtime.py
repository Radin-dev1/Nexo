from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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

    def prepare_image(self, image: str | Path | Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        size = self.model.config.image_size
        image = image.convert("RGB").resize((size, size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings | None = None,
        image: str | Path | Image.Image | None = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        settings = settings or GenerationSettings()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if image is not None:
            inputs["pixel_values"] = self.prepare_image(image)
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
