import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from nexo import NexoConfig, NexoForCausalLM


class ImageTextDataset(Dataset):
    def __init__(self, manifest, tokenizer, context_length, image_size, patch_size):
        self.manifest_dir = Path(manifest).resolve().parent
        with open(manifest, encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError("The JSONL manifest is empty")
        visual_tokens = (image_size // patch_size) ** 2
        self.text_length = context_length - visual_tokens
        if self.text_length < 2:
            raise ValueError("context_length must leave room for at least two text tokens")
        self.tokenizer = tokenizer
        self.image_size = image_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = self.manifest_dir / image_path
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size)
        )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = torch.from_numpy((pixels - 0.5) / 0.5).permute(2, 0, 1)
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.text_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixels,
            "labels": labels,
        }


def main():
    parser = argparse.ArgumentParser(description="Train Nexo on image-text pairs.")
    parser.add_argument("--manifest", required=True, help="JSONL with image and text fields")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", default="outputs/nexo-vision")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = ImageTextDataset(
        args.manifest,
        tokenizer,
        args.context_length,
        args.image_size,
        args.patch_size,
    )
    config = NexoConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.context_length,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config.auto_map = {
        "AutoConfig": "configuration_nexo.NexoConfig",
        "AutoModel": "modeling_nexo.NexoModel",
        "AutoModelForCausalLM": "modeling_nexo.NexoForCausalLM",
    }
    model = NexoForCausalLM(config)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
        ),
        train_dataset=dataset,
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    for filename in ("configuration_nexo.py", "modeling_nexo.py"):
        source = Path("nexo") / filename
        (Path(args.output) / filename).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
