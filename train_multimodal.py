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
        if "image" not in row or "text" not in row:
            raise ValueError("Each manifest row must contain 'image' and 'text' fields")

        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = self.manifest_dir / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train Nexo on image-text pairs.")
    parser.add_argument("--manifest", required=True, help="JSONL with image and text fields")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--base-model", default=None, help="Optional pretrained Nexo checkpoint")
    parser.add_argument("--output", default="outputs/nexo-vision")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    if args.fp16 and args.bf16:
        parser.error("Choose either --fp16 or --bf16, not both")
    if args.image_size % args.patch_size != 0:
        parser.error("--image-size must be divisible by --patch-size")
    return args


def main():
    args = parse_args()
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

    if args.base_model:
        model = NexoForCausalLM.from_pretrained(args.base_model)
        if len(tokenizer) != model.config.vocab_size:
            raise ValueError(
                "Tokenizer vocabulary does not match the base model. "
                "Use the tokenizer that was used to train the checkpoint."
            )
        if model.config.patch_size != args.patch_size:
            raise ValueError(
                "The base model patch size does not match --patch-size. "
                "Changing patch size would change the vision encoder weights."
            )
        if args.context_length > model.config.max_position_embeddings:
            raise ValueError(
                "--context-length cannot exceed the base model's max_position_embeddings"
            )
        model.config.image_size = args.image_size
    else:
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
        model = NexoForCausalLM(config)

    model.config.auto_map = {
        "AutoConfig": "configuration_nexo.NexoConfig",
        "AutoModel": "modeling_nexo.NexoModel",
        "AutoModelForCausalLM": "modeling_nexo.NexoForCausalLM",
    }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            save_total_limit=args.save_total_limit,
            report_to="none",
            remove_unused_columns=False,
            fp16=args.fp16,
            bf16=args.bf16,
            seed=args.seed,
            dataloader_num_workers=2,
        ),
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    for filename in ("configuration_nexo.py", "modeling_nexo.py"):
        source = Path("nexo") / filename
        (Path(args.output) / filename).write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
