import argparse
from itertools import chain

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from nexo import NexoConfig, NexoForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Nexo language model on a text file.")
    parser.add_argument("--data", required=True, help="UTF-8 training text file")
    parser.add_argument("--output", default="outputs/nexo")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--context-length", type=int, default=256)
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
    return args


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("text", data_files={"train": args.data})["train"]

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=False)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    def group(batch):
        combined = list(chain.from_iterable(batch["input_ids"]))
        size = (len(combined) // args.context_length) * args.context_length
        chunks = [
            combined[i : i + args.context_length]
            for i in range(0, size, args.context_length)
        ]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * len(chunk) for chunk in chunks],
        }

    tokenized = tokenized.map(group, batched=True)
    if len(tokenized) == 0:
        raise ValueError(
            "The corpus did not produce a full training sequence. "
            "Use more text or reduce --context-length."
        )

    config = NexoConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate_size,
    )
    config.auto_map = {
        "AutoConfig": "configuration_nexo.NexoConfig",
        "AutoModel": "modeling_nexo.NexoModel",
        "AutoModelForCausalLM": "modeling_nexo.NexoForCausalLM",
    }
    model = NexoForCausalLM(config)
    training_args = TrainingArguments(
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
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    for filename in ("configuration_nexo.py", "modeling_nexo.py"):
        source = f"nexo/{filename}"
        with open(source, "rb") as src, open(f"{args.output}/{filename}", "wb") as dst:
            dst.write(src.read())


if __name__ == "__main__":
    main()
