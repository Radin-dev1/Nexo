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
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    return parser.parse_args()


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
        chunks = [combined[i : i + args.context_length] for i in range(0, size, args.context_length)]
        return {"input_ids": chunks, "attention_mask": [[1] * len(x) for x in chunks]}

    tokenized = tokenized.map(group, batched=True)
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
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    for filename in ("configuration_nexo.py", "modeling_nexo.py"):
        source = f"nexo/{filename}"
        with open(source, "rb") as src, open(f"{args.output}/{filename}", "wb") as dst:
            dst.write(src.read())


if __name__ == "__main__":
    main()
