import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast


def main():
    parser = argparse.ArgumentParser(description="Train Nexo's byte-level BPE tokenizer.")
    parser.add_argument("--data", nargs="+", required=True, help="One or more UTF-8 text files")
    parser.add_argument("--output", default="outputs/nexo-tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=args.data,
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"],
    )
    tokenizer.save_model(str(output))
    fast = GPT2TokenizerFast(
        vocab_file=str(output / "vocab.json"),
        merges_file=str(output / "merges.txt"),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
    )
    fast.save_pretrained(output)
    print(f"Tokenizer saved to {output}")


if __name__ == "__main__":
    main()
