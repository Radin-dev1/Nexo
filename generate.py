import argparse

from nexo.runtime import GenerationSettings, NexoRuntime


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained Nexo model.")
    parser.add_argument("--model", default="outputs/nexo")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    runtime = NexoRuntime(args.model)
    print(runtime.generate(args.prompt, GenerationSettings(max_new_tokens=args.max_new_tokens)))


if __name__ == "__main__":
    main()
