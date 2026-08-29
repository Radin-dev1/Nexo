import argparse

from .runtime import GenerationSettings, NexoRuntime


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained Nexo model.")
    parser.add_argument("--model", default="outputs/nexo")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--image", help="Optional image to discuss")
    args = parser.parse_args()
    runtime = NexoRuntime(args.model)
    settings = GenerationSettings(max_new_tokens=args.max_new_tokens)
    print("Nexo is ready. Type /quit to exit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in {"/quit", "/exit"}:
            break
        if not prompt:
            continue
        print(f"Nexo: {runtime.generate(prompt, settings, image=args.image)}")


if __name__ == "__main__":
    main()
