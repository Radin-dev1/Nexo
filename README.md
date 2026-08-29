# Nexo

Nexo is a complete, original decoder-only AI project built with PyTorch and the Hugging Face Transformers interface. It includes causal multi-head self-attention, learned positional embeddings, custom tokenizer training, model training, terminal chat, text generation, and a FastAPI service.

The default model has about 30 million parameters. It starts with random weights: training data is what gives it useful behavior.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python train_tokenizer.py --data data/example.txt --output outputs/nexo-tokenizer
python train.py --data data/example.txt --tokenizer outputs/nexo-tokenizer --output outputs/nexo --epochs 1
python generate.py --model outputs/nexo --prompt "Nexo is"
nexo-chat --model outputs/nexo
```

For a useful model, replace `data/example.txt` with a substantial, cleaned text corpus you have permission to use. A GPU is strongly recommended. Adjust the architecture in `NexoConfig` or add command-line options before committing to a long training run.

## API server

```powershell
$env:NEXO_MODEL = "outputs/nexo"
nexo-serve --port 8000
```

Open `http://localhost:8000/docs` for the interactive API. Generate text with:

```powershell
curl.exe -X POST http://localhost:8000/v1/generate `
  -H "Content-Type: application/json" `
  -d '{"prompt":"Hello Nexo","max_new_tokens":80}'
```

Docker users can run `docker compose up --build` after placing trained weights in `outputs/nexo`.

## Architecture controls

`train.py` exposes `--hidden-size`, `--layers`, `--heads`, `--intermediate-size`, and `--context-length`. The defaults create a roughly 30M-parameter model. These values can be increased for more capacity when the dataset and available GPU compute justify it.

## Load with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("YOUR_HF_USERNAME/nexo")
model = AutoModelForCausalLM.from_pretrained(
    "YOUR_HF_USERNAME/nexo",
    trust_remote_code=True,
)
```

## Publish to Hugging Face

After authenticating with the current `hf` command-line client:

```powershell
hf auth login
hf upload YOUR_HF_USERNAME/nexo outputs/nexo --repo-type model
```

Review the generated files and model card before uploading. Publishing weights is an external action and should only be done after training and evaluation.

## Important limitations

- This repository provides the model architecture and training pipeline, not pretrained weights.
- Tiny datasets only verify that training works; they do not create a capable assistant.
- The simple training script is intended as a clear starting point, not a large-scale distributed training system.
- Only train on data you are legally and ethically allowed to use.
