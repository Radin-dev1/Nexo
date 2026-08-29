# Nexo

Nexo is a complete, original multimodal AI project built with PyTorch and the Hugging Face Transformers interface. It includes causal multi-head self-attention, a trainable image patch encoder, learned positional embeddings, custom tokenizer training, model training, terminal chat, image-aware text generation, and a FastAPI service.

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
nexo-chat --model outputs/nexo --image path/to/photo.jpg
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

To send an image through the API, include a PNG or JPEG as `image_base64`. Nexo resizes it to the configured image size, converts it into visual patch tokens, and places those tokens before the text prompt so causal attention can condition the response on the image.

## Train image understanding

Create a JSONL manifest with one image-caption or image-response pair per line. Relative image paths are resolved from the manifest's directory:

```json
{"image":"photos/dog.jpg","text":"A brown dog running through grass."}
{"image":"charts/sales.png","text":"The chart shows sales increasing from January to March."}
```

Then train the vision encoder and language model together:

```powershell
python train_multimodal.py `
  --manifest data/multimodal.jsonl `
  --tokenizer outputs/nexo-tokenizer `
  --output outputs/nexo-vision `
  --image-size 128 `
  --context-length 256
```

The trainer automatically reserves context positions for visual patches and masks padded text labels from the loss.

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
- Image understanding requires image-caption or visual-instruction training data; adding an image encoder alone does not create pretrained visual knowledge.
- Tiny datasets only verify that training works; they do not create a capable assistant.
- The simple training script is intended as a clear starting point, not a large-scale distributed training system.
- Only train on data you are legally and ethically allowed to use.
