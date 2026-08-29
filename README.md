# Nexo

Nexo is a compact, original decoder-only language model built with PyTorch and the Hugging Face Transformers interface. It has causal multi-head self-attention, learned positional embeddings, GELU feed-forward blocks, tied token/output embeddings, and a configurable architecture.

The default model has about 30 million parameters. It starts with random weights: training data is what gives it useful behavior.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python train.py --data data/example.txt --output outputs/nexo --epochs 1
python generate.py --model outputs/nexo --prompt "Nexo is"
```

For a useful model, replace `data/example.txt` with a substantial, cleaned text corpus you have permission to use. A GPU is strongly recommended. Adjust the architecture in `NexoConfig` or add command-line options before committing to a long training run.

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
