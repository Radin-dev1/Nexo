import json
from pathlib import Path


def test_kaggle_notebook_contains_training_and_hub_upload():
    path = Path("notebooks/kaggle_train_nexo.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )
    assert "train_multimodal.py" in source
    assert "UserSecretsClient().get_secret('HF_TOKEN')" in source
    assert "api.upload_folder" in source
    assert "AutoModelForCausalLM.from_pretrained" in source
