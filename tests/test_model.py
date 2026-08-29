import torch

from nexo import NexoConfig, NexoForCausalLM


def tiny_model():
    config = NexoConfig(
        vocab_size=101,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return NexoForCausalLM(config)


def test_forward_and_loss():
    model = tiny_model()
    input_ids = torch.randint(0, 101, (2, 12))
    output = model(input_ids=input_ids, labels=input_ids)
    assert output.logits.shape == (2, 12, 101)
    assert torch.isfinite(output.loss)


def test_causal_mask_blocks_future_tokens():
    model = tiny_model().eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 99, 98]])
    with torch.inference_mode():
        first = model(prefix).logits[:, 1]
        second = model(changed_future).logits[:, 1]
    torch.testing.assert_close(first, second)
