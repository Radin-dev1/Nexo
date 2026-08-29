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
        image_size=16,
        patch_size=8,
        bos_token_id=100,
        eos_token_id=100,
    )
    return NexoForCausalLM(config)


def test_forward_and_loss():
    model = tiny_model()
    input_ids = torch.randint(0, 101, (2, 12))
    output = model(input_ids=input_ids, labels=input_ids)
    assert output.logits.shape == (2, 12, 101)
    assert torch.isfinite(output.loss)


def test_base_model_accepts_precomputed_embeddings():
    model = tiny_model()
    embeddings = torch.rand(2, 6, model.config.hidden_size)
    output = model.nexo(inputs_embeds=embeddings)
    assert output.last_hidden_state.shape == embeddings.shape


def test_causal_mask_blocks_future_tokens():
    model = tiny_model().eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 99, 98]])
    with torch.inference_mode():
        first = model(prefix).logits[:, 1]
        second = model(changed_future).logits[:, 1]
    torch.testing.assert_close(first, second)


def test_image_conditioning_preserves_text_output_shape():
    model = tiny_model()
    input_ids = torch.randint(0, 101, (2, 8))
    pixels = torch.rand(2, 3, 16, 16)
    output = model(input_ids=input_ids, pixel_values=pixels, labels=input_ids)
    assert output.logits.shape == (2, 8, 101)
    assert torch.isfinite(output.loss)


def test_image_conditioned_generation():
    model = tiny_model().eval()
    input_ids = torch.tensor([[1, 2, 3]])
    pixels = torch.rand(1, 3, 16, 16)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            pixel_values=pixels,
            max_new_tokens=2,
            do_sample=False,
        )
    assert output.shape == (1, 5)
