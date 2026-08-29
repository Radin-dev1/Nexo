import math
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput

from .configuration_nexo import NexoConfig


class NexoAttention(nn.Module):
    def __init__(self, config: NexoConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = config.attention_dropout

    def forward(self, hidden_states, attention_mask=None):
        batch_size, sequence_length, width = hidden_states.shape
        qkv = self.qkv(hidden_states).view(
            batch_size, sequence_length, 3, self.num_heads, self.head_size
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        causal = torch.ones(
            sequence_length, sequence_length, dtype=torch.bool, device=hidden_states.device
        ).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
        context = (weights @ value).transpose(1, 2).contiguous().view(
            batch_size, sequence_length, width
        )
        return self.out(context)


class NexoBlock(nn.Module):
    def __init__(self, config: NexoConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attention = NexoAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout),
        )

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = hidden_states + self.attention(self.ln_1(hidden_states), attention_mask)
        return hidden_states + self.mlp(self.ln_2(hidden_states))


class NexoPreTrainedModel(PreTrainedModel):
    config_class = NexoConfig
    base_model_prefix = "nexo"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


class NexoModel(NexoPreTrainedModel):
    def __init__(self, config: NexoConfig):
        super().__init__(config)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([NexoBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, value):
        self.token_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        del kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        sequence_length = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError("Input is longer than max_position_embeddings")
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)
        hidden_states = self.dropout(inputs_embeds + self.position_embeddings(position_ids))
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return BaseModelOutput(last_hidden_state=self.final_norm(hidden_states))


class NexoForCausalLM(NexoPreTrainedModel, GenerationMixin):
    _tied_weights_keys: ClassVar[dict[str, str]] = {
        "lm_head.weight": "nexo.token_embeddings.weight"
    }

    def __init__(self, config: NexoConfig):
        super().__init__(config)
        self.nexo = NexoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vision_encoder = nn.Conv2d(
            config.num_image_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.vision_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.nexo.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.nexo.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, pixel_values=None, **kwargs
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

    def encode_image(self, pixel_values):
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [batch, channels, height, width]")
        patches = self.vision_encoder(pixel_values).flatten(2).transpose(1, 2)
        return self.vision_norm(patches)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        **kwargs,
    ):
        text_embeddings = self.get_input_embeddings()(input_ids)
        model_attention_mask = attention_mask
        if pixel_values is not None:
            image_embeddings = self.encode_image(pixel_values)
            text_embeddings = torch.cat((image_embeddings, text_embeddings), dim=1)
            image_mask = torch.ones(
                image_embeddings.shape[:2],
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=input_ids.device,
            )
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            model_attention_mask = torch.cat((image_mask, attention_mask), dim=1)
        outputs = self.nexo(
            inputs_embeds=text_embeddings,
            attention_mask=model_attention_mask,
            **kwargs,
        )
        text_hidden_states = outputs.last_hidden_state[:, -input_ids.shape[1] :]
        logits = self.lm_head(text_hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
