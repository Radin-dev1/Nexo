import math
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
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

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        del kwargs
        _, sequence_length = input_ids.shape
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError("Input is longer than max_position_embeddings")
        if position_ids is None:
            position_ids = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        hidden_states = self.dropout(
            self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return BaseModelOutput(last_hidden_state=self.final_norm(hidden_states))


class NexoForCausalLM(NexoPreTrainedModel):
    _tied_weights_keys: ClassVar[list[str]] = ["lm_head.weight"]

    def __init__(self, config: NexoConfig):
        super().__init__(config)
        self.nexo = NexoModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.nexo.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.nexo.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.nexo(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
