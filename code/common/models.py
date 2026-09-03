"""Small and confirmatory decoder models with inspectable ordinary MHA."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .geometry import Coefficients, initialize_attention_module


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 128
    width: int = 128
    layers: int = 1
    heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.0

    @property
    def head_width(self) -> int:
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        return self.width // self.heads

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class InspectableSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.width = config.width
        self.heads = config.heads
        self.head_width = config.head_width
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.width, config.width, bias=False)
        self.k_proj = nn.Linear(config.width, config.width, bias=False)
        self.v_proj = nn.Linear(config.width, config.width, bias=False)
        self.out_proj = nn.Linear(config.width, config.width, bias=False)
        self.last_head_output: Tensor | None = None

    def forward(
        self,
        x: Tensor,
        *,
        causal: bool = True,
        return_attention: bool = False,
        head_keep: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.heads, self.head_width).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)
        if causal:
            mask = torch.ones((length, length), dtype=torch.bool, device=x.device).tril()
            logits = logits.masked_fill(~mask, float("-inf"))
        probabilities = F.softmax(logits, dim=-1)
        probabilities = F.dropout(probabilities, self.dropout, self.training)
        head_output = probabilities @ v
        self.last_head_output = head_output.detach() if return_attention else None
        if head_keep is not None:
            head_output = head_output * head_keep.view(1, self.heads, 1, 1)
        merged = head_output.transpose(1, 2).contiguous().view(batch, length, self.width)
        return self.out_proj(merged), probabilities if return_attention else None


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = InspectableSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.width)
        hidden = config.width * config.ff_multiplier
        self.feed_forward = nn.Sequential(
            nn.Linear(config.width, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.width),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        *,
        return_attention: bool,
        head_keep: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        attended, probabilities = self.attention(
            self.norm1(x),
            return_attention=return_attention,
            head_keep=head_keep,
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x, probabilities


class DecoderLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(module.in_features))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0 / math.sqrt(self.config.width))

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        *,
        return_attention: bool = False,
        ablate: tuple[int, int] | None = None,
    ) -> dict[str, Tensor | list[Tensor]]:
        _, length = tokens.shape
        if length > self.config.context_length:
            raise ValueError("sequence exceeds configured context length")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        attention_maps: list[Tensor] = []
        for layer, block in enumerate(self.blocks):
            keep = None
            if ablate is not None and ablate[0] == layer:
                keep = torch.ones(self.config.heads, device=tokens.device)
                keep[ablate[1]] = 0
            x, attention = block(
                x, return_attention=return_attention, head_keep=keep
            )
            if attention is not None:
                attention_maps.append(attention)
        logits = self.lm_head(self.final_norm(x))
        output: dict[str, Tensor | list[Tensor]] = {"logits": logits}
        if targets is not None:
            output["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        if return_attention:
            output["attention"] = attention_maps
        return output

    def apply_query_key_geometry(
        self,
        layer_coefficients: Sequence[Sequence[Coefficients]],
        *,
        seed: int,
    ) -> list[dict[str, int | float | None]]:
        if len(layer_coefficients) != len(self.blocks):
            raise ValueError("one coefficient list is required per layer")
        records: list[dict[str, int | float | None]] = []
        for layer, (block, coefficients) in enumerate(
            zip(self.blocks, layer_coefficients, strict=True)
        ):
            if len(coefficients) != self.config.heads:
                raise ValueError("one coefficient tuple is required per head")
            layer_records = initialize_attention_module(
                block.attention,
                coefficients,
                seed=seed + 1_000_003 * layer,
            )
            records.extend({"layer": layer, **row} for row in layer_records)
        return records


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def query_key_state(model: DecoderLM) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for layer, block in enumerate(model.blocks):
        state[f"blocks.{layer}.attention.q_proj.weight"] = (
            block.attention.q_proj.weight.detach().cpu().clone()
        )
        state[f"blocks.{layer}.attention.k_proj.weight"] = (
            block.attention.k_proj.weight.detach().cpu().clone()
        )
    return state
