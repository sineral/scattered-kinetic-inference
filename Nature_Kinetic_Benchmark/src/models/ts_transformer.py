"""Time-Series Transformer for Nature kinetic traces.

Early fusion broadcasts catalyst loadings (x1) into every timestep, then a
Transformer encoder with optional readout (mean / attn / cls / attn_mean / mh_attn)
maps the sequence to mechanism logits.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


READOUT_CHOICES = ("mean", "attn", "cls", "attn_mean", "mh_attn")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentionPooling(nn.Module):
    def __init__(self, dim_hidden: int):
        super().__init__()
        self.score = nn.Linear(dim_hidden, 1)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.score(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        return torch.bmm(weights.unsqueeze(1), h).squeeze(1)


class MultiHeadAttentionPooling(nn.Module):
    """Several attention heads over time, then project back to dim_hidden."""

    def __init__(self, dim_hidden: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.score = nn.Linear(dim_hidden, num_heads)
        self.proj = nn.Linear(dim_hidden * num_heads, dim_hidden)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: (B, T, D) -> logits (B, T, H)
        logits = self.score(h)
        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(logits, dim=1)  # over time
        # (B, H, D)
        pooled = torch.einsum("bth,btd->bhd", weights, h)
        return self.proj(pooled.reshape(pooled.size(0), -1))


class TimeSeriesTransformer(nn.Module):
    """Early-fuse Transformer: x1 + x2 -> (T, 16) -> encoder -> readout -> logits."""

    def __init__(
        self,
        dim_x1: int = 4,
        dim_x2: int = 12,
        dim_hidden: int = 128,
        num_classes: int = 20,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        readout: str = "mean",
    ) -> None:
        super().__init__()
        if dim_hidden % num_heads != 0:
            raise ValueError(
                f"dim_hidden ({dim_hidden}) must be divisible by num_heads ({num_heads})"
            )
        readout = readout.lower()
        if readout not in READOUT_CHOICES:
            raise ValueError(
                f"Unknown readout '{readout}'. Choose from: {', '.join(READOUT_CHOICES)}"
            )
        self.readout = readout

        in_dim_total = dim_x1 + dim_x2
        self.input_projection = nn.Sequential(
            nn.LayerNorm(in_dim_total),
            nn.Linear(in_dim_total, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.pos_encoder = PositionalEncoding(d_model=dim_hidden)

        if readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_hidden))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=num_heads,
            dim_feedforward=dim_hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attn_pool = None
        self.mh_attn_pool = None
        self.attn_mean_gate = None
        if readout in {"attn", "attn_mean"}:
            self.attn_pool = AttentionPooling(dim_hidden)
        if readout == "attn_mean":
            # scalar gate: sigmoid(g)*attn + (1-sigmoid(g))*mean
            self.attn_mean_gate = nn.Parameter(torch.zeros(1))
        if readout == "mh_attn":
            self.mh_attn_pool = MultiHeadAttentionPooling(dim_hidden, num_heads=num_heads)

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden // 2, num_classes),
        )

    def _mean_pool(self, h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return h.mean(dim=1)
        mask_f = mask.unsqueeze(-1).float()
        return (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

    def _readout(
        self,
        encoded: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.readout == "cls":
            return encoded[:, 0, :]
        if self.readout == "attn":
            return self.attn_pool(encoded, mask)
        if self.readout == "mh_attn":
            return self.mh_attn_pool(encoded, mask)
        if self.readout == "attn_mean":
            h_attn = self.attn_pool(encoded, mask)
            h_mean = self._mean_pool(encoded, mask)
            gate = torch.sigmoid(self.attn_mean_gate)
            return gate * h_attn + (1.0 - gate) * h_mean
        return self._mean_pool(encoded, mask)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x1, x2 = inputs
        B, seq_len, _ = x2.shape

        x1_expanded = x1.unsqueeze(1).expand(-1, seq_len, -1)
        combined = torch.cat([x1_expanded, x2], dim=-1)  # (B, T, 16)

        h = self.input_projection(combined)
        key_padding_mask = ~mask if mask is not None else None

        if self.readout == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)
            h = self.pos_encoder(h)
            if key_padding_mask is not None:
                cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x2.device)
                key_padding_mask = torch.cat([cls_pad, key_padding_mask], dim=1)
            encoded = self.transformer_encoder(h, src_key_padding_mask=key_padding_mask)
        else:
            h = self.pos_encoder(h)
            encoded = self.transformer_encoder(h, src_key_padding_mask=key_padding_mask)

        pooled = self._readout(encoded, mask)
        return self.classifier(pooled)
