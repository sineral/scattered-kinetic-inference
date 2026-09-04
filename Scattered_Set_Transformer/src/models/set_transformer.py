"""
Full Set Transformer implementation with Induced Set Attention Blocks (ISAB)
and Pooling by Multihead Attention (PMA).

This module implements the architecture described in "Set Transformer: A
Framework for Attention-based Permutation-Invariant Neural Networks" by
Lee et al. (2019).  The key components are:

* **Set Attention Block (SAB)**: multihead self-attention followed by
  feed-forward layers with residual connections and layer normalisation.
* **Induced Set Attention Block (ISAB)**: reduces self-attention
  complexity by introducing a fixed number of learnable inducing
  vectors.  Attention is computed from these inducing points to the
  input set and back.
* **Pooling by Multihead Attention (PMA)**: aggregates a variable-size
  input set into a fixed number of outputs using learnable seed
  vectors.

This implementation supports masking for padded set elements and
exposes a single `SetTransformer` class that can be used in place
of simpler architectures.  Hyperparameters such as the number of
inducing points, number of layers and number of seeds for PMA can be
controlled via constructor arguments.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer feed-forward network with ReLU activation."""

    def __init__(self, dim: int, hidden_multiplier: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_multiplier * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_multiplier * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureGating(nn.Module):
    """Feature gating mechanism to learn feature importance/reliability.
    
    This layer learns to weight different input features based on their
    importance or reliability, which is particularly useful when dealing
    with noisy data where some features may be more robust than others.
    """

    def __init__(self, in_dim: int, reduction: int = 4) -> None:
        super().__init__()
        # Squeeze-and-excitation style gating
        self.gate = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // reduction, in_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n, dim)
        # Compute global average across set dimension for context
        context = x.mean(dim=1, keepdim=True)  # (batch, 1, dim)
        gates = self.gate(context)  # (batch, 1, dim)
        return x * gates  # Element-wise gating


class SAB(nn.Module):
    """Set Attention Block: self-attention followed by feed-forward with residual connections."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.proj_in = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        self.mha = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim_out)
        self.ff = FeedForward(dim_out, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, n, dim_in)
        h = self.proj_in(x)
        # key_padding_mask uses True at positions that should be ignored (i.e. padding)
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.mha(h, h, h, key_padding_mask=key_padding_mask)
        h = self.ln1(h + attn_out)
        ff_out = self.ff(h)
        out = self.ln2(h + ff_out)
        return out


class ISAB(nn.Module):
    """Induced Set Attention Block.

    Introduces `m` learnable inducing points to reduce the quadratic cost of
    self-attention to O(nm).
    """

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inducing: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_inducing = num_inducing
        # learnable inducing points: (m, dim_out)
        self.inducing = nn.Parameter(torch.randn(num_inducing, dim_out))
        self.proj_in = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        # first attention: from inducing points to input set
        self.mha1 = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim_out)
        # second attention: from input set to induced representation
        self.mha2 = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim_out)
        self.ff = FeedForward(dim_out, dropout=dropout)
        self.ln3 = nn.LayerNorm(dim_out)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, n, dim_in)
        B, N, _ = x.shape
        h = self.proj_in(x)
        # replicate inducing points across the batch: (batch, m, dim_out)
        I = self.inducing.unsqueeze(0).expand(B, -1, -1)
        # first attention: query=I, key/value=h
        # mask applies to the keys (input set)
        key_padding_mask = ~mask if mask is not None else None
        H, _ = self.mha1(I, h, h, key_padding_mask=key_padding_mask)
        H = self.ln1(I + H)  # residual on inducing points
        # second attention: query=h, key/value=H
        # no mask required here because H has no padding
        Y, _ = self.mha2(h, H, H)
        Y = self.ln2(h + Y)
        ff_out = self.ff(Y)
        out = self.ln3(Y + ff_out)
        return out


class PMA(nn.Module):
    """Pooling by Multihead Attention.

    Aggregates a variable-size set into `k` outputs using `k` learnable seed vectors.
    """

    def __init__(self, dim: int, num_heads: int, num_seeds: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(num_seeds, dim))
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, n, dim)
        B = x.size(0)
        # replicate seeds for each batch
        seeds = self.seeds.unsqueeze(0).expand(B, -1, -1)  # (batch, k, dim)
        # use key_padding_mask for the input set
        key_padding_mask = ~mask if mask is not None else None
        H, _ = self.mha(seeds, x, x, key_padding_mask=key_padding_mask)
        H = self.ln(seeds + H)
        ff_out = self.ff(H)
        out = self.ln2(H + ff_out)
        return out  # shape: (batch, k, dim)


class SetTransformer(nn.Module):
    """Full Set Transformer with ISAB encoder and PMA pooling.

    Parameters
    ----------
    in_dim: dimension of each input element.
    dim_hidden: internal hidden dimension for attention and feed-forward layers.
    num_classes: number of output classes.
    num_heads: number of attention heads.
    num_layers: number of ISAB blocks in the encoder.
    num_inducing: number of inducing points for each ISAB.
    num_pma_seeds: number of seed vectors in PMA pooling (use 1 for classification).
    dropout: dropout rate applied in attention and feed-forward layers.
    use_feature_gating: whether to use feature gating for learning feature importance.
    num_classifier_heads: number of parallel classification heads (ensemble for robustness).
    """

    def __init__(
        self,
        in_dim: int,
        dim_hidden: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 6,
        num_inducing: int = 16,
        num_pma_seeds: int = 1,
        dropout: float = 0.1,
        use_feature_gating: bool = True,
        num_classifier_heads: int = 1,
    ) -> None:
        super().__init__()
        self.use_feature_gating = use_feature_gating
        self.num_classifier_heads = num_classifier_heads
        
        # Enhanced input processing with optional feature gating
        if use_feature_gating:
            self.input_processing = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, dim_hidden),
                nn.ReLU(inplace=True),
                FeatureGating(dim_hidden),
                nn.Dropout(dropout),
            )
        else:
            self.input_processing = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, dim_hidden),
            )
        
        # encoder: stack of ISAB blocks
        self.encoder = nn.ModuleList([
            ISAB(
                dim_in=dim_hidden,
                dim_out=dim_hidden,
                num_heads=num_heads,
                num_inducing=num_inducing,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        # pooling
        self.pma = PMA(dim_hidden, num_heads, num_pma_seeds, dropout=dropout)
        
        # Multi-head classification (ensemble for robustness)
        if num_classifier_heads > 1:
            self.fc = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(dim_hidden * num_pma_seeds),
                    nn.Linear(dim_hidden * num_pma_seeds, dim_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(dim_hidden, num_classes),
                )
                for _ in range(num_classifier_heads)
            ])
        else:
            # Single classification head
            self.fc = nn.Sequential(
                nn.LayerNorm(dim_hidden * num_pma_seeds),
                nn.Linear(dim_hidden * num_pma_seeds, dim_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(dim_hidden, num_classes),
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, n, in_dim)
        # Enhanced input processing with optional gating
        h = self.input_processing(x)
        
        # Encoder blocks
        for block in self.encoder:
            h = block(h, mask)
        
        # Pooling
        pooled = self.pma(h, mask)  # (batch, k, dim_hidden)
        pooled_flat = pooled.view(pooled.size(0), -1)
        
        # Multi-head classification (ensemble) or single head
        if self.num_classifier_heads > 1:
            # Average predictions from multiple heads for robustness
            logits = torch.stack([head(pooled_flat) for head in self.fc], dim=0).mean(dim=0)
        else:
            logits = self.fc(pooled_flat)
        
        return logits