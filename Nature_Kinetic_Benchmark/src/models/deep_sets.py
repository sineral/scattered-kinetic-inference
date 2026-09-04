"""DeepSets baseline: phi MLP, masked mean pool, rho classifier."""

from typing import Optional

import torch
import torch.nn as nn


class DeepSets(nn.Module):
    """Permutation-invariant classifier over unordered [cat0, t, S, P] points.

    Parameters
    ----------
    in_dim : int
        Feature dimension of each set element (4 for raw kinetics).
    dim_hidden : int
        Hidden width of phi and rho.
    latent_dim : int
        Output width of phi before pooling.
    num_classes : int
        Number of mechanism classes.
    dropout : float
        Dropout in the classifier.
    """

    def __init__(
        self,
        in_dim: int = 4,
        dim_hidden: int = 128,
        latent_dim: int = 128,
        num_classes: int = 20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, dim_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, latent_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(latent_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(latent_dim, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, M, in_dim), mask: (B, M) True = valid element.
        """
        z = self.phi(x)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            z = z * mask_f
            global_feat = z.sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)
        else:
            global_feat = z.mean(dim=1)
        return self.rho(global_feat)
