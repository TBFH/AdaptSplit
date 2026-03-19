from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation: type[nn.Module] = nn.ReLU,
        output_activation: nn.Module | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation)
        # layers.append(nn.Tanh())    # constrain to [-1, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RewardModel(nn.Module):
    """Decoder f_psi in LaRe: latent reward factors -> scalar proxy reward."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        self.decoder = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims, activation=nn.ReLU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
