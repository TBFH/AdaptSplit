from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from adaptsplit.agent.reward_model.module import RewardModel


@dataclass
class RDConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    reward_lr: float = 3e-4
    device: str = "cpu"


class RDRewardDecomposer(nn.Module):
    """State-based Return Decomposition (RD) baseline.

    This baseline is included because LaRe-RD reuses the same update objective but replaces the raw
    state/action input with LLM-defined latent reward factors. Compared with the original MuJoCo
    repo, this implementation is closer to Eq. (6) in the paper: the sum of predicted step rewards
    is regressed to the trajectory-level episodic return.
    """

    def __init__(self, config: RDConfig) -> None:
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.device = torch.device(config.device)
        self.reward_model = RewardModel(input_dim=config.state_dim * 2 + config.action_dim, hidden_dims=(config.hidden_dim, config.hidden_dim))
        self.reward_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=config.reward_lr)
        self.loss_fn = nn.MSELoss(reduction="mean")

    def _one_hot_action(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() == 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        elif actions.dim() == 2 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        actions = actions.long()
        return F.one_hot(actions, num_classes=self.action_dim).float()

    def build_features(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        action_oh = self._one_hot_action(actions)
        delta = states - next_states
        return torch.cat([states, action_oh, delta], dim=-1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        features = self.build_features(states, actions, next_states)
        return self.reward_model(features)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        episodic_returns: torch.Tensor,
        masks: torch.Tensor,
    ) -> Dict[str, float]:
        self.train()
        self.optimizer.zero_grad()
        step_rewards = self.forward(states, actions, next_states).squeeze(-1)
        pred_returns = (step_rewards * masks).sum(dim=1)
        target_returns = episodic_returns.reshape(-1)
        loss = self.loss_fn(pred_returns, target_returns)
        loss.backward()
        self.optimizer.step()
        return {"rd_loss": float(loss.item())}
