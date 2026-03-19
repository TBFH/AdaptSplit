from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from adaptsplit.agent.reward_model.chat_with_llm import LLMCallConfig, callgpt
from adaptsplit.agent.reward_model.module import RewardModel
from adaptsplit.agent.reward_model.RD import RDConfig, RDRewardDecomposer


@dataclass
class LLMRDConfig(RDConfig):
    llm_model: str = "gpt-4.1"
    llm_temperature: float = 0.2
    llm_candidates: int = 5
    llm_response_dir: str = "./llm_reward_cache"
    response_id: int = 0


class LLMRDRewardDecomposer(RDRewardDecomposer):
    """LaRe-RD: LLM latent reward encoder + RD reward decoder."""

    def __init__(self, config: LLMRDConfig, prompt_context: Dict[str, Any]) -> None:
        super().__init__(config)
        self.config = config
        self.prompt_context = prompt_context
        self.response_dir = Path(config.llm_response_dir)
        self.rd_functions: List[str] = []
        self.factor_num = 0
        self._load_or_generate_functions()
        self.reward_model = RewardModel(input_dim=self.factor_num, hidden_dims=(config.hidden_dim, config.hidden_dim))
        self.reward_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=config.reward_lr)

    def _response_file(self) -> Path:
        return self.response_dir / f"response_{self.config.response_id}.npy"

    def _factor_num_file(self) -> Path:
        return self.response_dir / f"factor_num_{self.config.response_id}.npy"

    def _safe_exec(self, code: str):
        namespace: Dict[str, Any] = {"np": np}
        exec(code, namespace)
        if "evaluation_func" not in namespace:
            raise ValueError("Generated code does not contain evaluation_func")
        return namespace["evaluation_func"]

    def _load_or_generate_functions(self) -> None:
        self.response_dir.mkdir(parents=True, exist_ok=True)
        if not self._response_file().exists():
            callgpt(
                env_context=self.prompt_context,
                save_dir=str(self.response_dir),
                response_id=self.config.response_id,
                config=LLMCallConfig(
                    model=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    n_candidates=self.config.llm_candidates,
                    factor=True,
                ),
            )
        responses = np.load(self._response_file(), allow_pickle=True)
        self.factor_num = int(np.load(self._factor_num_file(), allow_pickle=True).item())
        self.rd_functions = [json.loads(str(resp))["Functions"] for resp in responses]

    def latent_forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self.rd_functions:
            raise RuntimeError("No LLM-generated latent reward functions were loaded.")

        func = self._safe_exec(self.rd_functions[0])
        device = states.device
        state_shape = states.shape

        flat_states = states.detach().cpu().numpy().reshape(-1, state_shape[-1])
        if actions.dim() == states.dim():
            flat_actions = actions.detach().cpu().numpy().reshape(-1, actions.shape[-1])
        else:
            flat_actions = actions.detach().cpu().numpy().reshape(-1, 1)

        factor_outputs = func(flat_states, flat_actions)
        factor_outputs = [np.asarray(arr, dtype=np.float32) for arr in factor_outputs]
        cat = np.concatenate(factor_outputs, axis=-1)
        if len(state_shape) == 3:
            tensor = torch.as_tensor(cat, device=device, dtype=torch.float32).reshape(state_shape[0], state_shape[1], -1)
        else:
            tensor = torch.as_tensor(cat, device=device, dtype=torch.float32).reshape(state_shape[0], -1)
        return tensor

    def forward(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        latent_rewards = self.latent_forward(states, actions)

        # batch-wise normalization
        if latent_rewards.dim() == 3:
            mean = latent_rewards.mean(dim=(0, 1), keepdim=True)
            std = latent_rewards.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        else:
            mean = latent_rewards.mean(dim=0, keepdim=True)
            std = latent_rewards.std(dim=0, keepdim=True).clamp_min(1e-6)

        latent_rewards = (latent_rewards - mean) / std
        latent_rewards = torch.clamp(latent_rewards, -5.0, 5.0)

        print(f"[LLMrd forward] latent_rewards.mean: {latent_rewards.mean()}")
        return self.reward_model(latent_rewards)
