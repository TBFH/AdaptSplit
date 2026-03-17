from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from adaptsplit.agent.reward_model.module import MLP
from adaptsplit.agent.sentence_embedding.chat import SentenceEmbedder
from adaptsplit.utils import Policy
from adaptsplit.request import Request


@dataclass
class PPOConfig:
    state_dim: int
    action_dim: int
    device: str = "cpu"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    minibatch_size: int = 64
    hidden_dim: int = 128
    reward_warmup_episodes: int = 5
    reward_updates_per_episode: int = 10
    reward_batch_size: int = 16


class RolloutBuffer:
    def __init__(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.infos: List[Dict[str, float]] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, float]] = None,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.dones.append(bool(done))
        self.infos.append(info or {})

    def __len__(self) -> int:
        return len(self.states)

    def as_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "states": torch.as_tensor(np.asarray(self.states), device=device, dtype=torch.float32),
            "actions": torch.as_tensor(np.asarray(self.actions), device=device, dtype=torch.long),
            "log_probs": torch.as_tensor(np.asarray(self.log_probs), device=device, dtype=torch.float32),
            "rewards": torch.as_tensor(np.asarray(self.rewards), device=device, dtype=torch.float32),
            "next_states": torch.as_tensor(np.asarray(self.next_states), device=device, dtype=torch.float32),
            "dones": torch.as_tensor(np.asarray(self.dones), device=device, dtype=torch.float32),
        }


class TrajectoryReplayBuffer:
    """Stores full episodes for training the reward model."""

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.episodes: List[Dict[str, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        episodic_return: float,
    ) -> None:
        episode = {
            "states": states.astype(np.float32),
            "actions": actions.astype(np.int64),
            "next_states": next_states.astype(np.float32),
            "episodic_return": np.asarray(episodic_return, dtype=np.float32),
        }
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if not self.episodes:
            raise ValueError("Cannot sample from an empty TrajectoryReplayBuffer")
        batch_size = min(batch_size, len(self.episodes))
        batch = random.sample(self.episodes, batch_size)
        states = torch.as_tensor(np.stack([ep["states"] for ep in batch], axis=0), device=device, dtype=torch.float32)
        actions = torch.as_tensor(np.stack([ep["actions"] for ep in batch], axis=0), device=device, dtype=torch.long)
        next_states = torch.as_tensor(np.stack([ep["next_states"] for ep in batch], axis=0), device=device, dtype=torch.float32)
        returns = torch.as_tensor(np.stack([ep["episodic_return"] for ep in batch], axis=0), device=device, dtype=torch.float32).reshape(-1)
        masks = torch.ones(states.shape[:2], device=device, dtype=torch.float32)
        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "episodic_returns": returns,
            "masks": masks,
        }


class DiscreteActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.policy = MLP(state_dim, action_dim, hidden_dims=(hidden_dim, hidden_dim), activation=nn.ReLU)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.policy(states)

    def distribution(self, states: torch.Tensor) -> Categorical:
        logits = self.forward(states)
        return Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.value_net = MLP(state_dim, 1, hidden_dims=(hidden_dim, hidden_dim), activation=nn.ReLU)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.value_net(states).squeeze(-1)


class PPOAgent:
    def __init__(self, config: PPOConfig, reward_model: Optional[nn.Module] = None) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.actor = DiscreteActor(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.critic = Critic(config.state_dim, config.hidden_dim).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.reward_model = reward_model

    @torch.no_grad()
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        state_tensor = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        dist = self.actor.distribution(state_tensor)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def train_reward_model(self, replay_buffer: TrajectoryReplayBuffer) -> Dict[str, float]:
        if self.reward_model is None:
            return {}
        if len(replay_buffer) < self.config.reward_warmup_episodes:
            return {}

        metrics: Dict[str, float] = {}
        for _ in range(self.config.reward_updates_per_episode):
            batch = replay_buffer.sample(self.config.reward_batch_size, self.device)
            update_metrics = self.reward_model.update(
                states=batch["states"],
                actions=batch["actions"],
                next_states=batch["next_states"],
                episodic_returns=batch["episodic_returns"],
                masks=batch["masks"],
            )
            metrics = update_metrics
        return metrics

    def _predict_step_rewards(self, rollout: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.reward_model is None:
            return rollout["rewards"]
        if self.reward_model is not None and hasattr(self.reward_model, "forward"):
            with torch.no_grad():
                pred = self.reward_model(
                    rollout["states"].unsqueeze(0),
                    rollout["actions"].unsqueeze(0),
                    rollout["next_states"].unsqueeze(0),
                ).squeeze(0).squeeze(-1)
            return pred
        return rollout["rewards"]

    def _compute_gae(self, rewards: torch.Tensor, states: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_values[t] * not_done - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * not_done * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def train_policy(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
        if len(rollout_buffer) == 0:
            return {}
        rollout = rollout_buffer.as_tensors(self.device)
        training_rewards = self._predict_step_rewards(rollout)
        advantages, returns = self._compute_gae(
            rewards=training_rewards,
            states=rollout["states"],
            next_states=rollout["next_states"],
            dones=rollout["dones"],
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = rollout["states"].shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)
        indices = np.arange(batch_size)
        metrics = {}

        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                mb_states = rollout["states"][mb_idx]
                mb_actions = rollout["actions"][mb_idx]
                mb_old_log_probs = rollout["log_probs"][mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                dist = self.actor.distribution(mb_states)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surrogate1 = ratios * mb_advantages
                surrogate2 = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean() - self.config.entropy_coef * entropy

                values = self.critic(mb_states)
                critic_loss = 0.5 * torch.mean((values - mb_returns) ** 2)

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optim.step()

                metrics = {
                    "actor_loss": float(actor_loss.item()),
                    "critic_loss": float(critic_loss.item()),
                    "entropy": float(entropy.item()),
                    "proxy_reward_mean": float(training_rewards.mean().item()),
                }
        return metrics

    def save(self, path_prefix: str) -> None:
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")
        if self.reward_model is not None:
            torch.save(self.reward_model.state_dict(), f"{path_prefix}_reward_model.pt")

    def load(self, path_prefix: str) -> None:
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pt", map_location=self.device))
        if self.reward_model is not None:
            self.reward_model.load_state_dict(torch.load(f"{path_prefix}_reward_model.pt", map_location=self.device))


class OnlineSchedulerPolicy:
    """
    Lightweight deployment-only policy wrapper.

    Only loads the trained actor for online scheduling.
    Critic and reward_model are not needed for inference.
    """

    def __init__(
        self,
        model: str,
        agent_outputs_dir: str,
        embedder_dir: str,
        device: str = "cuda"
    ):
        self.agent_outputs_dir = Path(agent_outputs_dir)
        self.device = torch.device(device)

        meta_path = self.agent_outputs_dir / f"{model}-deploy_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{model}-deploy_meta.json not found in {self.agent_outputs_dir}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.state_dim = int(meta["state_dim"])
        self.action_dim = int(meta["action_dim"])
        self.hidden_dim = int(meta["hidden_dim"])
        self.action_names: List[str] = list(meta["action_names"])
        self.use_state_norm = bool(meta.get("use_state_norm", False))

        self.actor = DiscreteActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        actor_path = self.agent_outputs_dir / f"{model}-final_actor.pt"
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor.eval()

        self.embedder = SentenceEmbedder(
            save_dir=embedder_dir,
            response_id=0
        )

        self.state_mean = None
        self.state_var = None
        norm_path = self.agent_outputs_dir / f"{model}-final_state_norm.npz"
        if self.use_state_norm and norm_path.exists():
            arr = np.load(norm_path)
            self.state_mean = arr["mean"]
            self.state_var = arr["var"]

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if self.state_mean is None or self.state_var is None:
            return state
        return ((state - self.state_mean) / np.sqrt(self.state_var + 1e-8)).astype(np.float32)
    
    def build_state_for_request(
        self,
        request: Request,
        profile: Dict[str, Any],
    ) -> np.ndarray:
        embedding = self.embedder.embed(request.prompt)
        embedding = np.asarray(embedding, dtype=np.float32)
        values = [
            float(len(request.prompt_token_ids)),
            float(request.sampling_params.ttft_slo),
            float(request.sampling_params.tpot_slo),
            float(profile["h_queue_len"]),
            float(profile["l_queue_len"]),
            float(profile["migration_len"]),
            float(profile["h_kv_cache_util"]),
            float(profile["l_kv_cache_util"]),
            float(profile["h_inflight"]),
            float(profile["l_inflight"]),
        ]
        scalars = np.asarray(values, dtype=np.float32)
        state = np.concatenate([embedding.astype(np.float32), scalars], axis=0)
        return state.astype(np.float32)

    @torch.no_grad()
    def predict(self, request: Request, profile: Dict[str, Any], deterministic: bool = True) -> Policy:
        state = self.build_state_for_request(request, profile)
        state = self.normalize_state(state)
        state_tensor = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        logits = self.actor(state_tensor)

        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())

        return Policy(self.action_names[action].lower())