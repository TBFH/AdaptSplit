from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple
import time
import copy
import numpy as np
import torch

from adaptsplit.agent.env_wrapper import AdaptsplitSchedulingEnv, EngineEndpoints, SchedulerEnvConfig
from adaptsplit.agent.PPO import PPOAgent, PPOConfig, RolloutBuffer, TrajectoryReplayBuffer
from adaptsplit.agent.reward_model.RD import RDConfig, RDRewardDecomposer
from adaptsplit.agent.reward_model.LLMrd import LLMRDConfig, LLMRDRewardDecomposer


class RunningNorm:
    def __init__(self, shape: int, eps: float = 1e-8) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m2 / total_count
        self.mean, self.var, self.count = new_mean, new_var, total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_env_config(path: str) -> SchedulerEnvConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    endpoints = EngineEndpoints(**payload["endpoints"])
    cfg = SchedulerEnvConfig(
        endpoints=endpoints,
        model=payload["model"],
        dataset_path=payload["dataset_path"],
        num_episode_requests=int(payload.get("num_episode_requests", 50)),
        reward_w1=float(payload.get("reward_w1", 0.5)),
        reward_w2=float(payload.get("reward_w2", 0.5)),
        seed=int(payload.get("seed", 1)),
        request_timeout_s=float(payload.get("request_timeout_s", 3600)),
        max_workers=int(payload.get("max_workers", 256)),
        dispatch_settle_seconds=float(payload.get("dispatch_settle_seconds", 0.0)),
        state_scalar_names=payload.get("state_scalar_names"),
        action_names=payload.get("action_names"),
        request_rates=payload.get("request_rates", []),
    )
    return cfg


def build_reward_model(args: argparse.Namespace, env: AdaptsplitSchedulingEnv):
    if args.rd_method == "none":
        return None
    if args.rd_method == "RD":
        return RDRewardDecomposer(
            RDConfig(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=args.hidden_dim,
                reward_lr=args.reward_lr,
                device=args.device,
            )
        )
    if args.rd_method == "LaRe_RD":
        return LLMRDRewardDecomposer(
            LLMRDConfig(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                hidden_dim=args.hidden_dim,
                reward_lr=args.reward_lr,
                device=args.device,
                llm_model=args.llm_model,
                llm_temperature=args.llm_temperature,
                llm_candidates=args.llm_candidates,
                llm_response_dir=args.llm_response_dir,
                response_id=0,
            ),
            prompt_context=env.get_prompt_context(),
        )
    raise ValueError(f"Unsupported rd_method={args.rd_method}")


def save_deploy_artifacts(
    output_dir: Path,
    env: AdaptsplitSchedulingEnv,
    args: argparse.Namespace,
    state_norm: RunningNorm | None,
) -> None:
    meta = {
        "model": env.config.model,
        "state_dim": int(env.state_dim),
        "action_dim": int(env.action_dim),
        "hidden_dim": int(args.hidden_dim),
        "action_names": env.action_names,
        "use_state_norm": bool(args.state_norm),
        "embedding_dim": int(env._embedding_dim),
        "scalar_feature_names": list(env._scalar_names),
    }
    with open(output_dir / f"{env.config.model}-deploy_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if state_norm is not None:
        np.savez(
            output_dir / f"{env.config.model}-final_state_norm.npz",
            mean=state_norm.mean,
            var=state_norm.var,
            count=state_norm.count,
        )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    env_cfg = load_env_config(args.env_config)
    env_cfg.seed = args.seed

    env = AdaptsplitSchedulingEnv(config=env_cfg)

    reward_model = build_reward_model(args, env)
    agent = PPOAgent(
        PPOConfig(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            device=args.device,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_epsilon=args.clip_epsilon,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            hidden_dim=args.hidden_dim,
            reward_warmup_episodes=args.reward_warmup_episodes,
            reward_updates_per_episode=args.reward_updates_per_episode,
            reward_batch_size=args.reward_batch_size,
        ),
        reward_model=reward_model,
    )

    trajectory_buffer = TrajectoryReplayBuffer(capacity=args.trajectory_buffer_size)
    state_norm = RunningNorm(shape=env.state_dim) if args.state_norm else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    try:
        for episode_idx in range(1, args.total_episodes + 1):
            rollout = RolloutBuffer()
            state, info = env.reset(seed=args.seed + episode_idx)
            if state_norm is not None:
                state_norm.update(state[None, :])
                state = state_norm.normalize(state)

            episode_states = []
            episode_actions = []
            episode_next_states = []
            episode_return = 0.0
            done = False

            while not done:
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, truncated, step_info = env.step(action)
                done = bool(terminated or truncated)

                policy_next_state = next_state
                if state_norm is not None and not done:
                    state_norm.update(next_state[None, :])
                    policy_next_state = state_norm.normalize(next_state)

                rollout.add(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    next_state=policy_next_state,
                    done=done,
                    info=step_info,
                )

                episode_states.append(state)
                episode_actions.append(action)
                episode_next_states.append(policy_next_state)
                episode_return += reward
                state = policy_next_state

            trajectory_buffer.add_episode(
                states=np.asarray(episode_states, dtype=np.float32),
                actions=np.asarray(episode_actions, dtype=np.int64),
                next_states=np.asarray(episode_next_states, dtype=np.float32),
                episodic_return=float(episode_return),
            )

            reward_metrics = agent.train_reward_model(trajectory_buffer)
            policy_metrics = agent.train_policy(rollout)
            episode_metrics = {
                "episode": episode_idx,
                "episodic_return": float(episode_return),
            }
            episode_metrics.update(reward_metrics)
            episode_metrics.update(policy_metrics)
            episode_metrics.update(rollout.infos[-1] if rollout.infos else {})
            history.append(episode_metrics)

            if episode_idx % args.log_interval == 0:
                log_metrics = copy.deepcopy(episode_metrics)
                log_metrics.pop("completed_requests", None)
                print(json.dumps(log_metrics, ensure_ascii=False, indent=2))

            if episode_idx % args.save_interval == 0:
                agent.save(str(output_dir / f"{env_cfg.model}-checkpoint_ep{episode_idx}"))
                with open(output_dir / f"{env_cfg.model}-history.json", "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
            
            time.sleep(10)

        agent.save(str(output_dir / f"{env_cfg.model}-final"))
        save_deploy_artifacts(output_dir, env, args, state_norm)
        with open(output_dir / f"{env_cfg.model}-history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO + LaRe-RD for edge LLM scheduling")
    parser.add_argument("--env-config", type=str, required=True, help="Path to the JSON environment config file.")  # !
    parser.add_argument("--output-dir", type=str, default="./outputs")  # !
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-episodes", type=int, default=100)   # !
    parser.add_argument("--log-interval", type=int, default=2)      # !
    parser.add_argument("--save-interval", type=int, default=10)    # !

    parser.add_argument("--rd-method", type=str, default="LaRe_RD", choices=["none", "RD", "LaRe_RD"])
    parser.add_argument("--reward-lr", type=float, default=1e-4)
    parser.add_argument("--llm-model", type=str, default="gpt-4.1")     # !
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--llm-candidates", type=int, default=5)
    parser.add_argument("--llm-response-dir", type=str, default="./llm_reward_cache")   # !

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=20)       # !
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reward-warmup-episodes", type=int, default=5)        # !
    parser.add_argument("--reward-updates-per-episode", type=int, default=20)   # !
    parser.add_argument("--reward-batch-size", type=int, default=16)
    parser.add_argument("--trajectory-buffer-size", type=int, default=5000)
    parser.add_argument("--state-norm", action="store_true")    # !
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())


'''
python -m adaptsplit.agent.ppo_main \
    --env-config /home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/env_config.json \
    --output-dir /home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/outputs \
    --llm-model glm-5 \
    --llm-response-dir /home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/reward_model/generated \
    --state-norm
'''