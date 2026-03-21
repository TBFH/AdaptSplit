from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, AsyncGenerator
import aiohttp
import asyncio
import numpy as np
import requests
from itertools import cycle, islice

from adaptsplit.agent.env_utils import *


class _Discrete:
    def __init__(self, n: int) -> None:
        self.n = n

class _Box:
    def __init__(self, low: float, high: float, shape: Tuple[int, ...], dtype: Any) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

class spaces:  # type: ignore
    Discrete = _Discrete
    Box = _Box


@dataclass
class RequestItem:
    request_id: str
    prompt: str
    input_length: int
    output_length: int
    ttft_slo_ms: float
    tpot_slo_ms: float
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequestItem":
        prompt = str(data.get("prompt"))
        if not prompt:
            raise ValueError("[Env] Each request item must include 'prompt'")
        request_id = str(data.get("request_id"))
        input_length = int(data.get("input_length"))
        output_length = int(data.get("output_length"))
        ttft_slo_ms = float(data.get("ttft_slo_ms"))
        tpot_slo_ms = float(data.get("tpot_slo_ms"))
        embedding_value = data.get("embedding", None)
        embedding = None
        if embedding_value is not None:
            embedding = np.asarray(embedding_value, dtype=np.float32)

        return cls(
            request_id=request_id,
            prompt=prompt,
            input_length=input_length,
            output_length=output_length,
            ttft_slo_ms=ttft_slo_ms,
            tpot_slo_ms=tpot_slo_ms,
            embedding=embedding,
        )


@dataclass
class EngineEndpoints:
    generate_url: str
    profile_url: str
    summary_url: str
    reset_url: str


@dataclass
class SchedulerEnvConfig:
    endpoints: EngineEndpoints
    model: str
    dataset_path: str
    num_episode_requests: int = 50
    reward_w1: float = 1.0
    reward_w2: float = 1.0
    seed: int = 42
    request_timeout_s: float = 3600.0
    max_workers: int = 256
    dispatch_settle_seconds: float = 0.0
    state_scalar_names: Optional[List[str]] = None
    action_names: List[str] = field(default_factory=list)
    request_rates: List[float] = field(default_factory=list)


class EngineHTTPClient:
    def __init__(self, endpoints: EngineEndpoints, timeout_s: float = 3600.0) -> None:
        self.endpoints = endpoints
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self._lock = threading.Lock()

    def _post_json(
        self, 
        url: str, 
        payload: Optional[Mapping[str, Any]] = None, 
        timeout_s: Optional[float] = None
    ) -> Dict[str, Any]:
        timeout = timeout_s or self.timeout_s
        with self._lock:
            response = self.session.post(url, json=payload or {}, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def generate(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        # return self._post_json(self.endpoints.generate_url, payload)
        return asyncio.run(
            send_request(
                self.endpoints.generate_url,
                payload["prompt"],
                payload["input_length"],
                payload["output_length"],
                payload["strategy"],
            )
        )

    def profile(self) -> Dict[str, Any]:
        return self._post_json(self.endpoints.profile_url, {})

    def summary(self, start: float, end: float) -> Dict[str, Any]:
        return self._post_json(self.endpoints.summary_url, {"start": start, "end": end})

    def reset(self) -> Optional[Dict[str, Any]]:
        return self._post_json(self.endpoints.reset_url, {})

    def close(self) -> None:
        self.session.close()


async def send_request(
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    policy: str
) -> RequestResult:
    headers = {"User-Agent": "Benchmark Client"}
    pload = {
        "prompt": prompt,
        "n": 1,
        "max_tokens": output_len,
        "ignore_eos": True,
        "stream": False,
        "policy": policy.lower()
    }

    # The maximum length of the input is 2048, limited by the embedding
    # table size.
    assert prompt_len+output_len < 2048
    
    request_start_time = time.time()
    request_output = None

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            try:
                output = json.loads(output)
            except:
                print("Failed to parse the response:")
                print(output)
                continue
            # Re-send the request if it failed.
            if "error" not in output:
                request_output = output
                break
            else:
                print(f"Failed to process the request: {output['error']}")
                print(f"Resending the request: {pload}")

    request_end_time = time.time()
    
    return RequestResult(
        prompt_len,
        output_len,
        request_start_time,
        request_end_time,
        token_timestamps=request_output["timestamps"],
        lifetime_events=request_output.get("lifetime_events", None)
    )


class AdaptsplitSchedulingEnv():
    """
    Custom episodic environment for AdaptSplit scheduling.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: SchedulerEnvConfig,
    ) -> None:
        self.config = config
        self.rng = random.Random(config.seed)   # random number generator
        self.np_rng = np.random.default_rng(config.seed)
        self.client = EngineHTTPClient(config.endpoints, timeout_s=config.request_timeout_s)
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)

        self.dataset: List[RequestItem] = self._load_dataset(config.dataset_path)
        if not self.dataset:
            raise ValueError("[Env] The dataset is empty. Please provide at least one request sample.")
        
        if not self.config.request_rates:
            raise ValueError("[Env] request_rates is empty. Please provide at least one request rate.") 
        self.dataset_cycle = cycle(self.dataset)

        self._embedding_dim = int(self._get_embedding(self.dataset[0]).shape[0])
        self._scalar_names = self._build_scalar_names()
        self.action_names = list(config.action_names)
        self.state_dim = self._embedding_dim + len(self._scalar_names)
        self.action_dim = len(self.action_names)
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        self._episode_requests: List[RequestItem] = []
        self._current_idx = 0
        self._current_request: Optional[RequestItem] = None
        self._pending_futures: List[Tuple[RequestItem, str, Future[RequestResult]]] = []
        self._completed_results: List[Dict[str, Any]] = []
        self._last_profile: Dict[str, Any] = {}
        self._step_count = 0

        self._epoch_count = 0
        self._intervals = []
        self._start_time = None

    # ------------------------- Dataset and feature utilities -------------------------
    def _load_dataset(self, dataset_path: str) -> List[RequestItem]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"[Env] Dataset file not found: {dataset_path}")

        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("[Env] The dataset file must contain a list of requests or a JSONL stream.")

        return [RequestItem.from_dict(row) for row in rows]

    def _get_embedding(self, item: RequestItem) -> np.ndarray:
        if item.embedding is None:
            raise ValueError(f"[Env] sentence embedding of request {item.request_id} could not be None")
        return item.embedding

    def _build_scalar_names(self) -> List[str]:
        if self.config.state_scalar_names is not None:
            return list(self.config.state_scalar_names)
        else:
            raise ValueError("[Env] state_scalar_names are needed in env_config")

    def _profile_to_scalars(self, item: RequestItem, profile: Mapping[str, Any]) -> np.ndarray:
        values = [
            float(item.input_length),
            float(item.ttft_slo_ms),
            float(item.tpot_slo_ms),
            float(profile["h_queue_len"]),
            float(profile["l_queue_len"]),
            float(profile["migration_len"]),
            float(profile["h_kv_cache_util"]),
            float(profile["l_kv_cache_util"]),
            float(profile["h_inflight"]),
            float(profile["l_inflight"]),
        ]
        return np.asarray(values, dtype=np.float32)

    def _compose_state(self, item: RequestItem, profile: Mapping[str, Any]) -> np.ndarray:
        embedding = self._get_embedding(item)
        scalars = self._profile_to_scalars(item, profile)
        state = np.concatenate([embedding.astype(np.float32), scalars], axis=0)
        assert state.shape[0] == self.state_dim, (state.shape[0], self.state_dim)
        return state.astype(np.float32)

    def _sample_episode_requests(self) -> List[RequestItem]:
        if len(self.dataset) >= self.config.num_episode_requests:
            # return self.rng.sample(self.dataset, self.config.num_episode_requests)
            return list(islice(self.dataset_cycle, self.config.num_episode_requests))
        return [self.rng.choice(self.dataset) for _ in range(self.config.num_episode_requests)]

    # ------------------------------ Engine interaction ------------------------------
    def _dispatch_generate(self, item: RequestItem, strategy: str) -> Future[Dict[str, Any]]:
        payload = {
            "prompt": item.prompt,
            "input_length": item.input_length,
            "output_length": item.output_length,
            "strategy": strategy,
        }
        return self.executor.submit(self.client.generate, payload)

    def _wait_all(self) -> None:
        completed: List[Dict[str, Any]] = []
        for item, strategy, future in self._pending_futures:
            result = future.result(timeout=self.config.request_timeout_s)
            completed.append({
                "request_id": item.request_id,
                "strategy": strategy,
                "input_length": item.input_length,
                "output_length": result.output_len,
                "ttft_slo_ms": item.ttft_slo_ms,
                "tpot_slo_ms": item.tpot_slo_ms,
                "ttft": result.ttft,
                "tpot": result.tpot,
                "start_time": result.start_time,
                "end_time": result.end_time,
            })
        self._completed_results = completed
        self._pending_futures.clear()

    def _compute_violation_rate(self) -> float:
        violations = 0
        for result in self._completed_results:
            ttft = result["ttft"]
            tpot = result["tpot"]
            ttft_slo = result["ttft_slo_ms"]
            tpot_slo = result["tpot_slo_ms"]
            if ttft >= ttft_slo or tpot >= tpot_slo:
                violations += 1
        return violations / len(self._completed_results)

    def _compute_episode_reward(self) -> Tuple[float, Dict[str, float]]:
        start_times, end_times = [], []
        sum_tokens = 0
        # req_throughputs = []
        for req in self._completed_results:
            start_times.append(req["start_time"])
            end_times.append(req["end_time"])
            sum_tokens += req["input_length"] + req["output_length"]
            # req_throughputs.append(
            #     (req["input_length"] + req["output_length"]) / (req["end_time"] - req["start_time"])
            # )
        latency = max(end_times) - min(start_times)
        powers_summary = self.client.summary(min(start_times), max(end_times))
        powers_summary.pop('pc-3090-1', None)

        # weighted_powers
        weighted_powers = []
        for node, power in powers_summary.items():
            if 'pc' in node:
                weighted_powers.append(power * 2.5)
            else:
                weighted_powers.append(power * 1.0)

        # throughput = sum(req_throughputs) / len(req_throughputs)
        throughput = sum_tokens / latency
        # total_avg_power = sum(powers_summary.values())
        total_avg_power = sum(weighted_powers)
        violation_rate = self._compute_violation_rate()
        energy_efficiency = throughput / total_avg_power

        reward = self.config.reward_w1 * energy_efficiency - self.config.reward_w2 * violation_rate
        metrics = {
            "sum_tokens": sum_tokens,
            "latency": latency,
            "throughput_token/s": throughput,
            "total_avg_power_W": total_avg_power,
            "energy_efficiency_token/J": energy_efficiency,
            "violation_rate": violation_rate,
            "episodic_reward": reward,
        }
        return reward, metrics

    # --------------------------------- Gym interface ---------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        if self._pending_futures:
            self._wait_all()
        self._completed_results.clear()
        self._pending_futures.clear()
        self._step_count = 0
        self._current_idx = 0
        self._episode_requests = self._sample_episode_requests()
        self._epoch_count += 1

        num_rates = len(self.config.request_rates)
        self._intervals = self._get_intervals(
            num_requests=self.config.num_episode_requests,
            process_name="possion",
            request_rate=self.config.request_rates[self._epoch_count % num_rates]
        )
        print(f"[Env] Started epoch {self._epoch_count} with request_rate {self.config.request_rates[self._epoch_count % num_rates]}")
        self._start_time = time.time()

        self._current_request = self._episode_requests[0]
        self.client.reset()
        self._last_profile = self.client.profile()
        state = self._compose_state(self._current_request, self._last_profile)
        info = {
            "request_id": self._current_request.request_id,
            "action_names": list(self.action_names),
        }
        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._current_request is None:
            raise RuntimeError("[Env] Call reset() before step().")
        if action < 0 or action >= len(self.action_names):
            raise ValueError(f"[Env] Invalid action={action}; expected one of [0, {len(self.action_names) - 1}].")

        while time.time() - self._start_time < self._intervals[self._step_count]:
            time.sleep(self.config.dispatch_settle_seconds)
        self._start_time = time.time()

        strategy = self.action_names[int(action)]
        current_request = self._current_request
        future = self._dispatch_generate(current_request, strategy)
        self._pending_futures.append((current_request, strategy, future))
        self._step_count += 1

        # if self.config.dispatch_settle_seconds > 0:
        #     time.sleep(self.config.dispatch_settle_seconds)

        done = self._step_count >= self.config.num_episode_requests
        truncated = False

        if done:
            print(f"[Env] Done epoch {self._epoch_count}")
            self._wait_all()
            reward, metrics = self._compute_episode_reward()
            terminal_state = np.zeros((self.state_dim,), dtype=np.float32)
            info = {
                **metrics,
                "request_rate": self.config.request_rates[self._epoch_count % len(self.config.request_rates)],
                "completed_requests": self._completed_results,
                # "last_strategy": strategy,
            }
            self._current_request = None
            return terminal_state, float(reward), True, truncated, info

        self._current_idx += 1
        self._current_request = self._episode_requests[self._current_idx]
        self._last_profile = self.client.profile()
        next_state = self._compose_state(
            self._current_request,
            self._last_profile
        )
        info = {
            "request_id": current_request.request_id,
            "selected_strategy": strategy,
            "next_request_id": self._current_request.request_id,
            "request_rate": self.config.request_rates[self._epoch_count % len(self.config.request_rates)]
        }
        return next_state, 0.0, False, truncated, info

    def close(self) -> None:
        try:
            if self._pending_futures:
                self._wait_all()
        finally:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.client.close()

    # ------------------------------- Async Request Sender ----------------------------
    def _get_intervals(
        self,
        num_requests: int,
        process_name: str = "possion",
        request_rate: float = 1.0,
        cv: float = 1.0,
    ) -> List[float]:
        interval_lens = num_requests
        assert request_rate not in [float("inf"), 0.0], "request_rate could not be float('inf') or 0.0"
        if process_name == "uniform":
            intervals = [1.0 / request_rate for _ in range(interval_lens)]
        elif process_name == "gamma":
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens).tolist()
        elif process_name == "possion":
            cv = 1
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens).tolist()
        else:
            raise ValueError(
                f"[Env] Unsupported prosess name: {process_name}, we currently support uniform, gamma and possion."
            )
        return intervals

    # ------------------------------- Prompt generation -------------------------------
    def get_prompt_context(self) -> Dict[str, Any]:
        example_request = self.dataset[0]
        example_state = self._compose_state(example_request, {k: 0.0 for k in self._scalar_names})
        return {
            "example_state": example_state.tolist(),
            "example_action": 1,
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
        }
