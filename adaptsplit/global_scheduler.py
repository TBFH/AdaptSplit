from typing import List, Callable, Tuple, Optional, Dict, Any
import random
import asyncio

from adaptsplit.single_stage_engine import (
    StepOutput,
    PrefillStageLLMEngine,
    DecodingStageLLMEngine
)
from adaptsplit.config import PrefillStageSchedConfig, ParallelConfig
from adaptsplit.logger import init_logger
from adaptsplit.request import Request, BatchedRequests, MigratingRequest
from adaptsplit.utils import Policy

from adaptsplit.agent.PPO import OnlineSchedulerPolicy

logger = init_logger(__name__)

random.seed(1)

class GlobalScheduler:
    """
    GlobalScheduler: A dispatcher of requests maintained within LLMEngine.

    It receives all requests from LLMEngine.generate() and should decide whether to 
    be processed by PrefillStageLLMEngine or DecodingStageLLMEngine for each request.
    An agent is built inside to perform this sheduling policy, which is trained 
    using the LLM-empowered deep reinforcement learning method.

    We designed three different inference route which also builds up the action space:
        - HPHD: Prefilling and Decoding in PrefillStageLLMEngine;
        - HPLD: Prefilling in PrefillStageLLMEngine, Decoding in DecodingStageLLMEngine;
        - LPLD: Prefilling and Decoding in DecodingStageLLMEngine;
    
    PrefillStageLLMEngine is deployed upon High-Performance edge nodes, and 
    DecodingStageLLMEngine is deployed upon Low-Performance edge nodes
    """

    def __init__(
        self,
        model: str,
        prefill_engines: List[PrefillStageLLMEngine],
        decoding_engine: DecodingStageLLMEngine,
        global_schedule_policy: str = 'default',
        profile_func_callback: Optional[Callable[[], Dict[str, Any]]] = None
    ):
        self.agent = OnlineSchedulerPolicy(
            model=model,
            agent_outputs_dir="/home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/outputs",
            embedder_dir="/home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/sentence_embedding/generated",
        )
        self.request_queue = asyncio.Queue()
        # engines
        self.prefill_engines = prefill_engines
        self.decoding_engine = decoding_engine
        self.profile_func_callback = profile_func_callback

        self.prefill_index = -1
        self.global_schedule_policy = global_schedule_policy
    
    def add_request(self, request: Request) -> None:
        # Add a request to the scheduler.
        self.request_queue.put_nowait(request)

    def _send_request(self, request: Request) -> None:
        # Todo:
        # Let agent decide policy action
        # execute action to send request

        if not request.policy:
            if self.global_schedule_policy == 'random':
                policy = random.choice(list(Policy))
            elif self.global_schedule_policy == 'hphd':
                policy = Policy.HPHD
            elif self.global_schedule_policy == 'hpld':
                policy = Policy.HPLD
            elif self.global_schedule_policy == 'lpld':
                policy = Policy.LPLD
            elif self.global_schedule_policy == 'default':
                assert (
                    self.profile_func_callback != None
                ), "[GlobalScheduler] No profile_func_callback found"
                assert (
                    request.sampling_params.ttft_slo and request.sampling_params.tpot_slo
                ), "[GlobalScheduler] ttft and tpot slo for request is needed when global_schedule_policy is 'default'"
                profiles = self.profile_func_callback()
                policy = self.agent.predict(request, profiles)
            else:
                raise ValueError(f"global schedule policy {self.global_schedule_policy} is not supported")
            request.policy = policy

        # print(f"[Global Scheduler] request_id: {request.request_id} policy: {request.policy}")
        if request.policy == Policy.HPHD or request.policy == Policy.HPLD:
            self.prefill_index = (self.prefill_index + 1) % len(self.prefill_engines)
            self.prefill_engines[self.prefill_index].add_request(request)
        elif request.policy == Policy.LPLD:
            self.decoding_engine.add_new_request(request)
        else:
            raise ValueError(f"Policy {request.policy} is not supported.")

    async def start_event_loop(self):
        while True:
            req = await self.request_queue.get()
            self._send_request(req)
            self.request_queue.task_done()