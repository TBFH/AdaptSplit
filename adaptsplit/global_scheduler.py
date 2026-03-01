from typing import List, Callable, Tuple
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

logger = init_logger(__name__)


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
        prefill_engines: List[PrefillStageLLMEngine],
        decoding_engine: DecodingStageLLMEngine,
    ):
        self.agent = None
        self.request_queue = asyncio.Queue()
        # engines
        self.prefill_engines = prefill_engines
        self.decoding_engine = decoding_engine

        self.prefill_index = -1
    
    def add_request(self, request: Request) -> None:
        # Add a request to the scheduler.
        self.request_queue.put_nowait(request)

    def _send_request(self, request: Request) -> None:
        # Todo:
        # Let agent decide policy action
        # execute action to send request

        # Randomly dispatching
        policy = random.choice(list(Policy))
        request.policy = policy
        print(f"[Global Scheduler] request_id: {request.request_id} policy: {policy}")
        if policy == Policy.HPHD or policy == Policy.HPLD:
            self.prefill_index = (self.prefill_index + 1) % len(self.prefill_engines)
            self.prefill_engines[self.prefill_index].add_request(request)
        elif policy == Policy.LPLD:
            self.decoding_engine.add_new_request(request)
        else:
            raise ValueError(f"Policy {policy} is not supported.")

    async def start_event_loop(self):
        while True:
            req = await self.request_queue.get()
            self._send_request(req)
            self.request_queue.task_done()