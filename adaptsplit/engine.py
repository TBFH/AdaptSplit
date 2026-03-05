import time
import copy
from typing import List, Optional, Tuple, Dict, AsyncGenerator
import asyncio
import math
import argparse

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from adaptsplit.config import (
    ModelConfig, 
    DisaggParallelConfig, 
    ParallelConfig, 
    CacheConfig, 
    PrefillStageSchedConfig,
    DecodingStageSchedConfig
)
from adaptsplit.logger import init_logger
from adaptsplit.request import (
    SamplingParams,
    Request,
    create_request,
)
from adaptsplit.tokenizer import get_tokenizer
from adaptsplit.utils import Counter, Policy
from adaptsplit.single_stage_engine import (
    StepOutput,
    PrefillStageLLMEngine,
    DecodingStageLLMEngine
)
from adaptsplit.lifetime import LifetimeEvent, LifetimeEventType
from adaptsplit.global_scheduler import GlobalScheduler

logger = init_logger(__name__)

# 配置相关环境变量，防止通信问题导致的程序卡死
import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] = 'eno4'


@ray.remote(num_gpus=1)
def resource_inspect():
    # GPU Overall Inspect
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    device_name = device.name()
    context = device.make_context()
    total_memory = device.total_memory() / (1024 ** 2)
    free_memory = cuda.mem_get_info()[0] / (1024 ** 2)
    used_memory = total_memory - free_memory
    context.pop()
    return {
        "GPU_Name": device_name,
        "Total_VRAM": total_memory,
        "Used_VRAM": used_memory,
        "Free_VRAM": free_memory,
    }


class LLMEngine:
    """
    LLMEngine: An LLMEngine launches the model executor workers and maintains runtime information.

    ## Overview

    This class, LLMEngine, receives requests from upper wrapper class and provides
    interface LLMEngine.generate() that yields the generated tokens for each request.

    It supports the feature of "disaggregate", which basically means to run 
    the prefill stage and the decoding stage on different GPUs to avoid interference.

    ## Implementation

    First let's inspect the automaton of one request:

            After
            prefill
            stage        |-------------|
    Waiting --------> Decoding <-------| After one decoding stage
                         |
                         |
                         V
                      Finished

    This class is implemented based on queues and event loops. There are three
    queues, two for scheduling and one for communication between event loops:
      - The waiting queue, maintained inside the PrefillStageScheduler, which
        contains all the requests that are waiting for processing.
      - The decoding queue, maintained inside the DecodingStageScheduler, which
        contains all the requests that need further decoding.
      - The "bridge" queue, which contains all the requests that have just finished
        the prefill stage but have not been accepted by the decoding stage.
        (Producer: prefill stage event loop, Consumer: decoding stage event loop)
      
    Two event loops are executed concurrently and endlessly:
      - Prefill stage event loop. This event loop fetches requests from the waiting
        queue, forwards them to the prefill stage, and then puts them into the
        "bridge" queue.
      - Decoding stage event loop. This event loop accepts requests from the
        "bridge" queue (put them into the decoding queue), and then fetches requests
        from the decoding queue, forwards them to the decoding stage, and then
        informs the caller of the generated tokens.

    Note: Users may not use LLMEngine directly, but use more user-friendly wrapper classes
    OfflineLLM and AsyncLLM instead.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        disagg_parallel_config: DisaggParallelConfig,
        cache_config: CacheConfig,
        prefill_sched_config: PrefillStageSchedConfig,
        decoding_sched_config: DecodingStageSchedConfig,
        prefill_devices: Optional[List[str]] = None,
        decoding_devices: Optional[List[str]] = None,
        global_schedule_policy: str = 'default'
    ):
        # pipeline_distribution definition
        if len(disagg_parallel_config.prefill.pipeline_distribution) == 0:
            pp_size = disagg_parallel_config.prefill.pipeline_parallel_size
            disagg_parallel_config.prefill.pipeline_distribution = [model_config.get_num_layers() // pp_size] * pp_size
        if len(disagg_parallel_config.decoding.pipeline_distribution) == 0:
            pp_size = disagg_parallel_config.decoding.pipeline_parallel_size
            disagg_parallel_config.decoding.pipeline_distribution = [model_config.get_num_layers() // pp_size] * pp_size

        self.model_config = model_config
        self.disagg_parallel_config = disagg_parallel_config
        self.cache_config = cache_config
        self.prefill_sched_config = prefill_sched_config
        self.decoding_sched_config = decoding_sched_config

        self.request_counter = Counter()
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
        )
        
        self.bridge_queue = asyncio.Queue()

        if not ray.is_initialized():
            ray.init(
                # include_dashboard=False
                address="ray://219.222.20.79:30807"
            )
        
        if prefill_devices != None:
            assert (
                len(prefill_devices) == disagg_parallel_config.prefill.data_parallel_size
            ), "num of available prefill devices does not fit data_parallel_size"
        if decoding_devices != None:
            assert (
                len(decoding_devices) == disagg_parallel_config.decoding.pipeline_parallel_size
            ), "num of available decoding devices does not fits pipeline_parallel_size"

        self.prefill_devices = prefill_devices
        self.decoding_devices = decoding_devices

        self.device_map = {}
        self._device_nodeid_mapping()

        self.node_resources = {}
        self._init_inspect()

        prefill_deployment, decoding_deployment = self._init_deployments()
        print(f"\033[1;34m prefill_deployment: {prefill_deployment} \t\t decoding_deployment: {decoding_deployment} \033[0m")
        
        logger.info("Initializing prefill stage LLM engine")
        self.prefill_engines: List[PrefillStageLLMEngine] = []
        for i in range(disagg_parallel_config.prefill.data_parallel_size):
            parallel_config = copy.deepcopy(disagg_parallel_config.prefill)
            parallel_config.data_parallel_rank = i
            self.prefill_engines.append(
                PrefillStageLLMEngine(
                    self.bridge_queue,
                    model_config,
                    parallel_config,
                    cache_config,
                    prefill_sched_config,
                    [prefill_deployment[i]],
                    self._on_new_step_output_callback,
                    self._on_new_lifetime_event_callback,
                    self.device_map,
                    self._on_request_fail_callback
                )
            )
        
        logger.info("Initializing decoding stage LLM engine")
        self.decoding_engine = DecodingStageLLMEngine(
            self.bridge_queue,
            model_config,
            disagg_parallel_config.decoding,
            cache_config,
            decoding_sched_config,
            decoding_deployment,
            [engine.clear_migrated_blocks_callback for engine in self.prefill_engines],
            self._on_new_step_output_callback,
            self._on_new_lifetime_event_callback,
            [engine.workers for engine in self.prefill_engines],
            self.device_map,
            [engine.add_request for engine in self.prefill_engines],
            self._on_request_fail_callback
        )
        
        # request_id -> list of StepOutput
        # Created when calling self.generate()
        # Cleared when the request is finished
        self.request_outputs: Dict[int, asyncio.Queue[StepOutput]] = {}
        
        # request_id -> list of LifetimeEvent
        # Created when calling self.generate()
        # Cleared by the caller of self.generate() (i.e. the engine does not clear that)
        # TODO: clear this automatically to avoid memory leak
        self.request_lifetime_events: Dict[int, List[LifetimeEvent]] = {}

        self.failed_requests: List[int] = []

        # Initialize global scheduler
        self.global_scheduler = GlobalScheduler(
            prefill_engines=self.prefill_engines,
            decoding_engine=self.decoding_engine,
            global_schedule_policy=global_schedule_policy
        )
        
        self.engine_initialized = False

    def _init_inspect(self):
        nodes = ray.nodes()
        futures = []
        for i, node in enumerate(nodes):
            if node['Alive']:
                node_id = node['NodeID']
                future = resource_inspect.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    )
                ).remote()
                futures.append((node_id, future))
        # Save & Print out Inspects Data
        for idx, (node_id, future) in enumerate(futures):
            result = ray.get(future)
            self.node_resources[node_id] = result
            # outlog = ''
            # outlog += f'[Worker{idx}] NodeID: {node_id}\n'
            # outlog += f'[Worker{idx}] Device: {self.device_map[node_id]}\n'
            # outlog += f'[Worker{idx}] Used/Total VRAM: {result["Used_VRAM"]/1024:.1f}/{result["Total_VRAM"]/1024:.1f} GB ({(result["Used_VRAM"]/result["Total_VRAM"])*100:.1f}%)\n'
            # outlog += f'[Worker{idx}] Free VRAM: {result["Free_VRAM"]/1024:.1f} GB\n'
            # print(outlog)

    def _device_nodeid_mapping(self):
        nodes = ray.nodes()
        for node in nodes:
            if node['Alive']:
                node_id = node['NodeID']
                resources = list(node['Resources'].keys())
                for el in resources:
                    if el.startswith('device'):
                        device = el.split(':',1)[1]
                self.device_map[node_id] = device

    def _init_deployments(self):
        selected = []

        def get_deployment(deployment:List, available_devices:List, parallel_size:int):
            if available_devices:
                device_map_rvt = {v:k for k,v in self.device_map.items()}
                for node in available_devices:
                    if node not in device_map_rvt:
                        raise RuntimeError(f"Node [{node}] not found in cluster")
                    if node in selected:
                        raise RuntimeError(f"Node [{node}] is already used")
                    deployment.append(device_map_rvt[node])
                    selected.extend([node, device_map_rvt[node]])
            else:
                cnt = 0
                for node_id, res in self.node_resources.items():
                    # if res['Free_VRAM'] > 4096 and 'jetson' in self.device_map[node_id] and node_id not in selected and cnt < parallel_size:
                    if res['Free_VRAM'] > 4096 and node_id not in selected and cnt < parallel_size:
                        deployment.append(node_id)
                        selected.extend([self.device_map[node_id], node_id])
                        cnt += 1
            return deployment

        prefill_deployment = []
        decoding_deployment = []
        # prefill_size = self.disagg_parallel_config.prefill.pipeline_parallel_size * self.disagg_parallel_config.prefill.tensor_parallel_size
        prefill_size = self.disagg_parallel_config.prefill.data_parallel_size
        decoding_size = self.disagg_parallel_config.decoding.pipeline_parallel_size * self.disagg_parallel_config.decoding.tensor_parallel_size
        return get_deployment(prefill_deployment, self.prefill_devices, prefill_size), \
                get_deployment(decoding_deployment, self.decoding_devices, decoding_size)
    
    def _on_new_step_output_callback(self, request_id: int, step_output: StepOutput):
        """
        Called by self.prefill_engine or self.decoding_engine when a new output token
        is generated
        """
        self.request_outputs[request_id].put_nowait(step_output)
        
    def _on_new_lifetime_event_callback(self, request_id: int, event: LifetimeEvent, dont_add_if_dup: bool = False):
        """
        Called by self.prefill_engine or self.decoding_engine when a new lifetime event
        is generated
        """
        def dup_check(event: LifetimeEvent) -> bool:
            for e in self.request_lifetime_events[request_id]:
                if e.event_type == event.event_type:
                    return True
            return False
        
        # if dont_add_if_dup == True and self.request_lifetime_events[request_id][-1].event_type == event.event_type, don't add it
        if dont_add_if_dup and len(self.request_lifetime_events[request_id]) > 0 and dup_check(event):
            return
        self.request_lifetime_events[request_id].append(event)

    def _on_request_fail_callback(self, request: Request):
        self.failed_requests.append(request.request_id)
        print(f"Num failed requests: {len(set(self.failed_requests))}, request.policy: {request.policy}")
        
    async def initialize(self):
        prefill_init_tasks = []
        for engine in self.prefill_engines:
            prefill_init_tasks.append(asyncio.create_task(engine.initialize()))
        await asyncio.gather(
            *prefill_init_tasks,
            self.decoding_engine.initialize()
        )
        self.decoding_engine.init_migrate_pairs()
        self.engine_initialized = True
        
    def _remote_call_all_workers(
        self, 
        func_name: str, 
        *args
    ):
        """
        call func_name on all workers, blocked until all workers finish, and return all the results
        """
        handlers = self._remote_call_all_workers_async(func_name, *args)
        return ray.get(handlers)

    def _remote_call_all_workers_async(
        self, 
        func_name: str,
        *args
    ):
        """
        call func_name asynchronously on all workers (prefill/decoding/both), return the futures immediately
        """
        handlers = []
        for engine in self.prefill_engines:
            handlers += engine._remote_call_all_workers_async(func_name, *args)
        handlers += self.decoding_engine._remote_call_all_workers_async(func_name, *args)
        return handlers

    async def _start_my_event_loop(self):
        pass
    
    async def start_all_event_loops(self):
        """
        start_all_event_loops: Start prefill_engine's, decoding_engine's, and
        mine (LLMEngine's) event loops
        """
        logger.info("Starting LLMEngine's event loops")
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before starting event loops."
        prefill_tasks = []
        for engine in self.prefill_engines:
            prefill_tasks.append(asyncio.create_task(engine.start_event_loop()))
        await asyncio.gather(
            *prefill_tasks,
            self.decoding_engine.start_event_loop(),
            self.global_scheduler.start_event_loop()
            # self._start_my_event_loop()
        )
        
    async def generate(
        self,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[str]],
        sampling_params: SamplingParams,
        arrival_time: Optional[float] = None,
        request_id: Optional[int] = None,
    ) -> AsyncGenerator[StepOutput, None]:
        """
        generate - Generate outputs for one request
        
        This function is intended to be used as an async generator, i.e., it can be
        used in a for loop. For example, `async for output in engine.generate(...)`
        """
        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before generating."
        req = create_request(
            prompt,
            prompt_token_ids,
            sampling_params,
            self.request_counter,
            self.tokenizer,
            arrival_time,
            request_id,
        )
        self.request_outputs[req.request_id] = asyncio.Queue()
        self.request_lifetime_events[req.request_id] = []
        
        self._on_new_lifetime_event_callback(req.request_id, LifetimeEvent(LifetimeEventType.Issued))
        self.global_scheduler.add_request(req)
        
        while True:
            try:
                step_output = await self.request_outputs[req.request_id].get()
            except asyncio.CancelledError:
                # The engine returns
                # Exception should be handled by the engine, not me
                return
            except GeneratorExit:
                return
            yield step_output
            if step_output.is_finished:
                break
                
        del self.request_outputs[req.request_id]

    def abort_request(self, request_id: int):
        for engine in self.prefill_engines:
            engine.abort_request(request_id)
        self.decoding_engine.abort_request(request_id)

    async def warmup(self):
        async def warmup_task(policy: Policy):
            req = create_request(
                None,
                [10],
                SamplingParams(max_tokens=2, ignore_eos=True),
                self.request_counter,
                self.tokenizer,
                None,
                None,
                policy
            )
            self.request_outputs[req.request_id] = asyncio.Queue()
            self.request_lifetime_events[req.request_id] = []
            self.global_scheduler.add_request(req)
            while True:
                try:
                    step_output = await self.request_outputs[req.request_id].get()
                except asyncio.CancelledError:
                    return
                except GeneratorExit:
                    return
                if step_output.is_finished:
                    break
            del self.request_outputs[req.request_id]
            del self.request_lifetime_events[req.request_id]

        assert self.engine_initialized, "Engine not initialized. Please call engine.initialize() before generating."
        request_tasks = []
        for _ in range(len(self.prefill_engines)):
            request_tasks.append(asyncio.create_task(warmup_task(Policy.HPHD)))
            request_tasks.append(asyncio.create_task(warmup_task(Policy.HPLD)))
        request_tasks.append(asyncio.create_task(warmup_task(Policy.LPLD)))
        await asyncio.gather(*request_tasks)
        print("Done Warming Up Engine.")
        await asyncio.sleep(1)

        
def add_engine_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-dummy-weights", action="store_true")
    
    parser.add_argument("--prefill-pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--prefill-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--decoding-pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--decoding-tensor-parallel-size", type=int, default=1)
    
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-blocks-per-req", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--swap-space", type=int, default=16)
    
    parser.add_argument("--prefill-sched-policy", type=str, default="fcfs")
    parser.add_argument("--prefill-max-batch-size", type=int, default=256)
    parser.add_argument("--prefill-max-tokens-per-batch", type=int, default=4096)
    
    parser.add_argument("--decoding-sched-policy", type=str, default="fcfs")
    parser.add_argument("--decoding-max-batch-size", type=int, default=256)
    parser.add_argument("--decoding-max-tokens-per-batch", type=int, default=8192)
    
    parser.add_argument("--simulator-mode", action="store_true")
    parser.add_argument("--profiler-data-path", type=str, default=None)
    parser.add_argument("--gpu-mem-size-gb", type=float, default=None)
    