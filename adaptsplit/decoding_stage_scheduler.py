from abc import ABC, abstractmethod
import copy
from typing import List, Callable, Tuple
import warnings
import torch
from tqdm import tqdm
import math

from adaptsplit.config import ParallelConfig, DecodingStageSchedConfig, ExtraConfig
from adaptsplit.logger import init_logger
from adaptsplit.request import Request, BatchedRequests, MigratingRequest
from adaptsplit.profiling import ProfilingDatabase
from adaptsplit.block_manager import BlockManager, BlockLocation
from adaptsplit.utils import Policy

logger = init_logger(__name__)

UPDATE_FREQUENCY = 2

class DecodingStageScheduler(ABC):
    """The abstract class for a decoding stage scheduler.
    It should maintain all the requests in the current systems and their
    runtime statistics which are needed for scheduling. Before each iteration
    begins, the LLMEngine will call get_next_batch() method to get a
    BatchedRequets object for the next iteration. After each iteration ends,
    the LLMEngine will call the pop_finished_requests() method to get the
    finished requests in the current iteration.
    """
    
    @abstractmethod
    def add_request(self, request: MigratingRequest) -> None:
        """
        Add a request to the scheduler.
        NOTE. The scheduler may choose to migrate the request proactively to
        improve the performance.
        """
        raise NotImplementedError()

    @abstractmethod
    def abort_request(self, request_id: int) -> None:
        """
        Abort a request from the scheduler.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_next_batch(self) -> BatchedRequests:
        """
        Get a batch of requests for the execution of next iteration.
        """
        raise NotImplementedError()

    @abstractmethod
    def pop_finished_requests(self) -> List[Request]:
        """
        Pop the finished requests from the scheduler.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_total_num_requests(self) -> int:
        """
        Get the total number of requests in the system.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_processing_num_requests(self) -> int:
        """
        Get the number of requests that are being processed.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_waiting_num_requests(self) -> int:
        """
        Get the number of requests that are waiting for processing.
        """
        raise NotImplementedError()

    @abstractmethod
    def print_status(self) -> None:
        """
        Print the status of the scheduler.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def update_pbar(self) -> None:
        """
        Show the progress bar of the scheduler.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def reset_batchsize_counter(self) -> None:
        """
        Clear all data about batchsize_counter.
        """
        raise NotImplementedError()
    
    async def post_process(self) -> None:
        """
        Post process after each iteration.
        """
        pass


class DecodingStageFCFSScheduler(DecodingStageScheduler):
    """A first-come-first-serve scheduler.
    Note: It supports pipeline parallelism. It maintains #pp disjoint batches which
    are in the pipeline under execution.
    Note: The requests are in waiting_queue or the batch_queues, and one request
    can only be in one queue at a time.
    """

    def __init__(
        self,
        sched_config: DecodingStageSchedConfig,
        parallel_config: ParallelConfig,
        block_manager: BlockManager,
        engine_migrate_block_callback: Callable,
        extra_configs: ExtraConfig
    ):
        assert (
            sched_config.policy == "fcfs"
        ), f"can not initialize a FCFS scheduler with policy {sched_config.policy}"
        self.sched_config = sched_config
        # If the request has not been accepted (i.e. it still resides in the "bridge" queu
        # and its block are still on the prefill stage engine's side), then it will be put
        # into the unaccepted queue.
        self.unaccepted_queue: List[MigratingRequest] = []
        # If the current batch is full, the requests will be put into the waiting queue.
        self.waiting_queue: List[Request] = []
        # If one request was in batch_queues before, but swapped out, it will be put into the swapped queue.
        self.swapped_queue: List[Request] = []
        # Since pipeline parallelism is used, there are multiple batches in the system.
        self.cur_index = -1
        self.batch_queues = [
            BatchedRequests() for i in range(parallel_config.pipeline_parallel_size)
        ]
        self.parallel_config = copy.deepcopy(parallel_config)
        self.extra_configs = copy.deepcopy(extra_configs)
        self.block_manager = block_manager
        self.engine_migrate_block_callback = engine_migrate_block_callback

        self.gpu_blocks = block_manager.max_num_gpu_blocks
        self.req_len_history = []
        self.sum_finished_req = 0
        self.init_batch_size = self.sched_config.max_batch_size

        if extra_configs.sched_bar:
            self.unaccepted_q_bar = None
            self.waiting_q_bar = None
            self.pbar_dict = []
            self.init_pbars()

    def init_pbars(self):
        # Unaccepted Queue
        self.unaccepted_q_bar = tqdm(
            total=999,
            desc=f'[Decoding] Unaccepted_Queue', 
            bar_format='{l_bar}{bar}| num_requests: {n_fmt}',
            # position=0,
            leave=True,
            ncols=100
        )
        # Waiting Queue
        self.waiting_q_bar = tqdm(
            total=999,
            desc=f'[Decoding] Waiting_Queue',
            bar_format='{l_bar}{bar}| num_requests: {n_fmt}',
            # position=1,
            leave=True,
            ncols=100
        )
        for i in range(len(self.batch_queues)):
            # 初始化进度条：固定格式，默认颜色
            pbar = tqdm(
                total=self.sched_config.max_batch_size,
                desc=f'[Decoding] Batch_Queue[{i}]',                           # 进度条左侧显示对象名称
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
                # position=2+i,   # 固定位置，避免刷屏
                leave=True,         # 运行结束后保留进度条
                ncols=100
            )
            self.pbar_dict.append(pbar)

    def update_pbar(self):
        COLOR_CODES = {
            'green': '\033[92m',
            'reset': '\033[0m',
        }
        # 更新队列信息
        self.unaccepted_q_bar.n = len(self.unaccepted_queue)
        self.unaccepted_q_bar.refresh()
        self.waiting_q_bar.n = len(self.waiting_queue)
        self.waiting_q_bar.refresh()
        # 更新各批次占用率
        for idx, batch in enumerate(self.batch_queues):
            tokens_to_cal = batch.get_num_input_tokens()
            num_req = len(batch)
            self.pbar_dict[idx].n = num_req
            if idx == self.cur_index:
                bar_format = f'{COLOR_CODES["reset"]}{{l_bar}}{COLOR_CODES["green"]}{{bar}}{COLOR_CODES["reset"]}| {{n_fmt}}/{{total_fmt}} [num_tokens: {tokens_to_cal}]'
            else:
                bar_format = f'{{l_bar}}{{bar}}| {{n_fmt}}/{{total_fmt}} [num_tokens: {tokens_to_cal}]'
            self.pbar_dict[idx].bar_format = bar_format
            self.pbar_dict[idx].refresh()

    def _update_max_batch_size(self, avg_req_len: int):
        print(f"\033[92m [decoding_stage_scheduler] avg_req_len: {avg_req_len} \033[0m")
        avg_block_per_req = avg_req_len / self.block_manager.cache_config.block_size
        max_req_on_fly = self.gpu_blocks / avg_block_per_req
        max_safe_batchsize = int(max_req_on_fly / self.parallel_config.pipeline_parallel_size)
        self.sched_config.max_batch_size = max(max_safe_batchsize, 16)
        print(f"\033[92m [decoding_stage_scheduler] max_batch_size updated to {self.sched_config.max_batch_size} \033[0m")

    def reset_batchsize_counter(self):
        self.req_len_history.clear()
        self.sum_finished_req = 0
        self.sched_config.max_batch_size = self.init_batch_size

    def _get_block_needed(self, length: int):
        block_size = self.block_manager.cache_config.block_size
        return (length + block_size - 1) // block_size
        
    def _check_add_to_cur_batch(self, request: Request) -> bool:
        batch_size_target = math.ceil((
            sum([len(batch) for batch in self.batch_queues])
            + len(self.waiting_queue)
        ) / self.parallel_config.pipeline_parallel_size)

        return (
            # len(self.batch_queues[self.cur_index]) < self.sched_config.max_batch_size
            len(self.batch_queues[self.cur_index]) < min(batch_size_target, self.sched_config.max_batch_size)
        ) and (
            self.batch_queues[self.cur_index].get_num_input_tokens()
            + request.get_num_input_tokens()
            <= self.sched_config.max_tokens_per_batch
        ) and (
            sum([
                sum([
                    self._get_block_needed(req.get_input_len() + req.get_output_len())
                    for req in self.batch_queues[index].requests
                ])
                for index in range(self.parallel_config.pipeline_parallel_size)
            ]) 
            + sum([
                self._get_block_needed(req.get_input_len() + req.get_output_len())
                for req in self.waiting_queue if req.policy == Policy.HPLD and req.request_id != request.request_id
            ]) 
            + self._get_block_needed(request.get_input_len() + request.get_output_len())
            <= self.block_manager.max_num_gpu_blocks
        )
    
    # Add Prefill Requests
    def add_new_request(self, request: Request) -> None:
        self.waiting_queue.append(request)

    # Requests-related methods
    async def add_request(self, migrating_req: MigratingRequest) -> None:
        # We take a simple approach here: Accept any request that comes in.
        self.unaccepted_queue.append(migrating_req)

    def abort_request(self, request_id: int) -> None:
        # scan the current batch
        for queue in self.batch_queues:
            for _, request in enumerate(queue.requests):
                if request.request_id == request_id:
                    # This request may be under processed by the model currently,
                    # so it is not safe to delete it from current batch directly.
                    # Mark it as finished will release the resources it holds finally.
                    request.is_finished = True
                    return

        # scan the waiting queue
        for i, request in enumerate(self.waiting_queue):
            if request.request_id == request_id:
                del self.waiting_queue[i]
                return

    def _get_last_stage_batch(self) -> BatchedRequests:
        last_stage_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size
        return self.batch_queues[last_stage_index]

    def pop_finished_requests(self) -> List[Request]:
        finished_reqs = self._get_last_stage_batch().pop_finished_requests()

        if self.extra_configs.auto_batchsize:
            for req in finished_reqs:
                self.req_len_history.append(req.get_input_len() + req.get_output_len())
            self.sum_finished_req += len(finished_reqs)
            if self.sum_finished_req >= UPDATE_FREQUENCY:
                self.sum_finished_req = self.sum_finished_req % UPDATE_FREQUENCY
                avg_req_len = int(sum(self.req_len_history) / len(self.req_len_history))
                self._update_max_batch_size(avg_req_len)

        return finished_reqs

    def get_next_batch(self) -> BatchedRequests:
        self.cur_index = (
            self.cur_index + 1
        ) % self.parallel_config.pipeline_parallel_size

        # Check whether the blocks on GPU is enough for the next batch.
        # If not, swap out the last request
        # while sum([
        #     sum([
        #         self._get_block_needed(req.get_input_len() + req.get_output_len())
        #         for req in self.batch_queues[index].requests
        #     ])
        #     for index in range(self.parallel_config.pipeline_parallel_size)
        # ]) + sum([
        #     self._get_block_needed(req.get_input_len())
        #     for req in self.waiting_queue
        # ]) > self.block_manager.max_num_gpu_blocks:
        #     logger.info("No enough GPU blocks. Swap-out triggered")
        #     request = self.batch_queues[self.cur_index].requests.pop(-1)
        #     self.swapped_queue.append(request)
        #     self.block_manager.swap_out_requests([request])

        # Try to add in some new requests. Consider requests in the swapped queue first.
        while len(self.waiting_queue) > 0:
            request = self.waiting_queue[0]
            if self._check_add_to_cur_batch(request):
                self.batch_queues[self.cur_index].add_request(request)
                self.waiting_queue.pop(0)
            else:
                break
        return self.batch_queues[self.cur_index]

    # Getter functions
    def get_total_num_requests(self) -> int:
        return self.get_processing_num_requests() + self.get_waiting_num_requests()

    def get_processing_num_requests(self) -> int:
        num = 0
        for batch in self.batch_queues:
            num = num + len(batch.requests)
        return num

    def get_waiting_num_requests(self) -> int:
        return len(self.waiting_queue)

    def __repr__(self) -> str:
        return (
            f"FCFS(max_batch_size={self.sched_config.max_batch_size}, "
            f"max_tokens_per_batch={self.sched_config.max_tokens_per_batch})"
        )
    
    def print_status(self) -> None:
        logger.info(f"(decoding) requests: {len(self.unaccepted_queue)} unaccepted, {len(self.waiting_queue)} waiting, {self.get_processing_num_requests()} processing")

    async def post_process(self) -> None:
        def should_accept(migrating_req: MigratingRequest) -> bool:
            return sum([self._get_block_needed(len(req.prompt_token_ids))
                        for req in self.waiting_queue if req.policy == Policy.HPLD
                    ]) < self.block_manager.max_num_gpu_blocks * self.sched_config.waiting_block_prop_threshold \
                    and self._get_block_needed(len(migrating_req.req.prompt_token_ids) + migrating_req.req.get_output_len()) <= self.block_manager.get_num_avail_gpu_blocks()
        while len(self.unaccepted_queue) > 0:
            migrating_req = self.unaccepted_queue[0]
            if should_accept(migrating_req):
                self.unaccepted_queue.pop(0)
                await self.engine_migrate_block_callback(migrating_req)
                self.waiting_queue.append(migrating_req.req)
            else:
                break
    
def get_decoding_stage_scheduler(
    sched_config: DecodingStageSchedConfig,
    parallel_config: ParallelConfig,
    block_manager: BlockManager,
    engine_migrate_block_callback: Callable,
    extra_configs: ExtraConfig
) -> DecodingStageScheduler:
    if sched_config.policy == "fcfs":
        return DecodingStageFCFSScheduler(sched_config, parallel_config, block_manager, engine_migrate_block_callback, extra_configs)
    else:
        raise NotImplementedError(
            f"scheduler policy {sched_config.policy} is not supported"
        )
        