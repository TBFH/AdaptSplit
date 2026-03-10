from adaptsplit import OfflineLLM, SamplingParams
from adaptsplit.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    PrefillStageSchedConfig,
    DecodingStageSchedConfig,
    ExtraConfig
)

import argparse
import time
from typing import List, Dict

from partitioning.utils import profile_power, RainbowLogger, JsonHelper

logger = None
json_helper = None

# MODEL_PATH = "/mnt/Data/austin/hf_models/opt-1.3b"
MODEL_PATH = "/mnt/Data/austin/hf_models/Llama-2-7b-chat-hf"
# MODEL_PATH = "/mnt/Data/austin/hf_models/Meta-Llama-3-8B-Instruct"

def profiling(
    deployments: List[str],
    max_batch_size: int,
    nlayer_threshold: int,
    args
) -> List[List[int]]:
    llm = OfflineLLM(
        model_config=ModelConfig(
            model=MODEL_PATH,
            tokenizer=None
        ),
        disagg_parallel_config=DisaggParallelConfig(
            prefill=ParallelConfig(
                data_parallel_size=1
            ),
            decoding=ParallelConfig(
                pipeline_parallel_size=len(deployments),
                pipeline_distribution=[1] * len(deployments)
            )
        ),
        prefill_devices=['pc-4090'],
        decoding_devices=deployments,
        cache_config=CacheConfig(
            block_size=16,
            max_num_blocks_per_req=1024,
            gpu_memory_utilization=args.vram_util,
            cpu_swap_space=1.0
        ),
        prefill_sched_config=PrefillStageSchedConfig(
            policy="fcfs",
            max_batch_size=16,
            max_tokens_per_batch=16384
        ),
        decoding_sched_config=DecodingStageSchedConfig(
            policy="fcfs",
            max_batch_size=16,
            max_tokens_per_batch=16384,
            waiting_block_prop_threshold=0.5
        ),
        global_schedule_policy="lpld",
        extra_configs=ExtraConfig(
            pb_profile=True,
            pb_nlayer_thres=nlayer_threshold,
            pb_max_batchsize=max_batch_size
        )
    )
    prefiling_res = llm.collect_prebenchmark_profiles()

    del llm
    import gc
    # print("Collecting Garbage ...")
    gc.collect()
    time.sleep(30)
    logger.warning(f"Prebenchmark Profiling Done")

    return prefiling_res


def pre_benchmarks(
    deployments: List[str],
    batch_sizes: List[int],
    num_layer: int,
    n_input_tokens: int,
    n_output_tokens: int
) -> Dict:
    benchmarks = []
    powers = []

    for bs in batch_sizes:
        llm = OfflineLLM(
            model_config=ModelConfig(
                model=MODEL_PATH,
                tokenizer=None
            ),
            disagg_parallel_config=DisaggParallelConfig(
                prefill=ParallelConfig(
                    data_parallel_size=1
                ),
                decoding=ParallelConfig(
                    pipeline_parallel_size=len(deployments),
                    pipeline_distribution=[num_layer] * len(deployments)
                )
            ),
            prefill_devices=['pc-4090'],
            decoding_devices=deployments,
            cache_config=CacheConfig(
                block_size=16,
                max_num_blocks_per_req=1024,
                gpu_memory_utilization=args.vram_util,
                cpu_swap_space=1.0
            ),
            prefill_sched_config=PrefillStageSchedConfig(
                policy="fcfs",
                max_batch_size=bs,
                max_tokens_per_batch=16384
            ),
            decoding_sched_config=DecodingStageSchedConfig(
                policy="fcfs",
                max_batch_size=bs,
                max_tokens_per_batch=16384,
                waiting_block_prop_threshold=0.5
            ),
            global_schedule_policy="lpld",
            extra_configs=ExtraConfig(
                req_pbar=True,
                prebenchmark=True
            )
        )
        # 准备数据
        text = " ".join(["a" for _ in range(1, n_input_tokens+1)])
        requests = [(text, n_input_tokens, n_output_tokens) for _ in range(bs*4*30)]
        # 构造输入
        prompts = []
        sampling_params_list = []
        for prompt, input_len, output_len in requests:
            sampling_params = SamplingParams(
                ignore_eos=True,
                max_tokens=output_len,
            )
            prompts.append(prompt)
            sampling_params_list.append(sampling_params)
        # 初始化worker会导致功耗激增，需冷却一会儿，避免影响推理过程的功耗记录
        print("Preparing for inference ...")
        time.sleep(30)
        # 测试批大小
        start = time.time()
        llm.generate(prompts=prompts, sampling_params=sampling_params_list)
        prebenchmarks = llm.collect_exec_times()
        bm = [(sum(stats)/len(stats)) for stats in prebenchmarks]
        node_powers = profile_power(deployments, time.time() - start, 1)
        ps = []
        for node in deployments:
            ps.append(node_powers[node])
        powers.append(ps)
        benchmarks.append(bm)

        # 手动垃圾回收
        del llm
        import gc
        print("Collecting Garbage ...")
        gc.collect()
        time.sleep(30)
        logger.warning(f"Pre-Benchmark Num_Layer={num_layer} Batch_Size={bs} Done")
        logger.green(f"benchmarks: {bm} \n powers: {ps}")
    
    # print("benchmarks", benchmarks)
    # print("powers", powers)
    benchmarks = list(zip(*benchmarks))
    powers = list(zip(*powers))

    return {
        "model": MODEL_PATH.split('/')[-1],
        "devices": deployments,
        "num_layer": num_layer,
        "batch_sizes": batch_sizes,
        "n_input_tokens": n_input_tokens,
        "n_output_tokens": n_output_tokens,
        "batch_latencys_ms": [[j*1000 for j in i] for i in benchmarks],
        "powers_W": powers,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-length', type=int, default=1)
    parser.add_argument('-o', '--output-length', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--vram-util', type=float, default=0.8)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--record', action='store_true', default=False)
    parser.add_argument('--json', action='store_true', default=False)
    parser.add_argument('--deployments', type=str, default="['jetson-64g-4', 'jetson-16g-2', 'jetson-16g-8', 'jetson-8g-1']")
    args = parser.parse_args()

    # Logger init
    if args.log:
        logger = RainbowLogger("/home/austin/repos/AdaptSplit/evals/stats/logs").logger
    else:
        logger = RainbowLogger().logger
    # Json Helper init
    if args.json:
        json_helper = JsonHelper(dir_path="/home/austin/repos/AdaptSplit/evals/stats/jsons")
    # Devices for Deployment
    deployments = eval(args.deployments)

    # Prebenchmark Profiling
    profiles = profiling(
        deployments=deployments,
        max_batch_size=300,
        nlayer_threshold=6,
        args=args
    )
    logger.warning(f"Prebenchmark Profiling Results: {profiles}")

    # 测试batch_size与各节点的推理时间及功耗的关系
    for i, deployments in enumerate(profiles):
        res = pre_benchmarks(
            deployments=deployments,
            batch_sizes=[1, 4, 8, 16, 32, 64, 128, 256],
            num_layer=i+1,
            n_input_tokens=args.input_length,
            n_output_tokens=args.output_length,
        )
        if json_helper:
            json_helper.append_dict(res)
