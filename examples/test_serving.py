import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='The model to use', default='/mnt/Data/austin/hf_models/opt-1.3b')
args = parser.parse_args()

# Sample prompts.
prompts = [
    "Life blooms like a flower. Far away or by the road. Waiting",
    "A quick brown fox",
    "Artificial intelligence is",
    "To be or not to be,",
    "one two three four"
] * 16

# Create a sampling params object.
sampling_params = []
# mt = [64, 32, 8, 42, 72] * 12
for i in range(len(prompts)):
    sampling_param = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=8, stop=[]
        # temperature=0.8, top_p=0.95, max_tokens=mt[i], stop=[]
    )
    sampling_params.append(sampling_param)

# Create an LLM for offline inference.
llm = OfflineLLM(
    model_config=ModelConfig(
        model=args.model,
        tokenizer=None
    ),
    disagg_parallel_config=DisaggParallelConfig(
        prefill=ParallelConfig(
            data_parallel_size=2
        ),
        decoding=ParallelConfig(
            pipeline_parallel_size=4,
            # pipeline_distribution=[9, 6, 6, 3]
        )
    ),
    prefill_devices=['pc-3090', 'pc-4090'],
    decoding_devices=['jetson-64g-4', 'jetson-16g-2', 'jetson-16g-8', 'jetson-8g-1'],
    cache_config=CacheConfig(
        block_size=16,
        max_num_blocks_per_req=1024,
        gpu_memory_utilization=0.8,
        cpu_swap_space=1.0
    ),
    prefill_sched_config=PrefillStageSchedConfig(
        policy="fcfs",
        max_batch_size=5,
        max_tokens_per_batch=16384
    ),
    decoding_sched_config=DecodingStageSchedConfig(
        policy="fcfs",
        max_batch_size=5,
        max_tokens_per_batch=16384,
        waiting_block_prop_threshold=0.5    # 可调整以防止Prefill死锁
    ),
    global_schedule_policy="lpld",
    extra_configs=ExtraConfig(
        print_log=False,
        sched_bar=False,
        req_pbar=True,
        enable_records=False,
        records_dir="/home/austin/repos/AdaptSplit/evals/stats/testing",
        pptimer_url="http://219.222.20.79:31063",
        prebenchmark=False
    )
)

# Generate texts from the prompts. The output is a list of Request objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
# llm.collect_records()

# Print the outputs.
# for prompt, step_outputs in zip(prompts, outputs):
#     # new_token_ids = [step_output.new_token_id for step_output in step_outputs]
#     # output_text = llm.tokenizer.decode(new_token_ids)
#     print(
#         f"Prompt: {prompt!r}, Generated text: {' '.join([step_output.new_token for step_output in step_outputs])!r} ({len(step_outputs)} tokens generated)."
#     )
