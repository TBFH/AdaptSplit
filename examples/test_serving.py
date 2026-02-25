import argparse
from adaptsplit import OfflineLLM, SamplingParams
from adaptsplit.config import (
    ModelConfig,
    DisaggParallelConfig,
    ParallelConfig,
    CacheConfig,
    PrefillStageSchedConfig,
    DecodingStageSchedConfig
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
]

# Create a sampling params object.
# sampling_params = SamplingParams(
#     # temperature=0.8, top_p=0.95, max_tokens=64, stop=["\n"]
#     temperature=0.8, top_p=0.95, max_tokens=64, stop=[]
# )
sampling_params = []
mt = [64, 32, 8, 42, 72]
for i in range(len(prompts)):
    sampling_param = SamplingParams(
        # temperature=0.8, top_p=0.95, max_tokens=64, stop=["\n"]
        temperature=0.8, top_p=0.95, max_tokens=mt[i], stop=[]
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
            pipeline_parallel_size=1
        ),
        decoding=ParallelConfig(
            pipeline_parallel_size=4,
            pipeline_distribution=[9, 6, 6, 3]
        )
    ),
    prefill_devices=['pc-3090'],
    decoding_devices=['jetson-64g-1', 'jetson-64g-4', 'jetson-16g-2', 'jetson-16g-8'],
    cache_config=CacheConfig(
        block_size=16,
        max_num_blocks_per_req=1024,
        gpu_memory_utilization=0.8,
        cpu_swap_space=1.0
    ),
    prefill_sched_config=PrefillStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    ),
    decoding_sched_config=DecodingStageSchedConfig(
        policy="fcfs",
        max_batch_size=4,
        max_tokens_per_batch=16384
    )
)

# Generate texts from the prompts. The output is a list of Request objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

# Print the outputs.
for prompt, step_outputs in zip(prompts, outputs):
    # new_token_ids = [step_output.new_token_id for step_output in step_outputs]
    # output_text = llm.tokenizer.decode(new_token_ids)
    print(
        f"Prompt: {prompt!r}, Generated text: {' '.join([step_output.new_token for step_output in step_outputs])!r} ({len(step_outputs)} tokens generated)."
    )
