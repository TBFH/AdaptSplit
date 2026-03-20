from typing import List, Tuple, Optional, Dict, Any
import json
import random
import os, sys
import argparse
import tqdm

import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerBase, AutoTokenizer

from adaptsplit.agent.sentence_embedding.chat import SentenceEmbedder


'''
python -m adaptsplit.agent.prepare_training_datasets \
    --tokenizer /mnt/Data/austin/hf_models/opt-1.3b \
    --output-dir /home/austin/datasets/agent_training/ \
    --sample
'''

'''
python -m adaptsplit.agent.prepare_training_datasets \
    --tokenizer /mnt/Data/austin/hf_models/opt-1.3b \
    --output-dir /home/austin/datasets/agent_training/ \
    --reset
'''


# MODEL_PATH = "/mnt/Data/austin/hf_models/opt-1.3b"
# MODEL_PATH = "/mnt/Data/austin/hf_models/Llama-2-7b-chat-hf"
# MODEL_PATH = "/mnt/Data/austin/hf_models/Meta-Llama-3-8B-Instruct"

DATASET_PATH = {
    "sharegpt": "/home/austin/datasets/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
    "alpaca": "/home/austin/datasets/alpaca/alpaca_data_cleaned.json",
    "humaneval": "/home/austin/datasets/humaneval/standard_humaneval.jsonl",
    "longbench": "/home/austin/datasets/longbench/data"
}

TTFT_TPOT_SLO = {
    "sharegpt": {
        "opt-1.3b": (1500, 750),
        "Llama-2-7b-chat-hf": (2500, 1250),
        "Meta-Llama-3-8B-Instruct": (5000, 1500),
    },
    "alpaca": {
        "opt-1.3b": (1500, 750),
        "Llama-2-7b-chat-hf": (2500, 1250),
        "Meta-Llama-3-8B-Instruct": (500, 300),
    },
    "humaneval": {
        "opt-1.3b": (500, 1000),
        "Llama-2-7b-chat-hf": (1000, 1250),
        "Meta-Llama-3-8B-Instruct": (2000, 1500),
    },
    "longbench": {
        "opt-1.3b": (3000, 200),
        "Llama-2-7b-chat-hf": (15000, 750),
        "Meta-Llama-3-8B-Instruct": (20000, 1000),
    },
}


def sample_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_names: List[str],
    args: argparse.Namespace,
):
    result: List[Dict[str, Any]] = []
    id_counter = []
    embedder = SentenceEmbedder(
        save_dir=args.embedder_save_dir,
        response_id=args.embedder_response_id
    )

    def append_all(
        sets: List[Dict[str, Any]],
        dataset_name: str,
        prompt: str,
        prompt_len: int,
        output_len: int,
        id_counter: List
    ):
        sets.append({
            "request_id": len(id_counter),
            "dataset_name": dataset_name,
            "prompt": prompt,
            "input_length": prompt_len,
            "output_length": output_len,
            "ttft_slo_ms": TTFT_TPOT_SLO[dataset_name][args.model][0],
            "tpot_slo_ms": TTFT_TPOT_SLO[dataset_name][args.model][1],
            "embedding": embedder.embed(prompt)
        })
        id_counter.append(True)

    def save(sets: List[Dict[str, Any]], save_name: str):
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, save_name), 'w', encoding='utf-8') as f:
            json.dump(sets, f, ensure_ascii=False, indent=4)
            print(f"Saved {save_name}")


    if "sharegpt" in dataset_names:
        current_counter = 0
        is_train = True
        for_evals = []

        # Load the dataset.
        with open(DATASET_PATH["sharegpt"], "r") as f:
            dataset = json.load(f)
        
        for data in tqdm.tqdm(dataset):
            num_conversations = len(data["conversations"])
            
            # Filter out the conversations with less than args.sharegpt_min_turns turns.
            if num_conversations < args.sharegpt_min_turns or \
                num_conversations < args.sharegpt_min_prompt_turns + 1:
                continue
                
            num_prompt_turns = random.randint(
                args.sharegpt_min_prompt_turns,
                min(num_conversations - 1, args.sharegpt_max_prompt_turns)
            )
            
            start_idx = 0 if data["conversations"][0]["from"] == "human" else 1
            prompt = "\n".join([data["conversations"][i]["value"] for i in range(start_idx, num_prompt_turns)])
            if not prompt:
                continue
            completion = data["conversations"][num_prompt_turns]["value"]
            prompt_token_ids = tokenizer(prompt).input_ids
            completion_token_ids = tokenizer(completion).input_ids
            
            prompt_len = len(prompt_token_ids)
            output_len = len(completion_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len + output_len >= 2048:
                # Prune too long sequences. (It exceeded max_positional_embedding)
                continue
            
            if is_train:
                append_all(result, "sharegpt", prompt, prompt_len, output_len, id_counter)
                current_counter += 1
                if current_counter >= 500:
                    is_train = False
            else:
                append_all(for_evals, "sharegpt", prompt, prompt_len, output_len, id_counter)

        save(for_evals, f"{args.model}-sharegpt-eval.json")
    
    
    if "alpaca" in dataset_names:
        current_counter = 0
        is_train = True
        for_evals = []

        with open(DATASET_PATH["alpaca"], "r") as f:
            dataset = json.load(f)
        # extract the input and output
        dataset = [
            (data["instruction"] + data["input"], data["output"]) for data in dataset
        ]

        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

        # Filter out too long sequences.
        for prompt, prompt_token_ids, output_len in tqdm.tqdm(tokenized_dataset):
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 and output_len < 4:
                # Prune too short sequences.
                continue
            if prompt_len > 1024 or prompt_len + output_len > 2048:
                # Prune too long sequences.
                continue
            
            if is_train:
                append_all(result, "alpaca", prompt, prompt_len, output_len, id_counter)
                current_counter += 1
                if current_counter >= 500:
                    is_train = False
            else:
                append_all(for_evals, "alpaca", prompt, prompt_len, output_len, id_counter)

        save(for_evals, f"{args.model}-alpaca-eval.json")
    

    if "humaneval" in dataset_names:
        current_counter = 0
        is_train = True
        for_evals = []

        filtered_dataset = []
        with open(DATASET_PATH["humaneval"], "r") as f:
            for line in tqdm.tqdm(f.readlines()):
                if line.strip() == "":
                    continue
                data = json.loads(line)
                context = data["prompt"]
                context_token_ids = tokenizer(context).input_ids
                answer = data["canonical_solution"]
                answer_token_ids = tokenizer(answer).input_ids
                if len(context_token_ids) + len(answer_token_ids) >= 2048:
                    continue

                if is_train:
                    append_all(filtered_dataset, "humaneval", context, len(context_token_ids), len(answer_token_ids), id_counter)
                    current_counter += 1
                    if current_counter >= 80:
                        is_train = False
                else:
                    append_all(for_evals, "humaneval", context, len(context_token_ids), len(answer_token_ids), id_counter)
        
        save(for_evals, f"{args.model}-humaneval-eval.json")
        
        filtered_dataset = filtered_dataset * 6
        result = result + filtered_dataset
    

    if "longbench" in dataset_names:
        current_counter = 0
        is_train = True
        for_evals = []

        # find all .jsonl files under the dataset_path
        files = []
        for root, dirs, filenames in os.walk(DATASET_PATH["longbench"]):
            for filename in filenames:
                if filename.endswith(".jsonl"):
                    files.append(os.path.join(root, filename))
        
        for file in tqdm.tqdm(files):
            with open(file, "r") as f:
                for line in tqdm.tqdm(f.readlines()):
                    if line.strip() == "": continue
                    data = json.loads(line)
                    
                    context = data["context"][:40000]    # truncate to the first 40000 chars to reduce tokenization time
                    context_token_ids = tokenizer(context).input_ids
                    answer_token_ids = tokenizer(data["answers"][0]).input_ids
                    context_len = len(context_token_ids)
                    answer_len = len(answer_token_ids)
                    
                    context_len_allowed = min(2040 - answer_len, random.randint(args.longbench_min_prompt_len, args.longbench_max_prompt_len))
                    context_token_ids = context_token_ids[:context_len_allowed]
                    prompt = tokenizer.decode(context_token_ids)

                    if is_train:
                        append_all(result, "longbench", prompt, len(context_token_ids), answer_len, id_counter)
                        current_counter += 1
                        if current_counter >= 15:
                            is_train = False
                    else:
                        append_all(for_evals, "longbench", prompt, len(context_token_ids), answer_len, id_counter)
            current_counter = 0
            is_train = True
        
        save(for_evals, f"{args.model}-longbench-eval.json")
    

    random.shuffle(result)
    save(result, f"{args.model}-train.json")
    print(f"Total num request: {len(result)}")


def reset_dataset():
    def reset(file_path: str, reset_id: bool):
        with open(file_path, "r") as f:
            dataset = json.load(f)
        for idx, data in enumerate(dataset):
            if reset_id:
                data["request_id"] = idx
            data["ttft_slo_ms"] = TTFT_TPOT_SLO[data["dataset_name"]][args.model][0]
            data["tpot_slo_ms"] = TTFT_TPOT_SLO[data["dataset_name"]][args.model][1]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
            print(f"Reset SLO to {file_path}")
    
    reset(os.path.join(args.output_dir, f"{args.model}-train.json"), reset_id=True)
    for dataset_name in args.datasets:
        file_path = os.path.join(args.output_dir, f"{args.model}-{dataset_name}-eval.json")
        reset(file_path, reset_id=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="['alpaca', 'humaneval', 'longbench']")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--sharegpt-min-turns", type=int, default=2)
    parser.add_argument("--sharegpt-min-prompt-turns", type=int, default=1)
    parser.add_argument("--sharegpt-max-prompt-turns", type=int, default=1)
    
    parser.add_argument("--longbench-min-prompt-len", type=int, default=1900)
    parser.add_argument("--longbench-max-prompt-len", type=int, default=2048)

    parser.add_argument("--embedder-save-dir", type=str, default="/home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/sentence_embedding/generated")
    parser.add_argument("--embedder-response-id", type=int, default=0)

    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--reset", action="store_true")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.datasets = eval(args.datasets)
    args.model = args.tokenizer.split('/')[-1]
    
    if args.sample:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
        sample_dataset(tokenizer, args.datasets, args)
    if args.reset:
        reset_dataset()