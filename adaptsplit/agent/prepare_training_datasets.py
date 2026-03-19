# Todo：
## 采样给定数据集
## 对所有的采样的请求做好Sentence Embedding
## 对不同类型的数据集请求预先设定好SLO目标
## 数据集保存为json格式

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

# MODEL_PATH = "/mnt/Data/austin/hf_models/opt-1.3b"
# MODEL_PATH = "/mnt/Data/austin/hf_models/Llama-2-7b-chat-hf"
# MODEL_PATH = "/mnt/Data/austin/hf_models/Meta-Llama-3-8B-Instruct"

# DATASET_PATH = "/home/austin/datasets/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"


TTFT_TPOT_SLO = {
    "sharegpt": {
        "opt-1.3b": (1500, 750),
        "Llama-2-7b-chat-hf": (2500, 1500),
        "Meta-Llama-3-8B-Instruct": (3000, 2000)
    },
    "humaneval": {
        "opt-1.3b": (500, 750),
    }
}


def sample_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    output_path: str,
    args: argparse.Namespace,
):
    """
    sample_dataset: Sample the given dataset and return a list of request with sentence embedding for agent training.
    """

    result: List[Dict[str, Any]] = []
    id_counter = 0
    embedder = SentenceEmbedder(
        save_dir=args.embedder_save_dir,
        response_id=args.embedder_response_id
    )

    if dataset_name.lower() == "sharegpt":
        # Load the dataset.
        with open(dataset_path, "r") as f:
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
            # if prompt_len < 4 and output_len < 4:
            #     # Prune too short sequences.
            #     continue
            if prompt_len + output_len >= 2048:
                # Prune too long sequences. (It exceeded max_positional_embedding)
                continue
            
            result.append({
                "request_id": id_counter,
                "prompt": prompt,
                "input_length": prompt_len,
                "output_length": output_len,
                "ttft_slo_ms": TTFT_TPOT_SLO[args.dataset][args.model][0],
                "tpot_slo_ms": TTFT_TPOT_SLO[args.dataset][args.model][1],
                "embedding": embedder.embed(prompt)
            })
            id_counter += 1
            
            if len(result) > args.num_req:
                break

        
        
        filtered_dataset = []
        with open("/home/austin/datasets/humaneval/standard_humaneval.jsonl", "r") as f:
            for line in f.readlines():
                if line.strip() == "": continue
                data = json.loads(line)
                context = data["prompt"]
                context_token_ids = tokenizer(context).input_ids
                answer = data["canonical_solution"]
                answer_token_ids = tokenizer(answer).input_ids
                if len(context_token_ids) + len(answer_token_ids) >= 2048:
                    continue
                filtered_dataset.append({
                    "request_id": id_counter,
                    "prompt": context,
                    "input_length": len(context_token_ids),
                    "output_length": len(answer_token_ids),
                    "ttft_slo_ms": TTFT_TPOT_SLO["humaneval"][args.model][0],
                    "tpot_slo_ms": TTFT_TPOT_SLO["humaneval"][args.model][1],
                    "embedding": embedder.embed(context)
                })
                id_counter += 1
        filtered_dataset = filtered_dataset * 2
        result = result + filtered_dataset
        random.shuffle(result)
        
        # return Dataset(f"sharegpt-mt-{args.sharegpt_min_turns}-mipt-{args.sharegpt_min_prompt_turns}-mxpt-{args.sharegpt_max_prompt_turns}", result)
        # return Dataset(f"sharegpt", result)
    
    # elif dataset_name.lower() == "alpaca":
    #     with open(dataset_path, "r") as f:
    #         dataset = json.load(f)

    #     # extract the input and output
    #     dataset = [
    #         (data["instruction"] + data["input"], data["output"]) for data in dataset
    #     ]

    #     prompts = [prompt for prompt, _ in dataset]
    #     prompt_token_ids = tokenizer(prompts).input_ids
    #     completions = [completion for _, completion in dataset]
    #     completion_token_ids = tokenizer(completions).input_ids
    #     tokenized_dataset = []
    #     for i in range(len(dataset)):
    #         output_len = len(completion_token_ids[i])
    #         tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    #     # Filter out too long sequences.
    #     filtered_dataset: List[TestRequest] = []
    #     for prompt, prompt_token_ids, output_len in tokenized_dataset:
    #         prompt_len = len(prompt_token_ids)
    #         if prompt_len < 4 and output_len < 4:
    #             # Prune too short sequences.
    #             continue
    #         if prompt_len > 1024 or prompt_len + output_len > 2048:
    #             # Prune too long sequences.
    #             continue
    #         filtered_dataset.append(TestRequest(prompt, prompt_len, output_len))

    #     return Dataset("alpaca", filtered_dataset)

    # elif dataset_name.lower() == "mmlu":
    #     dataset = []
    #     choices = ["A", "B", "C", "D"]
    #     data_path = dataset_path
    #     subjects = sorted(
    #         [
    #             f.split("_test.csv")[0]
    #             for f in os.listdir(os.path.join(data_path, "test"))
    #             if "_test.csv" in f
    #         ]
    #     )

    #     for sub in subjects:
    #         test_df = pd.read_csv(
    #             os.path.join(data_path, "test", sub + "_test.csv"), header=None
    #         )
    #         for i in range(test_df.shape[0]):
    #             prompt = test_df.iloc[i, 0]
    #             k = test_df.shape[1] - 2
    #             for j in range(k):
    #                 prompt += "\n{}. {}".format(choices[j], test_df.iloc[i, j + 1])
    #             prompt += "\nAnswer:"
    #             output = test_df.iloc[i, k + 1]
    #             dataset.append((prompt, output))

    #     print("MMLU dataset size:", len(dataset))

    #     prompts = [prompt for prompt, _ in dataset]
    #     prompt_token_ids = tokenizer(prompts).input_ids
    #     completions = [completion for _, completion in dataset]
    #     completion_token_ids = tokenizer(completions).input_ids
    #     tokenized_dataset = []
    #     for i in range(len(dataset)):
    #         output_len = len(completion_token_ids[i])
    #         tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    #     # Filter out too long sequences.
    #     filtered_dataset: List[TestRequest] = []
    #     for prompt, prompt_token_ids, output_len in tokenized_dataset:
    #         prompt_len = len(prompt_token_ids)
    #         if prompt_len < 4 and output_len < 4:
    #             # Prune too short sequences.
    #             continue
    #         if prompt_len > 1024 or prompt_len + output_len > 2048:
    #             # Prune too long sequences.
    #             continue
    #         filtered_dataset.append(TestRequest(prompt, prompt_len, output_len))

    #     return Dataset("mmlu", filtered_dataset)

    # elif dataset_name.lower() == "longbench":
    #     # find all .jsonl files under the dataset_path
    #     files = []
    #     for root, dirs, filenames in os.walk(dataset_path):
    #         for filename in filenames:
    #             if filename.endswith(".jsonl"):
    #                 files.append(os.path.join(root, filename))
        
    #     filtered_dataset = []
    #     for file in tqdm.tqdm(files):
    #         with open(file, "r") as f:
    #             for line in f.readlines():
    #                 if line.strip() == "": continue
    #                 data = json.loads(line)
                    
    #                 context = data["context"][:40000]    # truncate to the first 40000 chars to reduce tokenization time
    #                 context_token_ids = tokenizer(context).input_ids
    #                 answer_token_ids = tokenizer(data["answers"][0]).input_ids
    #                 context_len = len(context_token_ids)
    #                 answer_len = len(answer_token_ids)
                    
    #                 context_len_allowed = min(2040 - answer_len, random.randint(args.longbench_min_prompt_len, args.longbench_max_prompt_len))
    #                 context_token_ids = context_token_ids[:context_len_allowed]
                    
    #                 filtered_dataset.append(TestRequest(
    #                     tokenizer.decode(context_token_ids),
    #                     len(context_token_ids),
    #                     answer_len
    #                 ))
                    
    #     # return Dataset(f"longbench-mipl-{args.longbench_min_prompt_len}-mxpl-{args.longbench_max_prompt_len}", filtered_dataset)
    #     return Dataset(f"longbench", filtered_dataset)
    
    # elif dataset_name.lower() == "humaneval":
    #     filtered_dataset = []
    #     with open(dataset_path, "r") as f:
    #         for line in f.readlines():
    #             if line.strip() == "": continue
    #             data = json.loads(line)
                
    #             context = data["prompt"]
    #             context_token_ids = tokenizer(context).input_ids
    #             answer = data["canonical_solution"]
    #             answer_token_ids = tokenizer(answer).input_ids
                
    #             if len(context_token_ids) + len(answer_token_ids) >= 2048:
    #                 continue
                
    #             filtered_dataset.append(TestRequest(
    #                 context,
    #                 len(context_token_ids),
    #                 len(answer_token_ids)
    #             ))
        
    #     # Copy the dataset for 10 times since it's too small.
    #     filtered_dataset = filtered_dataset * 10
        
    #     return Dataset("humaneval", filtered_dataset)
    
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.dataset}-{args.model}-train.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sharegpt")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--sharegpt-min-turns", type=int, default=2)
    parser.add_argument("--sharegpt-min-prompt-turns", type=int, default=1)
    parser.add_argument("--sharegpt-max-prompt-turns", type=int, default=1)
    
    parser.add_argument("--longbench-min-prompt-len", type=int, default=1900)
    parser.add_argument("--longbench-max-prompt-len", type=int, default=2048)
    
    parser.add_argument("--num-req", type=int, default=float('inf'))

    parser.add_argument("--embedder-save-dir", type=str, default="/home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/sentence_embedding/generated")
    parser.add_argument("--embedder-response-id", type=int, default=0)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.model = args.tokenizer.split('/')[-1]
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    dataset = sample_dataset(args.dataset_path, tokenizer, args.dataset, args.output_dir, args)
    print(f"Saved to {args.output_dir}")



    '''
    python -m adaptsplit.agent.prepare_training_datasets \
        --dataset sharegpt \
        --dataset-path /home/austin/datasets/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json \
        --tokenizer /mnt/Data/austin/hf_models/opt-1.3b \
        --output-dir /home/austin/datasets/agent_training/ \
        --num-req 5000
    '''
    