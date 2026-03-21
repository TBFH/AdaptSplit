# LaRe-style PPO scheduler for edge LLM routing

This package adapts the **LaRe-RD** idea to your heterogeneous edge inference scheduling problem.

Files:
- `env_wrapper.py`: custom episodic environment that interacts with your inference engine over HTTP.
- `prompt_template.py`: standardized task/state/action prompt for LLM latent reward generation.
- `chat_with_gpt.py`: environment prompting + self-prompting + pre-verification.
- `module.py`: shared MLP and reward decoder `f_psi`.
- `RD.py`: state-based return decomposition baseline.
- `LLMrd.py`: **LaRe-RD** = LLM latent reward encoder + RD reward decoder.
- `PPO.py`: discrete PPO agent and replay buffers.
- `ppo_main.py`: training entrypoint.

## Expected request dataset schema
A JSON list or JSONL stream where each row contains at least:

```json
{
  "request_id": "req-001",
  "prompt": "translate this paragraph ...",
  "input_length": 1536,
  "output_length_hint": 256,
  "ttft_slo_ms": 1200,
  "tpot_slo_ms": 80,
  "meta": {
    "temperature": 0.7,
    "max_new_tokens": 256
  }
}
```

The environment accepts common aliases such as `text`, `input_tokens`, `max_new_tokens`, etc.

## Expected engine endpoints
All endpoints are POST JSON endpoints.

### `/generate`
Input includes the original request plus `strategy ∈ {HPHD, HPLD, LPLD}`.
Output should ideally include per-request metrics such as:

```json
{
  "request_id": "req-001",
  "ttft_ms": 850.0,
  "tpot_ms": 72.5
}
```

### `/profile`
Should return cluster resource signals, for example:

```json
{
  "h_queue_len": 3,
  "l_queue_len": 7,
  "h_kv_cache_util": 0.42,
  "l_kv_cache_util": 0.55,
  "h_batch_size": 8,
  "l_batch_size": 16,
  "h_inflight": 5,
  "l_inflight": 9
}
```

### `/summary`
Should return episode-level throughput and power statistics, for example:

```json
{
  "throughput": 12.4,
  "avg_total_power": 645.2
}
```

## Training

```bash
python ppo_main.py \
  --env-config /path/to/env_config.json \
  --rd-method LaRe_RD \
  --total-episodes 500 \
  --llm-model gpt-4.1 \
  --llm-candidates 5 \
  --state-norm
```

## Notes
- `LaRe_RD` follows the paper's high-level structure: LLM defines a **latent reward encoder** `ϕ(s,a)`, and a learned reward decoder `f_psi` maps those factors into step-wise proxy rewards.
- The action space is discrete and matches your three routing strategies.
- By default the code uses a placeholder hashing embedder. Replace it with your real request embedding module, or pre-store embeddings in the dataset.



## 运行流程

- 首先运行`sentence_embedding/chat.py`内的main，让LLM辅助得到Sentence Embedder，保存到目录下的`generated`文件下，生成`dialog.json`对话回答过程，以及最终可行答案`response.npy`。
- 接着，运行`prepare_training_datasets.py`，指定原始数据集路径，同时加载上面的Sentence Embedder，即找到`response.npy`并加载，收集请求的prompt，input_len等信息，并将prompt经过embbed之后的向量一起保存为json格式的数据集；
- 运行`reward_model/chat_with_llm.py`内的main，让LLM辅助得到Latent Reward函数，保存到目录下的`generated`文件下，生成`dialog.json`对话回答过程，`factor_num`潜在向量维度，以及最终可行答案`response.npy`。方便训练时直接读取得到Latent Reward函数以及`factor_num`。
- 开始训练前，启动在线推理引擎AsyncLLM（注意配置好需要部署的模型，需要与将要用于训练的模型和数据集、SLO目标等对应）
- 接着则可以运行`ppo_main.py`开始训练agent。
- 在global_scheduler中加载agent，以及sentence embedder，其中时AsyncLLM中global_scheduler_policy设置为`default`，即通过agent做请求的调度决策。对于到来的请求，先进行sentence embedding，并获取请求对应SLO及各方面资源情况，通过`_compose_state`构造状态state，给到agent得到action，继而实现调度。


1. 使用以下命令启动在线引擎：

```bash
python -m adaptsplit.api_server.adaptsplit_api_server \
    --host localhost \
    --port 32313 \
    --model /mnt/Data/austin/hf_models/Meta-Llama-3-8B-Instruct \
    --decoding-devices "['jetson-64g-4', 'jetson-16g-2', 'jetson-16g-8', 'jetson-16g-7']" \
    --global-schedule-policy random \
    --auto-batchsize
```

2. 使用以下命令开始PPO开始训练：

```bash
python -m adaptsplit.agent.ppo_main \
    --env-config /home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/env_config.json \
    --output-dir /home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/outputs \
    --llm-model glm-5 \
    --llm-response-dir /home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/reward_model/generated \
    --state-norm
```


## 在线调度推理

在线调度时其实只需要 final_actor.pt。final_critic.pt 是 PPO 训练时做 value 估计用的，final_reward_model.pt 是 LaRe-RD 训练时把 latent reward factors 映射成 proxy reward 用的；真正部署时，决策链路是 state → policy network → action，不是靠 reward model 做动作选择。

训练结束后，目录将包含：
- final_actor.pt
- final_critic.pt
- final_reward_model.pt
- history.json
- deploy_meta.json
- final_state_norm.npz（如果训练时开了 --state-norm）

结论：
- final_actor.pt：必须要，在线调度就靠它。
- final_critic.pt：在线调度不需要，只在 PPO 训练里估计 value。
- final_reward_model.pt：在线调度通常不需要，它是 LaRe-RD 训练期把 latent reward factors 变成 proxy reward 的 decoder。LaRe 论文里也是先用 fψ(ϕ(s,a)) 生成 proxy rewards，再用这些 proxy rewards 去优化 policy；真正部署时执行的是训练好的 policy。

需要注意的几点：
- 第一，state 的构造顺序必须和训练完全一致。也就是 embedding、input_lengt等state信息 这些顺序不能变。不然 actor 读到的就不是训练分布。
- 第二，embedding 模型必须一致。训练时如果你用的是某个真实 sentence embedding 模型，在线也必须用同一个；不能训练时用 A，部署时换成完全不同的 B。LaRe 的 latent reward 是建立在符号状态之上的，状态语义漂了，策略也会漂。论文也强调 latent reward encoder 是基于任务状态语义来工作的。
- 第三，如果训练时用了 --state-norm，在线必须加载同一份 norm 统计量。这是最容易忽略、也最容易导致“离线验证很好、在线动作很怪”的原因。

