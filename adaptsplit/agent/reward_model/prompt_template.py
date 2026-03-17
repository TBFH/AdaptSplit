import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np


Factor_role_instrutor = "ROLE INSTRUCTION: You are good at understanding systems problems and writing Python code. \
You will receive: (1) the task objective, (2) the meaning of every state dimension, and (3) the meaning of each action. \
First understand the scheduling problem. Then identify a small set of reward-related latent factors that together evaluate the current state-action pair from different aspects. \
The latent factors should be semantically interpretable and directly related to long-term episode success, such as SLO risk, energy efficiency, load balancing, KV-cache transfer overhead, \
or resource headroom, but you must decide the factors from the provided state/action meaning rather than inventing hidden variables. \
Finally write a Python function `evaluation_func(state, action)` where `state` has shape (batch_size, state_dim) and `action` has shape (batch_size, 1). Both state and action are NumPy array \
Return a Python list of several NumPy arrays, each with shape (batch_size, 1), where each array is one latent reward factor. \
Do not use information that is not explicitly represented. Avoid division by zero. Keep the implementation generic and vectorized. \
Return JSON with exactly these keys: `Understand`, `Analyze`, `Functions`. `Functions` must contain only valid Python code. \
Required JSON schema:\n \
{\n \
    Understand: brief task understanding,\n \
    Analyze: step-by-step latent factor analysis,\n \
    Functions: 'def evaluation_func(state, action):\\n    ...\\n    return [a list of evaluation factor arrays]'\n \
}\n"

Task_description = "You are analyzing a reinforcement-learning environment for online scheduling of heterogeneous edge LLM inference. \
Each action chooses one routing strategy for the current request. The long-term objective is to maximize energy efficiency \
(throughput divided by average total power) while minimizing TTFT/TPOT SLO violations across the whole episode. \
The episode reward is R = w1 * (throughput / average_total_power) - w2 * violation_rate. \
Requests are served by two node groups: H (high performance, high power) and L (low performance, low power). \
Inference method in node group H use data parallelism, where each node is capable of loading the entire LLM. \
Inference method in node group L use layer-wise pipeline parallelism, where each node is loaded with different part of the LLM. \
Prefill and Decode have different compute characteristics, so actions trade SLO achievement, KV-cache transfer cost, \
load balancing, and energy efficiency (which equals to throughput / average_total_power). "

State_form = "The observation is 20 dimensions: \
0-9: a 10 dimensions sentence embedding vector of the input request; \
10: number of input token of the input request; \
11: TTFT(Time To First Token) SLO target in ms for this request; \
12: TPOT(average Time Per Output Token) SLO target in ms for this request; \
13: total number of request in the waiting queue of node group H; \
14: number of request in the waiting queue of node group L; \
15: total number of request waiting to migrate KV Cache from node group H to L; \
16: average usage of GPU memory for KV Cache of node group H; \
17: usage of GPU memory for KV Cache of node group L; \
18: average number of inflight request of node group H; \
19: number of inflight request of node group L; "

Action_form = "The action is a single discrete routing strategy: \
0: HPHD: Run Prefill on the high-performance group H and also run Decode on H. Best when SLO is tight, the output is very short, or when KV-cache transfer overhead must be avoided. \
1: HPLD: Run Prefill on the high-performance group H, then transfer KV cache and the first token to the low-power group L for Decode. Best for TTFT-sensitive requests with long input or balanced long input/output. \
2: LPLD: Run both Prefill and Decode on the low-power group L. Best for loose SLO, energy efficiency, and requests with short input but relatively long output."


class SchedulerPrompt:
    def __init__(self, context: Dict[str, Any], factor: bool = True) -> None:
        self.context = context
        self.factor = factor
        self.role_instruction = Factor_role_instrutor
        self.task_description = Task_description
        self.state_form = State_form
        self.action_form = Action_form

    def get_messages(self) -> List[Dict[str, str]]:
        content = (
            f"{self.role_instruction}\n\n"
            f"TASK:\n{self.task_description}\n\n"
            f"STATE FORM:\n{self.state_form}\n\n"
            f"ACTION FORM:\n{self.action_form}\n\n"
        )
        return [{"role": "user", "content": content}]

    def _safe_exec(self, code: str):
        namespace: Dict[str, Any] = {"np": np}
        exec(code, namespace)
        if "evaluation_func" not in namespace:
            raise ValueError("The generated code does not define evaluation_func")
        return namespace["evaluation_func"]

    def factor_check(self, out_content: Sequence[str]) -> Tuple[bool, int, str, int]:
        error_idx, error_content = -1, ""
        pass_check = True
        factor_num = 0
        example_state = np.asarray([self.context["example_state"]] * 2, dtype=np.float32)
        example_action = np.asarray([[self.context["example_action"]]] * 2, dtype=np.int64)

        for i, content in enumerate(out_content):
            try:
                payload = json.loads(content)
                code = payload["Functions"]
                func = self._safe_exec(code)
                outputs = func(example_state, example_action)
                if not isinstance(outputs, list):
                    raise ValueError("evaluation_func must return a Python list")
                factor_num = len(outputs)
                if (not self.factor) and factor_num != 1:
                    raise ValueError("Direct-reward mode requires exactly one output array")
                if factor_num == 0:
                    raise ValueError("At least one factor must be returned")
                for arr in outputs:
                    arr = np.asarray(arr)
                    if arr.shape != (2, 1):
                        raise ValueError(
                            f"Each returned factor must have shape (batch_size, 1); got {arr.shape}"
                        )
            except Exception as exc:
                pass_check = False
                error_idx = i
                error_content = (
                    "There is an error in your previous answer. "
                    f"Please fix it while keeping the same JSON schema. Error: {exc}"
                )
                break
        return pass_check, error_idx, error_content, factor_num
