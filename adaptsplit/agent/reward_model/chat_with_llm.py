from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptsplit.agent.reward_model.prompt_template import SchedulerPrompt


@dataclass
class LLMCallConfig:
    model: str = "gpt-4.1"
    temperature: float = 0.2
    max_retries: int = 5
    n_candidates: int = 5
    factor: bool = True


class OpenAIResponsesClient:
    """Thin wrapper around the official OpenAI Python SDK.

    The code intentionally keeps the dependency localized so the rest of the training code can be
    imported without requiring the OpenAI package.
    """

    def __init__(self, model: str, temperature: float = 0.2) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError(
                "The `openai` package is required for chat_with_gpt.py. Install it with `pip install openai`."
            ) from exc

        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
        )
        self.model = model
        self.temperature = temperature

    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=list(messages),
            # temperature=self.temperature,
        )
        text = getattr(response, "output_text", None)
        if text is None:
            # Fall back to the raw model response structure.
            text = str(response)
        return text


class JSONRepairError(RuntimeError):
    pass


def _ensure_json_payload(text: str) -> str:
    text = text.strip()
    # if text.startswith("```"):
    #     text = text.strip("`")
    #     if text.startswith("json"):
    #         text = text[4:].strip()
    payload = json.loads(text)
    for required_key in ["Understand", "Analyze", "Functions"]:
        if required_key not in payload:
            raise JSONRepairError(f"Missing key: {required_key}")
    return json.dumps(payload, ensure_ascii=False)


def _build_summary_messages(base_messages: Sequence[Dict[str, str]], candidate_responses: Sequence[str]) -> List[Dict[str, str]]:
    messages = list(base_messages)
    messages.append(
        {
            "role": "user",
            "content": (
                "Below are multiple candidate JSON responses that each contain a latent reward encoding function. "
                "Please summarize them into one improved response with the exact same JSON schema and only one final function.\n\n"
                + "\n\n".join(
                    [f"Candidate {i + 1}:\n{resp}" for i, resp in enumerate(candidate_responses)]
                )
            ),
        }
    )
    return messages


def _build_repair_messages(
    base_messages: Sequence[Dict[str, str]],
    previous_response: str,
    error_message: str,
) -> List[Dict[str, str]]:
    messages = list(base_messages)
    messages.append(
        {
            "role": "user",
            "content": (
                "Your previous answer was not executable or did not satisfy the schema constraints. "
                "Please return a repaired answer with the same JSON schema.\n\n"
                f"Previous answer:\n{previous_response}\n\n"
                f"Verifier feedback:\n{error_message}"
            ),
        }
    )
    return messages


def _save_outputs(save_dir: str, response_id: int, responses: Sequence[str], dialog: Sequence[Dict[str, Any]], factor_num: int) -> None:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f"response_{response_id}.npy", np.asarray(list(responses), dtype=object))
    np.save(path / f"factor_num_{response_id}.npy", np.asarray(factor_num, dtype=np.int64))
    with open(path / f"dialog_{response_id}.json", "w", encoding="utf-8") as f:
        json.dump(list(dialog), f, ensure_ascii=False, indent=2)


def callgpt(
    env_context: Dict[str, Any],
    save_dir: str,
    response_id: int = 0,
    config: Optional[LLMCallConfig] = None,
) -> Tuple[List[str], int]:
    """Generate and verify an LLM latent reward function.

    This mirrors the LaRe logic: environment prompting, self-prompting over multiple candidates,
    and pre-verification with execution feedback.
    """

    cfg = config or LLMCallConfig()
    prompt = SchedulerPrompt(context=env_context, factor=cfg.factor)
    base_messages = prompt.get_messages()
    llm = OpenAIResponsesClient(model=cfg.model, temperature=cfg.temperature)

    dialog: List[Dict[str, Any]] = [{"stage": "base_messages", "messages": base_messages}]

    candidate_responses: List[str] = []
    for idx in range(cfg.n_candidates):
        raw = llm.generate(base_messages)
        print(f"[callgpt] candidate_{idx} response generated.")
        normalized = _ensure_json_payload(raw)
        candidate_responses.append(normalized)
        dialog.append({"stage": f"candidate_{idx}", "response": json.loads(normalized)})

    if len(candidate_responses) == 1:
        merged = candidate_responses[0]
    else:
        summary_messages = _build_summary_messages(base_messages, candidate_responses)
        raw = llm.generate(summary_messages)
        print(f"[callgpt] summary response generated.")
        merged = _ensure_json_payload(raw)
        dialog.append({"stage": "self_prompting_summary", "messages": summary_messages, "response": json.loads(merged)})

    verified_responses = [merged]
    factor_num = 0
    for repair_idx in range(cfg.max_retries):
        pass_check, error_idx, error_msg, factor_num = prompt.factor_check(verified_responses)
        if pass_check:
            _save_outputs(save_dir=save_dir, response_id=response_id, responses=verified_responses, dialog=dialog, factor_num=factor_num)
            print(f"[callgpt] function check passed.")
            return list(verified_responses), factor_num

        repair_messages = _build_repair_messages(base_messages, verified_responses[error_idx], error_msg)
        raw = llm.generate(repair_messages)
        print(f"[callgpt] repair_{repair_idx} response generated.")
        repaired = _ensure_json_payload(raw)
        verified_responses[error_idx] = repaired
        dialog.append(
            {
                "stage": f"repair_{repair_idx}",
                "messages": repair_messages,
                "response": json.loads(repaired),
                "error_message": error_msg,
            }
        )

    raise RuntimeError(
        f"Failed to obtain an executable latent reward function after {cfg.max_retries} repair attempts."
    )


if __name__ == "__main__":
    env_context = {
        "example_state": [0.0] * 20,
        "example_action": 1,
        "state_dim": 20,
        "action_dim": 3,
    }
    save_dir = Path("/home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/reward_model/generated")
    response_id = 0
    config = LLMCallConfig(
        model="qwen-plus"
    )
    response_file = save_dir / f"response_{response_id}.npy"
    factor_num_file = save_dir / f"factor_num_{response_id}.npy"

    # Testing
    save_dir.mkdir(exist_ok=True)
    if not response_file.exists():
        callgpt(
            env_context,
            save_dir,
            response_id,
            config
        )
        print("latent reward function generated.")
    
    responses = np.load(response_file, allow_pickle=True)
    factor_num = np.load(factor_num_file, allow_pickle=True)
    code = json.loads(str(responses[0]))["Functions"]
    namespace = {"np": np}
    exec(code, namespace)
    function = namespace["evaluation_func"]
    print("latent reward function loaded.")

    test_state = [1.0] * 20
    test_action = 1
    example_state = np.asarray([test_state] * 2, dtype=np.float32)
    example_action = np.asarray([[test_action]] * 2, dtype=np.int64)
    latent_reward = function(example_state, example_action)
    print(f"test state: {test_state}")
    print(f"test action: {test_action}")
    print(f"latent reward result: {latent_reward}")
    print(f"factor_num: {factor_num}")
