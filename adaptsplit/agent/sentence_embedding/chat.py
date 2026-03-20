import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from adaptsplit.agent.sentence_embedding.prompting import SentenceEmbeddingPrompt


@dataclass
class LLMCallConfig:
    model: str = "gpt-4.1"
    temperature: float = 0.2
    max_retries: int = 5
    n_candidates: int = 5


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
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.temperature = temperature

    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            # temperature=self.temperature,
        )
        out = completion.choices[0].message.content
        return str(out)


class JSONRepairError(RuntimeError):
    pass


def _ensure_json_payload(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
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
                "Below are multiple candidate JSON responses that each contain a sentence embedding function. "
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


def _save_outputs(save_dir: str, response_id: int, responses: Sequence[str], dialog: Sequence[Dict[str, Any]]) -> None:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / f"response_{response_id}.npy", np.asarray(list(responses), dtype=object))
    with open(path / f"dialog_{response_id}.json", "w", encoding="utf-8") as f:
        json.dump(list(dialog), f, ensure_ascii=False, indent=2)


def callgpt(
    save_dir: str,
    response_id: int = 0,
    config: Optional[LLMCallConfig] = None,
) -> Tuple[List[str], int]:
    cfg = config or LLMCallConfig()
    prompt = SentenceEmbeddingPrompt()
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
    for repair_idx in range(cfg.max_retries):
        pass_check, error_idx, error_msg = prompt.check(verified_responses)
        if pass_check:
            _save_outputs(save_dir=save_dir, response_id=response_id, responses=verified_responses, dialog=dialog)
            print(f"[callgpt] function check passed.")
            return list(verified_responses)

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
        f"Failed to obtain an executable sentence embedding function after {cfg.max_retries} repair attempts."
    )


class SentenceEmbedder:
    def __init__(self, save_dir: str, response_id: int, config: LLMCallConfig = LLMCallConfig()) -> None:
        self.save_dir = Path(save_dir)
        self.response_id = response_id
        self.config = config

        response_file = self.save_dir / f"response_{response_id}.npy"
        self.save_dir.mkdir(exist_ok=True)
        if not response_file.exists():
            callgpt(
                save_dir,
                response_id,
                config
            )
            print("[SentenceEmbedder] Sentence embedding function generated.")
        
        responses = np.load(response_file, allow_pickle=True)
        code = json.loads(str(responses[0]))["Functions"]
        namespace = {}
        exec(code, namespace)
        self.se_function = namespace["sentence_embedding"]
        print("[SentenceEmbedder] Sentence embedding function loaded.")

    def embed(self, prompt: str) -> List[float]:
        return self.se_function(prompt)



if __name__ == "__main__":
    save_dir = Path("/home/austin/repos/AdaptSplit/AdaptSplit/adaptsplit/agent/sentence_embedding/generated")
    response_id = 0
    config = LLMCallConfig(
        model="glm-5",
        n_candidates=3
    )
    response_file = save_dir / f"response_{response_id}.npy"

    # Testing
    save_dir.mkdir(exist_ok=True)
    if not response_file.exists():
        callgpt(
            save_dir,
            response_id,
            config
        )
        print("sentence embedding function generated.")
    
    responses = np.load(response_file, allow_pickle=True)
    code = json.loads(str(responses[0]))["Functions"]
    namespace = {}
    exec(code, namespace)
    se_function = namespace["sentence_embedding"]
    print("sentence embedding function loaded.")

    test_prompt = "Hello, how are you today."
    se = se_function(test_prompt)
    print(f"test_prompt: {test_prompt}")
    print(f"sentence embedding result: {se}")