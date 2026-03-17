import json
from typing import Any, Dict, List, Sequence, Tuple

Task_description = "You are an expert in LLM systems, request characterization, and interpretable feature engineering. \
You are studying LLM inference service optimization under heterogeneous SLOs, now you need to extract features from each incoming request.  \
A Python function is needed to maps a prompt string to an interpretable 10-dimensional 1D vector for scheduling-oriented request characterization. \
Requirements: \
- Write a function and provide valid executable Python code as a string. \
- You must decide the meaning of all 10 dimensions. \
- Each dimension must be interpretable, with clear semantic meaning relevant to LLM request scheduling,  \
  such as prompt complexity, structure, task type, reasoning difficulty, expected output burden, or other useful factors. \
- The design should be practical, self-contained, and based on lightweight heuristics or text analysis, without external APIs or unavailable services. \
- The vector should be stable and meaningful for similar prompts. \
- Output JSON only, with nothing else. \
Return strictly in JSON with exactly these keys: \
{ \
    Understand: briefly explain your understanding of the task., \
    Analyze: explain the meaning of each of the 10 dimensions and why they are useful., \
    Functions: 'def sentence_embedding(prompt: str):\n    ...\n    return [a list of exactly 10 numeric values]' \
}"


class SentenceEmbeddingPrompt:
    def __init__(self) -> None:
        self.task_description = Task_description

    def get_messages(self) -> List[Dict[str, str]]:
        content = self.task_description
        return [{"role": "user", "content": content}]

    def _safe_exec(self, code: str):
        namespace = {}
        exec(code, namespace)
        if "sentence_embedding" not in namespace:
            raise ValueError("The generated code does not define sentence_embedding")
        return namespace["sentence_embedding"]

    def check(self, out_content: Sequence[str]) -> Tuple[bool, int, str, int]:
        error_idx, error_content = -1, ""
        pass_check = True
        example_prompt = "You are an expert in LLM systems, request characterization, and interpretable feature engineering."

        for i, content in enumerate(out_content):
            try:
                payload = json.loads(content)
                code = payload["Functions"]
                func = self._safe_exec(code)
                outputs = func(example_prompt)
                if not isinstance(outputs, list):
                    raise ValueError("sentence_embedding must return a Python list")
                embbeding_dim = len(outputs)
                if embbeding_dim != 10:
                    raise ValueError("sentence_embedding must return a list of exactly 10 numeric values")
            except Exception as exc:
                pass_check = False
                error_idx = i
                error_content = (
                    "There is an error in your previous answer. "
                    f"Please fix it while keeping the same JSON schema. Error: {exc}"
                )
                break
        return pass_check, error_idx, error_content
