import json
from typing import Any, Dict, List, Sequence, Tuple

Task_description = "You are an expert in LLM systems, request characterization, and interpretable feature engineering. \
You are studying LLM inference service optimization under heterogeneous SLOs, and now need to extract features from each incoming request. \
A Python function is needed to analyze the input prompt and map it into a fixed-dimensional 1D vector for scheduling-oriented request characterization. \
Instead of using generic prompt properties such as complexity or structure, the vector should represent the application type of the request. \
For example, requests may belong to categories such as ChatBot, Code Completion, Summarization, Discriminative Tasks or other meaningful application types decided by you. \
Different application types usually imply different TTFT, TPOT SLO preferences. For example, ChatBot requests are usually highly interactive and often require tight TPOT constraints for smooth user experience; \
Code Completion is latency-sensitive and often requires both fast TTFT and low TPOT; Summarization tasks are typically less interactive and may tolerate relatively loose TTFT constraints; other types may have their own typical SLO patterns. \
Discriminative Tasks such as credit verification and data labeling usually have only one single or few tokens as output, which requires very loose TPOT.  \
Requirements: \
- Write a function and provide valid executable Python code as a string. \
- You must decide the number of dimensions of the vector is 5 dimensions. \
- Each dimension must correspond to one interpretable application type category. \
- The function should analyze the input prompt and output a fixed-dimensional numeric vector indicating how strongly the request belongs to each application type. \
- The categories must be designed by you and should be meaningful for LLM request scheduling under heterogeneous SLOs. \
- When analyzing the application types, explicitly explain the typical TTFT, TPOT, or end-to-end latency sensitivity associated with each type, so that the resulting vector is useful for scheduling decisions. \
- Each dimension must be interpretable, with clear semantic meaning, and the full vector should help characterize the request from the perspective of application type and likely SLO preference. \
- The design should be practical, self-contained, and based on lightweight heuristics or text analysis, without external APIs or unavailable services. \
- The vector should be stable and meaningful for similar prompts. \
- Output JSON only, with nothing else. \
Return strictly in JSON with exactly these keys: \
{ \
    Understand: briefly explain your understanding of the task, \
    Analyze: explain the chosen application-type dimensions, why they are useful, why the dimensionality was selected, and the typical TTFT/TPOT/SLO characteristics of each type, \
    Functions: 'def sentence_embedding(prompt: str):\n    ...\n    return [a list of exactly 5 numeric values]' \
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
                if embbeding_dim != 5:
                    raise ValueError("sentence_embedding must return a list of exactly 5 numeric values")
            except Exception as exc:
                pass_check = False
                error_idx = i
                error_content = (
                    "There is an error in your previous answer. "
                    f"Please fix it while keeping the same JSON schema. Error: {exc}"
                )
                break
        return pass_check, error_idx, error_content
