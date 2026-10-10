from abc import ABC, abstractmethod
import json
import re
from typing import Any, Callable, Dict, List, Optional, Union
import urllib.request
import urllib.error


class ModelAdapter(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        pass


class CallableModelAdapter(ModelAdapter):
    def __init__(self, generate_fn: Callable[..., str]) -> None:
        self.generate_fn = generate_fn

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return self.generate_fn(prompt, **kwargs)


class MockModelAdapter(ModelAdapter):
    def __init__(self, responses: Optional[List[str]] = None, default_response: str = "") -> None:
        self.responses = responses or []
        self.default_response = default_response
        self.call_count = 0
        self.prompts_received: List[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.prompts_received.append(prompt)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        self.call_count += 1
        return self.default_response


class OllamaModelAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout: int = 120,
        auto_resolve_model: bool = True,
        num_predict: Optional[int] = None,
        keep_alive: Optional[Union[str, int]] = None,
    ) -> None:
        self.raw_model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.auto_resolve_model = auto_resolve_model
        if num_predict is not None and num_predict <= 0:
            raise ValueError("num_predict must be a positive integer")
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.model_name = (
            self.resolve_model_name(model_name)
            if auto_resolve_model
            else model_name
        )

    @staticmethod
    def normalize_model_name(name: str) -> str:
        """
        Normalizes model name strings (e.g., 'llama 3.1b' -> 'llama3.1',
        'llama-3.1' -> 'llama3.1').
        """
        import re

        if not name:
            return "llama3.1"
        cleaned = name.strip().lower()
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = cleaned.replace("-", "")
        # Remove trailing 'b' following version numbers (e.g. '3.1b' -> '3.1')
        cleaned = re.sub(r"(\d+\.?\d*)b$", r"\1", cleaned)
        return cleaned

    def list_installed_models(self) -> List[str]:
        """Fetches the list of currently installed model tags from Ollama API."""
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(
            url, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                models = res_json.get("models", [])
                if isinstance(models, list):
                    return [
                        m.get("name", "")
                        for m in models
                        if isinstance(m, dict) and "name" in m
                    ]
                return []
        except Exception:
            return []

    def is_server_running(self) -> bool:
        """Checks if the Ollama local HTTP API is reachable and responding."""
        url = f"{self.base_url}/api/tags"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def resolve_model_name(self, candidate_name: str) -> str:
        """
        Resolves model name aliases and tags against installed Ollama models.
        Handles variations like 'llama 3.1b', 'llama3.1', and 'llama3.1:latest'.
        """
        normalized = self.normalize_model_name(candidate_name)
        installed = self.list_installed_models()

        if not installed:
            return normalized

        # 1. Exact match with tag
        for m in installed:
            if candidate_name.lower() == m.lower() or normalized == m.lower():
                return m

        # 2. Tag without :latest or base tag match
        for m in installed:
            base_tag = m.split(":")[0].lower()
            if normalized == base_tag or candidate_name.lower() == base_tag:
                return m

        # 3. Sub-string / fuzzy match
        for m in installed:
            m_norm = self.normalize_model_name(m.split(":")[0])
            if normalized in m_norm or m_norm in normalized:
                return m

        # 4. Fallback if single Llama model is installed
        if len(installed) == 1 and normalized.startswith("llama"):
            return installed[0]

        return normalized

    def generate(self, prompt: str, **kwargs: Any) -> str:
        url = f"{self.base_url}/api/generate"
        target_model = kwargs.get("model", self.model_name)
        if self.auto_resolve_model and target_model != self.model_name:
            target_model = self.resolve_model_name(target_model)

        options = {
            "temperature": kwargs.get("temperature", self.temperature),
        }
        num_predict = kwargs.get("num_predict", self.num_predict)
        user_input = str(kwargs.get("user_input", ""))
        if (
            num_predict is not None
            and num_predict == self.num_predict
            and re.search(
                r"\b(detailed|in depth|in detail|comprehensive|step[- ]by[- ]step|long explanation|elaborate)\b",
                user_input,
                re.IGNORECASE,
            )
        ):
            num_predict = max(num_predict, 192)
        if num_predict is not None:
            options["num_predict"] = num_predict

        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        keep_alive = kwargs.get("keep_alive", self.keep_alive)
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                return res_json.get("response", "")
        except urllib.error.HTTPError as e:
            err_detail = ""
            try:
                raw_body = e.read().decode("utf-8")
                err_data = json.loads(raw_body)
                if isinstance(err_data, dict) and "error" in err_data:
                    err_detail = err_data["error"]
                else:
                    err_detail = raw_body
            except Exception:
                pass
            msg = f"Ollama HTTP error {e.code} for model '{target_model}' at {url}"
            if err_detail:
                msg += f": {err_detail}"
            raise RuntimeError(msg)
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to Ollama at {url}. "
                f"Ensure Ollama is running ('ollama serve'). Error: {str(e)}"
            )


class LocalOpenAICompatibleAdapter(ModelAdapter):
    provider_label = "OpenAI-compatible"

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "default",
        api_key: str = "not-needed",
        temperature: float = 0.7,
        timeout: int = 60,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    def generate(self, prompt: str, **kwargs: Any) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": kwargs.get("model", self.model_name),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
        }
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(self.extra_body)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                choices = res_json.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    if not content and any(
                        message.get(key) for key in ("reasoning_content", "reasoning")
                    ):
                        raise RuntimeError(
                            "Provider returned reasoning without a final answer."
                        )
                    return content if isinstance(content, str) else ""
                return ""
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Local LLM server at {url}: {str(e)}")
