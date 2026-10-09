from abc import ABC, abstractmethod
import json
from typing import Any, Callable, Dict, List, Optional
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
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.model_name),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
            },
        }
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
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {url}: {str(e)}")


class LocalOpenAICompatibleAdapter(ModelAdapter):
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "default",
        api_key: str = "not-needed",
        temperature: float = 0.7,
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": kwargs.get("model", self.model_name),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.temperature),
        }
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
                    return choices[0].get("message", {}).get("content", "")
                return ""
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Local LLM server at {url}: {str(e)}")
