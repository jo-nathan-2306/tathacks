from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional


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
