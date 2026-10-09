from models.base import (
    CallableModelAdapter,
    LocalOpenAICompatibleAdapter,
    MockModelAdapter,
    ModelAdapter,
    OllamaModelAdapter,
)


def get_model(provider: str = "ollama", **kwargs) -> ModelAdapter:
    """Convenience factory function to instantiate model adapters by provider name."""
    provider_lower = provider.strip().lower()
    if provider_lower in ("ollama", "llama", "llama3", "llama3.1", "local_ollama"):
        return OllamaModelAdapter(**kwargs)
    elif provider_lower in ("openai_compatible", "vllm", "lmstudio", "local_openai"):
        return LocalOpenAICompatibleAdapter(**kwargs)
    elif provider_lower in ("mock", "test"):
        return MockModelAdapter(**kwargs)
    raise ValueError(
        f"Unknown model provider: '{provider}'. Supported: 'ollama', 'openai_compatible', 'mock'"
    )


__all__ = [
    "CallableModelAdapter",
    "LocalOpenAICompatibleAdapter",
    "MockModelAdapter",
    "ModelAdapter",
    "OllamaModelAdapter",
    "get_model",
]
