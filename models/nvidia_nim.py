"""NVIDIA NIM adapter using its OpenAI-compatible chat-completions API."""

from __future__ import annotations

import os
from typing import Optional

from models.base import LocalOpenAICompatibleAdapter


NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_NEMOTRON_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


class NvidiaNimAdapter(LocalOpenAICompatibleAdapter):
    """Use NVIDIA NIM without exposing its API key to the browser."""

    provider_label = "NVIDIA NIM"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = NVIDIA_NEMOTRON_MODEL,
        timeout: int = 90,
        max_tokens: int = 128,
    ) -> None:
        key = (api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "Set NVIDIA_API_KEY in the repo .env file before enabling NVIDIA NIM."
            )
        super().__init__(
            base_url=NVIDIA_NIM_BASE_URL,
            model_name=model_name,
            api_key=key,
            temperature=0.4,
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "reasoning_budget": 0,
            },
        )
