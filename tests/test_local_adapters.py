import json
import pytest
from unittest.mock import MagicMock, patch
from defense.pipeline import DefensivePipeline
from models.base import LocalOpenAICompatibleAdapter, OllamaModelAdapter


def test_ollama_adapter_generation():
    fake_response = json.dumps({"response": "This is a response from Ollama."}).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        adapter = OllamaModelAdapter(model_name="llama3")
        pipeline = DefensivePipeline(model=adapter)
        res = pipeline.run(user_input="Hello from local client")
        assert res.final_output == "This is a response from Ollama."


def test_local_openai_compatible_adapter():
    fake_response = json.dumps({
        "choices": [{"message": {"content": "Response from local vLLM or LM Studio."}}]
    }).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        adapter = LocalOpenAICompatibleAdapter(
            base_url="http://localhost:8000/v1", model_name="mistral"
        )
        pipeline = DefensivePipeline(model=adapter)
        res = pipeline.run(user_input="Tell me about local inference")
        assert res.final_output == "Response from local vLLM or LM Studio."
