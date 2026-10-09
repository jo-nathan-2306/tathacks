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


def test_ollama_model_name_normalization():
    assert OllamaModelAdapter.normalize_model_name("llama 3.1b") == "llama3.1"
    assert OllamaModelAdapter.normalize_model_name("llama 3.1") == "llama3.1"
    assert OllamaModelAdapter.normalize_model_name("llama-3.1") == "llama3.1"
    assert OllamaModelAdapter.normalize_model_name("llama3.1b") == "llama3.1"
    assert OllamaModelAdapter.normalize_model_name("llama3.1:latest") == "llama3.1:latest"
    assert OllamaModelAdapter.normalize_model_name("llama3") == "llama3"


def test_ollama_model_name_resolution_with_installed():
    adapter = OllamaModelAdapter(model_name="llama 3.1b", auto_resolve_model=False)
    with patch.object(adapter, "list_installed_models", return_value=["llama3.1:latest"]):
        assert adapter.resolve_model_name("llama 3.1b") == "llama3.1:latest"
        assert adapter.resolve_model_name("llama3.1") == "llama3.1:latest"
        assert adapter.resolve_model_name("llama 3.1") == "llama3.1:latest"


def test_pipeline_from_ollama_factory():
    fake_response = json.dumps({"response": "Factory response"}).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        pipeline = DefensivePipeline.from_ollama(model_name="llama3.1")
        res = pipeline.run(user_input="Test factory")
        assert "Factory response" in res.final_output


def test_live_ollama_llama31_if_server_running():
    adapter = OllamaModelAdapter(model_name="llama 3.1b")
    if not adapter.is_server_running():
        pytest.skip("Local Ollama server is not running")

    installed = adapter.list_installed_models()
    if not any("llama3.1" in m for m in installed):
        pytest.skip("llama3.1 model not installed in local Ollama")

    pipeline = DefensivePipeline(model=adapter)
    res = pipeline.run("What is 10 + 10? Answer with just the number.")
    assert len(res.final_output) > 0
    assert "20" in res.final_output
