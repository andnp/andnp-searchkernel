"""Unit tests for the Ollama LLMProvider adapter.

These MOCK the HTTP client to avoid requiring a real Ollama daemon.
"""

from unittest import mock

import httpx
import pytest

from searchkernel.adapters.llm import OllamaLLMProvider
from searchkernel.domain import Tier
from searchkernel.ports.llm import LLMProvider


def _mock_client(*, status_code: int = 200, json_body: dict, side_effect=None):
    response = mock.Mock(status_code=status_code)
    response.json = mock.Mock(return_value=json_body)
    response.text = str(json_body)

    client = mock.AsyncMock()
    if side_effect is not None:
        client.post.side_effect = side_effect
    else:
        client.post.return_value = response

    return client


def test_satisfies_port():
    provider = OllamaLLMProvider("llama3.2")
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_complete_text_response_mocked():
    provider = OllamaLLMProvider("llama3.2")
    client = _mock_client(
        json_body={"message": {"content": "This is a test completion."}}
    )

    with mock.patch("httpx.AsyncClient", return_value=client):
        result = await provider.complete("Test prompt", tier=Tier.FAST)

    assert result == "This is a test completion."
    call = client.post.call_args
    assert call.args[0].endswith("/api/chat")
    assert call.kwargs["json"]["model"] == "llama3.2"
    assert call.kwargs["json"]["messages"] == [
        {"role": "user", "content": "Test prompt"}
    ]
    assert "format" not in call.kwargs["json"]


@pytest.mark.asyncio
async def test_complete_json_response_mocked():
    provider = OllamaLLMProvider("llama3.2")
    client = _mock_client(
        json_body={"message": {"content": '{"reasoning": "Test", "conclusion": "OK"}'}}
    )

    with mock.patch("httpx.AsyncClient", return_value=client):
        response_format = {"type": "object", "properties": {}}
        result = await provider.complete(
            "Test prompt", response_format=response_format, tier=Tier.SMART
        )

    assert result == {"reasoning": "Test", "conclusion": "OK"}
    assert client.post.call_args.kwargs["json"]["format"] == response_format


@pytest.mark.asyncio
async def test_non_200_status_raises():
    provider = OllamaLLMProvider("llama3.2")
    client = _mock_client(status_code=500, json_body={})

    with (
        mock.patch("httpx.AsyncClient", return_value=client),
        pytest.raises(RuntimeError, match="status 500"),
    ):
        await provider.complete("Test prompt")


@pytest.mark.asyncio
async def test_timeout_raises_runtime_error():
    provider = OllamaLLMProvider("llama3.2")
    client = _mock_client(
        json_body={}, side_effect=httpx.TimeoutException("timed out")
    )

    with (
        mock.patch("httpx.AsyncClient", return_value=client),
        pytest.raises(RuntimeError, match="timed out"),
    ):
        await provider.complete("Test prompt")


@pytest.mark.asyncio
async def test_json_parse_error_on_invalid_json():
    provider = OllamaLLMProvider("llama3.2")
    client = _mock_client(
        json_body={"message": {"content": "This is not valid JSON {"}}
    )

    with (
        mock.patch("httpx.AsyncClient", return_value=client),
        pytest.raises(RuntimeError, match="not valid JSON"),
    ):
        await provider.complete("Test prompt", response_format={"type": "object"})


@pytest.mark.asyncio
async def test_response_requires_string_message_content():
    provider = OllamaLLMProvider("llama3.2")
    client = _mock_client(json_body={"message": {}})

    with (
        mock.patch("httpx.AsyncClient", return_value=client),
        pytest.raises(TypeError, match="message.content"),
    ):
        await provider.complete("Test prompt")


@pytest.mark.asyncio
async def test_tier_fast_and_smart_use_same_model():
    provider = OllamaLLMProvider("llama3.2")
    client = _mock_client(
        json_body={"message": {"content": "Response"}}
    )

    with mock.patch("httpx.AsyncClient", return_value=client):
        await provider.complete("Prompt", tier=Tier.FAST)
        assert client.post.call_args.kwargs["json"]["model"] == "llama3.2"

        await provider.complete("Prompt", tier=Tier.SMART)
        assert client.post.call_args.kwargs["json"]["model"] == "llama3.2"


@pytest.mark.asyncio
async def test_complete_reuses_client_and_closes_it():
    client = _mock_client(json_body={"message": {"content": "Response"}})
    provider = OllamaLLMProvider("llama3.2")

    with mock.patch("httpx.AsyncClient", return_value=client) as constructor:
        await provider.complete("Prompt")
        await provider.complete("Prompt")
        await provider.aclose()

    constructor.assert_called_once_with(timeout=120.0)
    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_context_manager_closes_ollama_client():
    client = _mock_client(json_body={"message": {"content": "Response"}})

    with mock.patch("httpx.AsyncClient", return_value=client):
        async with OllamaLLMProvider("llama3.2") as provider:
            await provider.complete("Prompt")

    client.aclose.assert_awaited_once_with()
