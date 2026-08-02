"""Unit tests for the Ollama EmbeddingProvider adapter.

These MOCK the HTTP client to avoid requiring a real Ollama daemon.
"""

from unittest import mock

import pytest

from searchkernel.adapters.embedding import OllamaEmbeddingProvider
from searchkernel.ports.embedding import EmbeddingProvider


def _mock_response(json_body: dict) -> mock.Mock:
    response = mock.Mock()
    response.raise_for_status = mock.Mock()
    response.json = mock.Mock(return_value=json_body)
    return response


def test_satisfies_port():
    provider = OllamaEmbeddingProvider("qwen3-embedding:0.6b", dim=1024)
    assert isinstance(provider, EmbeddingProvider)


def test_explicit_dim_skips_show_probe():
    with mock.patch("httpx.Client.post") as mock_post:
        OllamaEmbeddingProvider("qwen3-embedding:0.6b", dim=1024)
        mock_post.assert_not_called()


def test_resolve_dim_from_show_endpoint():
    with mock.patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _mock_response(
            {"model_info": {"qwen3.embedding_length": 1024}}
        )

        provider = OllamaEmbeddingProvider("qwen3-embedding:0.6b")

        assert provider.dim == 1024
        call = mock_post.call_args
        assert call.args[0].endswith("/api/show")
        assert call.kwargs["json"] == {"model": "qwen3-embedding:0.6b"}


def test_resolve_dim_raises_when_missing():
    with mock.patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _mock_response({"model_info": {}})

        with pytest.raises(RuntimeError, match="could not determine"):
            OllamaEmbeddingProvider("qwen3-embedding:0.6b")


def test_embed_returns_vectors_in_order():
    provider = OllamaEmbeddingProvider("qwen3-embedding:0.6b", dim=2)
    texts = ["cats are mammals", "postgresql is a database"]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    with mock.patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _mock_response({"embeddings": vectors})

        result = provider.embed(texts)

        assert result == vectors
        call = mock_post.call_args
        assert call.args[0].endswith("/api/embed")
        assert call.kwargs["json"] == {
            "model": "qwen3-embedding:0.6b",
            "input": texts,
        }


def test_custom_base_url_is_used():
    provider = OllamaEmbeddingProvider(
        "qwen3-embedding:0.6b", dim=1, base_url="http://gpu-box:11434/"
    )

    with mock.patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _mock_response({"embeddings": [[0.1]]})
        provider.embed(["x"])
        call = mock_post.call_args
        assert call.args[0] == "http://gpu-box:11434/api/embed"


def test_embed_rejects_count_mismatch():
    provider = OllamaEmbeddingProvider("qwen3-embedding:0.6b", dim=2)
    with mock.patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _mock_response({"embeddings": [[0.1, 0.2]]})
        with pytest.raises(RuntimeError, match="for 2 inputs"):
            provider.embed(["x", "y"])


def test_embed_rejects_wrong_dimension():
    provider = OllamaEmbeddingProvider("qwen3-embedding:0.6b", dim=2)
    with mock.patch("httpx.Client.post") as mock_post:
        mock_post.return_value = _mock_response({"embeddings": [[0.1]]})
        with pytest.raises(RuntimeError, match="dimension 2"):
            provider.embed(["x"])


def test_context_manager_closes_http_client():
    with mock.patch("httpx.Client") as client_type:
        provider = OllamaEmbeddingProvider("qwen3-embedding:0.6b", dim=2)
        with provider as active_provider:
            assert active_provider is provider

        client_type.return_value.close.assert_called_once()


def test_dimension_probe_failure_closes_http_client():
    with mock.patch("httpx.Client") as client_type:
        client_type.return_value.post.side_effect = RuntimeError("probe failed")

        with pytest.raises(RuntimeError, match="probe failed"):
            OllamaEmbeddingProvider("qwen3-embedding:0.6b")

        client_type.return_value.close.assert_called_once()
