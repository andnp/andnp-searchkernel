import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest import mock

import httpx
import pytest

from searchkernel.adapters.federation import (
    HttpSearchSource,
    SearchSourceAuthenticationError,
    SearchSourceHTTPError,
    SearchSourcePayloadTooLargeError,
    SearchSourceSchemaError,
    SearchSourceTimeoutError,
)
from searchkernel.ports.federation import (
    CallerAuthorizationContext,
    SearchRequest,
    SearchResponse,
    SearchSource,
    SourceCapabilities,
    SourceIdentity,
)

IDENTITY = SourceIdentity("memory", "remote", "andy")


def _response(payload: object, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("GET", "https://source.example"),
    )


def _client(response: httpx.Response | None = None) -> tuple[mock.MagicMock, mock.AsyncMock]:
    client = mock.AsyncMock()
    if response is not None:
        client.post.return_value = response
        client.get.return_value = response
    context = mock.MagicMock()
    context.__aenter__ = mock.AsyncMock(return_value=client)
    context.__aexit__ = mock.AsyncMock(return_value=False)
    return context, client


def _source(**kwargs: object) -> HttpSearchSource:
    return HttpSearchSource("https://source.example/", IDENTITY, **kwargs)


@pytest.mark.asyncio
async def test_search_posts_json_with_request_trace_and_caller_context() -> None:
    response = SearchResponse(source=IDENTITY)
    context, client = _client(_response(response.to_dict()))
    request = SearchRequest(
        "incident review",
        request_id="request-1",
        trace_id="trace-1",
        caller=CallerAuthorizationContext(
            caller_id="devkit",
            tenant_id="andy",
            scopes=("search:read",),
        ),
    )

    with mock.patch("httpx.AsyncClient", return_value=context):
        result = await _source().search(request)

    assert result == response
    call = client.post.call_args
    assert call.args[0] == "https://source.example/v1/search"
    assert json.loads(call.kwargs["content"]) == request.to_dict()
    assert call.kwargs["headers"]["X-Request-ID"] == "request-1"
    assert call.kwargs["headers"]["X-Trace-ID"] == "trace-1"


@pytest.mark.asyncio
async def test_fetch_capabilities_and_health_validate_v1_payloads() -> None:
    capabilities = SourceCapabilities(
        supports_filters=False,
        max_top_k=25,
    )
    context, client = _client()
    client.get.side_effect = [
        _response(capabilities.to_dict()),
        _response({"status": "ok", "contract_version": "v1"}),
    ]
    source = _source()

    with mock.patch("httpx.AsyncClient", return_value=context):
        assert await source.fetch_capabilities() == capabilities
        assert await source.health() == {
            "status": "ok",
            "contract_version": "v1",
        }

    assert source.capabilities() == capabilities
    assert [call.args[0] for call in client.get.call_args_list] == [
        "https://source.example/v1/search/capabilities",
        "https://source.example/v1/health",
    ]


@pytest.mark.asyncio
async def test_timeout_is_explicit_source_degradation() -> None:
    context, client = _client()

    async def slow_post(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(0.05)

    client.post.side_effect = slow_post
    source = _source(timeout_s=0.01)

    with (
        mock.patch("httpx.AsyncClient", return_value=context),
        pytest.raises(SearchSourceTimeoutError),
    ):
        await source.search(SearchRequest("query"))


@pytest.mark.asyncio
async def test_http_and_auth_failures_are_explicit() -> None:
    for status_code, error in (
        (401, SearchSourceAuthenticationError),
        (503, SearchSourceHTTPError),
    ):
        context, _ = _client(_response({"error": "unavailable"}, status_code=status_code))
        with (
            mock.patch("httpx.AsyncClient", return_value=context),
            pytest.raises(error),
        ):
            await _source().search(SearchRequest("query"))


@pytest.mark.asyncio
async def test_invalid_schema_is_not_an_empty_success() -> None:
    context, _ = _client(_response({"contract_version": "v1", "hits": []}))

    with (
        mock.patch("httpx.AsyncClient", return_value=context),
        pytest.raises(SearchSourceSchemaError, match="invalid search response"),
    ):
        await _source().search(SearchRequest("query"))


@pytest.mark.asyncio
async def test_response_size_is_bounded_before_json_decoding() -> None:
    response = httpx.Response(
        200,
        content=b"x" * 11,
        request=httpx.Request("POST", "https://source.example/v1/search"),
    )
    context, _ = _client(response)

    with (
        mock.patch("httpx.AsyncClient", return_value=context),
        pytest.raises(SearchSourcePayloadTooLargeError),
    ):
        await _source(max_response_bytes=10).search(SearchRequest("query"))


@pytest.mark.asyncio
async def test_request_size_and_caller_deadline_are_bounded() -> None:
    context, client = _client(_response(SearchResponse(source=IDENTITY).to_dict()))
    source = _source(max_request_bytes=10)

    with (
        mock.patch("httpx.AsyncClient", return_value=context),
        pytest.raises(SearchSourcePayloadTooLargeError),
    ):
        await source.search(SearchRequest("query"))
    client.post.assert_not_called()

    context, client = _client()
    source = _source()
    request = SearchRequest(
        "query",
        deadline_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    with (
        mock.patch("httpx.AsyncClient", return_value=context),
        pytest.raises(SearchSourceTimeoutError),
    ):
        await source.search(request)
    client.post.assert_not_called()


def test_http_source_implements_search_source_port() -> None:
    assert isinstance(_source(), SearchSource)
