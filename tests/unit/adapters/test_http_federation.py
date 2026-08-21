import asyncio
import json
from collections.abc import Mapping
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


def _client(response: httpx.Response | None = None) -> mock.AsyncMock:
    client = mock.AsyncMock()
    if response is not None:
        client.post.return_value = response
        client.get.return_value = response
    return client


def _source(
    *,
    timeout_s: float = 5.0,
    verify: bool | str = True,
    max_request_bytes: int = 1_048_576,
    max_response_bytes: int = 4_194_304,
    headers: Mapping[str, str] | None = None,
) -> HttpSearchSource:
    return HttpSearchSource(
        "https://source.example/",
        IDENTITY,
        timeout_s=timeout_s,
        verify=verify,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
        headers=headers,
    )


@pytest.mark.asyncio
async def test_search_posts_json_with_request_trace_and_caller_context() -> None:
    response = SearchResponse(source=IDENTITY)
    client = _client(_response(response.to_dict()))
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

    with mock.patch("httpx.AsyncClient", return_value=client):
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
    client = _client()
    client.get.side_effect = [
        _response(capabilities.to_dict()),
        _response({"status": "ok", "contract_version": "v1"}),
    ]
    source = _source()

    with mock.patch("httpx.AsyncClient", return_value=client):
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
    client = _client()

    async def slow_post(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(0.05)

    client.post.side_effect = slow_post
    source = _source(timeout_s=0.01)

    with (
        mock.patch("httpx.AsyncClient", return_value=client),
        pytest.raises(SearchSourceTimeoutError),
    ):
        await source.search(SearchRequest("query"))


@pytest.mark.asyncio
async def test_http_and_auth_failures_are_explicit() -> None:
    for status_code, error in (
        (401, SearchSourceAuthenticationError),
        (503, SearchSourceHTTPError),
    ):
        client = _client(_response({"error": "unavailable"}, status_code=status_code))
        with (
            mock.patch("httpx.AsyncClient", return_value=client),
            pytest.raises(error),
        ):
            await _source().search(SearchRequest("query"))


@pytest.mark.asyncio
async def test_invalid_schema_is_not_an_empty_success() -> None:
    client = _client(_response({"contract_version": "v1", "hits": []}))

    with (
        mock.patch("httpx.AsyncClient", return_value=client),
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
    client = _client(response)

    with (
        mock.patch("httpx.AsyncClient", return_value=client),
        pytest.raises(SearchSourcePayloadTooLargeError),
    ):
        await _source(max_response_bytes=10).search(SearchRequest("query"))


@pytest.mark.asyncio
async def test_request_size_and_caller_deadline_are_bounded() -> None:
    client = _client(_response(SearchResponse(source=IDENTITY).to_dict()))
    source = _source(max_request_bytes=10)

    with (
        mock.patch("httpx.AsyncClient", return_value=client),
        pytest.raises(SearchSourcePayloadTooLargeError),
    ):
        await source.search(SearchRequest("query"))
    client.post.assert_not_called()

    client = _client()
    source = _source()
    request = SearchRequest(
        "query",
        deadline_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    with (
        mock.patch("httpx.AsyncClient", return_value=client),
        pytest.raises(SearchSourceTimeoutError),
    ):
        await source.search(request)
    client.post.assert_not_called()


def test_http_source_implements_search_source_port() -> None:
    assert isinstance(_source(), SearchSource)


@pytest.mark.asyncio
async def test_source_reuses_client_and_closes_it() -> None:
    client = _client()
    client.get.side_effect = [
        _response({"status": "ok"}),
        _response({"status": "ok"}),
    ]
    source = _source()

    with mock.patch("httpx.AsyncClient", return_value=client) as constructor:
        await source.health()
        await source.health()
        await source.aclose()

    constructor.assert_called_once_with(timeout=5.0, verify=True)
    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("verify", ["/etc/ssl/custom-ca.pem", False])
async def test_source_passes_custom_verify_to_client(verify: bool | str) -> None:
    """Custom CA paths and disabled verification reach the HTTP client."""
    client = _client(_response({"status": "ok"}))

    with mock.patch("httpx.AsyncClient", return_value=client) as constructor:
        await _source(verify=verify).health()

    constructor.assert_called_once_with(timeout=5.0, verify=verify)


@pytest.mark.asyncio
async def test_source_context_manager_closes_client() -> None:
    client = _client(_response({"status": "ok"}))

    with mock.patch("httpx.AsyncClient", return_value=client):
        async with _source() as source:
            await source.health()

    client.aclose.assert_awaited_once_with()
