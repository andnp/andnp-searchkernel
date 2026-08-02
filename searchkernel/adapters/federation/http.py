"""Async HTTP/JSON implementation of the federated search source port."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, MutableMapping
from datetime import UTC, datetime
from typing import Any, Literal, Self

from searchkernel.ports.federation import (
    FEDERATION_CONTRACT_VERSION,
    JsonValue,
    SearchRequest,
    SearchResponse,
    SourceCapabilities,
    SourceIdentity,
)

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_MAX_REQUEST_BYTES = 256 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_ERROR_BODY_CHARS = 512


class SearchSourceError(RuntimeError):
    """Base error for failures that degrade an HTTP search source."""


class SearchSourceHTTPError(SearchSourceError):
    """The remote source returned a non-success HTTP status."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class SearchSourceAuthenticationError(SearchSourceHTTPError):
    """The remote source rejected the caller's authentication."""


class SearchSourcePayloadTooLargeError(SearchSourceError):
    """A request or response exceeded the configured transport bound."""


class SearchSourceSchemaError(SearchSourceError):
    """The remote source returned invalid JSON or an invalid v1 payload."""


class SearchSourceTimeoutError(SearchSourceError, TimeoutError):
    """The source request exceeded its configured or caller deadline."""


class SearchSourceTransportError(SearchSourceError):
    """The HTTP transport failed before a response was received."""


class HttpSearchSource:
    """A production HTTP adapter for a v1 federated search source.

    ``fetch_capabilities`` should normally be called during source startup.
    Until then, ``capabilities`` exposes the safe v1 defaults required by the
    synchronous ``SearchSource`` port.
    """

    def __init__(
        self,
        base_url: str,
        source_identity: SourceIdentity,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_request_bytes < 1:
            raise ValueError("max_request_bytes must be positive")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        if not isinstance(source_identity, SourceIdentity):
            raise TypeError("source_identity must be SourceIdentity")

        self.source_identity = source_identity
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._headers = dict(headers or {})
        self._capabilities = SourceCapabilities()
        self._client: Any | None = None
        self._client_lock: asyncio.Lock | None = None
        self._closed = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the shared HTTP client, if it has been created."""
        self._closed = True
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> Any:
        if self._closed:
            raise RuntimeError("HTTP search source is closed")
        if self._client is None:
            if self._client_lock is None:
                self._client_lock = asyncio.Lock()
            async with self._client_lock:
                if self._client is None:
                    import httpx

                    self._client = httpx.AsyncClient(timeout=self._timeout_s)
        return self._client

    def capabilities(self) -> SourceCapabilities:
        """Return the last validated capabilities advertised by the source."""
        return self._capabilities

    async def fetch_capabilities(self) -> SourceCapabilities:
        """Fetch and cache the source's validated v1 capabilities."""
        payload = await self._request_json("GET", "/v1/search/capabilities")
        try:
            capabilities = SourceCapabilities.from_dict(payload)
        except (TypeError, ValueError) as error:
            raise SearchSourceSchemaError(
                f"invalid capabilities response: {error}"
            ) from error
        self._capabilities = capabilities
        return capabilities

    async def health(self) -> Mapping[str, JsonValue]:
        """Fetch the source health document and validate its JSON envelope."""
        payload = await self._request_json("GET", "/v1/health")
        contract_version = payload.get("contract_version")
        if contract_version is not None and contract_version != FEDERATION_CONTRACT_VERSION:
            raise SearchSourceSchemaError(
                "health response has unsupported contract_version: "
                f"{contract_version!r}"
            )
        return payload

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a bounded v1 search request against the remote source."""
        if not isinstance(request, SearchRequest):
            raise TypeError("request must be SearchRequest")
        if request.contract_version != FEDERATION_CONTRACT_VERSION:
            raise SearchSourceSchemaError(
                f"unsupported request contract_version: {request.contract_version}"
            )

        payload = await self._request_json(
            "POST",
            "/v1/search",
            body=request.to_json().encode("utf-8"),
            request_id=request.request_id,
            trace_id=request.trace_id,
            deadline_at=request.deadline_at,
        )
        try:
            response = SearchResponse.from_dict(payload)
        except (TypeError, ValueError) as error:
            raise SearchSourceSchemaError(
                f"invalid search response: {error}"
            ) from error
        if response.source != self.source_identity:
            raise SearchSourceSchemaError(
                "search response source does not match configured source identity"
            )
        return response

    async def _request_json(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        body: bytes | None = None,
        request_id: str = "",
        trace_id: str = "",
        deadline_at: datetime | None = None,
    ) -> dict[str, Any]:
        import httpx

        if body is not None and len(body) > self._max_request_bytes:
            raise SearchSourcePayloadTooLargeError(
                f"request payload exceeds {self._max_request_bytes} bytes"
            )
        timeout_s = self._request_timeout(deadline_at)
        headers: MutableMapping[str, str] = dict(self._headers)
        headers.setdefault("Accept", "application/json")
        if body is not None:
            headers.setdefault("Content-Type", "application/json")
        if request_id:
            headers["X-Request-ID"] = request_id
        if trace_id:
            headers["X-Trace-ID"] = trace_id

        url = f"{self._base_url}{path}"
        try:
            client = await self._get_client()
            operation = (
                client.get(url, headers=headers)
                if method == "GET"
                else client.post(url, content=body, headers=headers)
            )
            response = await asyncio.wait_for(operation, timeout=timeout_s)
        except asyncio.CancelledError:
            raise
        except (TimeoutError, httpx.TimeoutException) as error:
            raise SearchSourceTimeoutError(
                f"{method} {path} timed out after {timeout_s:.3g}s"
            ) from error
        except httpx.HTTPError as error:
            raise SearchSourceTransportError(
                f"{method} {path} transport failed: {error}"
            ) from error

        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = 0
            if declared_length > self._max_response_bytes:
                raise SearchSourcePayloadTooLargeError(
                    f"response payload exceeds {self._max_response_bytes} bytes"
                )
        content = response.content
        if len(content) > self._max_response_bytes:
            raise SearchSourcePayloadTooLargeError(
                f"response payload exceeds {self._max_response_bytes} bytes"
            )
        if response.status_code in (401, 403):
            raise SearchSourceAuthenticationError(
                response.status_code,
                f"{method} {path} authentication failed with status "
                f"{response.status_code}",
            )
        if response.status_code < 200 or response.status_code >= 300:
            detail = content.decode("utf-8", errors="replace")[
                :_MAX_ERROR_BODY_CHARS
            ]
            raise SearchSourceHTTPError(
                response.status_code,
                f"{method} {path} returned status {response.status_code}: {detail}",
            )
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SearchSourceSchemaError(
                f"{method} {path} returned invalid JSON"
            ) from error
        if not isinstance(payload, dict) or any(
            not isinstance(key, str) for key in payload
        ):
            raise SearchSourceSchemaError(
                f"{method} {path} response must be a JSON object"
            )
        return payload

    def _request_timeout(self, deadline_at: datetime | None) -> float:
        timeout_s = self._timeout_s
        if deadline_at is not None:
            if deadline_at.tzinfo is None or deadline_at.utcoffset() is None:
                raise SearchSourceTimeoutError("request deadline must be timezone-aware")
            remaining = (
                deadline_at.astimezone(UTC) - datetime.now(UTC)
            ).total_seconds()
            timeout_s = min(timeout_s, remaining)
        if timeout_s <= 0:
            raise SearchSourceTimeoutError("source request deadline has elapsed")
        return timeout_s
