"""Ollama LLMProvider adapter.

Calls a local Ollama daemon's HTTP API for completions instead of shelling
out to a CLI or loading model weights in-process. Useful when multiple
processes need LLM completions and should share one running model via the
daemon rather than each holding its own copy in RAM.

This is an ADDITIVE port implementation; no other LLM adapters are modified.
"""

from __future__ import annotations

import asyncio
from typing import Any, Self

from searchkernel.domain import Tier


class OllamaLLMProvider:
    """LLMProvider backed by the Ollama HTTP API.

    Requires an Ollama daemon reachable at ``base_url`` with ``model_name``
    already pulled (``ollama pull <model_name>``). Both FAST and SMART tiers
    use the same model; Ollama has no separate tiering of its own.
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
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
            raise RuntimeError("Ollama LLM provider is closed")
        if self._client is None:
            if self._client_lock is None:
                self._client_lock = asyncio.Lock()
            async with self._client_lock:
                if self._client is None:
                    import httpx

                    self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any] | None = None,
        tier: Tier = Tier.FAST,
    ) -> str | dict[str, Any]:
        """Generate a completion via the Ollama chat API.

        Args:
            prompt: The input prompt string.
            response_format: Optional JSON Schema dict for structured output,
                           passed through as Ollama's ``format`` parameter.
            tier: Performance tier (FAST/SMART); both use the same model.

        Returns:
            If response_format is None: the raw completion string.
            If response_format is set: JSON-parsed dict.

        Raises:
            RuntimeError: If the Ollama request fails.
        """
        import httpx

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if response_format is not None:
            payload["format"] = response_format

        try:
            client = await self._get_client()
            response = await client.post(f"{self._base_url}/api/chat", json=payload)
        except httpx.TimeoutException as e:
            raise RuntimeError(
                f"ollama completion timed out after {self._timeout}s for model "
                f"{self.model_name}"
            ) from e
        except httpx.HTTPError as e:
            raise RuntimeError(f"ollama request failed: {e}") from e

        if response.status_code != 200:
            raise RuntimeError(
                f"ollama returned status {response.status_code}: {response.text}"
            )

        body = response.json()
        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise TypeError("ollama response missing string message.content")

        if response_format is not None:
            import json

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"ollama output is not valid JSON: {content[:200]}"
                ) from e
            if not isinstance(parsed, dict):
                raise RuntimeError("ollama structured output must be a JSON object")
            return parsed

        return content
