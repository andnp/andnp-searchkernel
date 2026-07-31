"""ReRanker module using cross-encoder for candidate re-ranking.

Uses circuit breaker pattern to protect against model loading/inference failures.
This follows the same pattern as VectorIndex._embedding_circuit_breaker to ensure
consistent resilience across model-dependent components.

Circuit breaker behavior:
- Model loading and prediction are protected by circuit breaker
- On repeated failures, circuit opens and rerank returns original rankings
- Circuit automatically recovers after recovery_timeout
"""

import logging
import math
import threading
from collections.abc import Iterable
from typing import Protocol, TypeGuard

from searchkernel.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
)

logger = logging.getLogger(__name__)


def _sigmoid(x: float):
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class ContentProvider(Protocol):
    def __call__(self, chunk_id: str, /) -> str | None: ...


class CrossEncoderModelProtocol(Protocol):
    def predict(self, inputs: list[tuple[str, str]]) -> object: ...


class CrossEncoderProtocol(Protocol):
    def predict(self, inputs: list[tuple[str, str]]) -> Iterable[object]: ...


def _is_iterable_scores(value: object) -> TypeGuard[Iterable[object]]:
    return isinstance(value, Iterable)


class FloatScore(Protocol):
    def __float__(self) -> float: ...


def _is_float_score(value: object) -> TypeGuard[FloatScore]:
    return callable(getattr(value, "__float__", None))


class _CrossEncoderAdapter:
    def __init__(self, model: CrossEncoderModelProtocol):
        self._model = model

    def predict(self, inputs: list[tuple[str, str]]) -> Iterable[object]:
        predictions = self._model.predict(inputs)
        if not _is_iterable_scores(predictions):
            raise TypeError("Cross-encoder returned a non-iterable score result")
        return predictions


class ReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model: CrossEncoderProtocol | None = None
        self._model_lock = threading.Lock()

        # Circuit breaker for model failure protection
        # Follows same pattern as VectorIndex._embedding_circuit_breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=2,
            window_duration=60.0,
        )

    @property
    def circuit_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._circuit_breaker.state

    def _ensure_model_loaded(self) -> None:
        """Load reranker model with thread safety and circuit breaker."""
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return

            def load_model():
                from sentence_transformers import CrossEncoder

                return _CrossEncoderAdapter(CrossEncoder(self._model_name))

            self._model = self._circuit_breaker.call(load_model)

    def _protected_predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run prediction with circuit breaker protection."""
        self._ensure_model_loaded()
        model = self._model
        assert model is not None
        predictions = self._circuit_breaker.call(
            lambda: model.predict(pairs)
        )
        scores: list[float] = []
        for score in predictions:
            if isinstance(score, (str, bytes)) or _is_float_score(score):
                scores.append(float(score))
            else:
                raise TypeError("Cross-encoder returned a non-numeric score")
        return scores

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        content_provider: ContentProvider,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        pairs: list[tuple[str, str, str]] = []
        for chunk_id, _ in candidates:
            content = content_provider(chunk_id)
            if content:
                pairs.append((chunk_id, query, content))

        if not pairs:
            return candidates[:top_n]

        query_content_pairs = [(query, content) for _, _, content in pairs]

        try:
            scores = self._protected_predict(query_content_pairs)
        except CircuitBreakerOpen:
            logger.warning("ReRanker circuit open, returning original rankings")
            return candidates[:top_n]

        chunk_scores = [
            (chunk_id, _sigmoid(float(score)))
            for (chunk_id, _, _), score in zip(pairs, scores, strict=False)
        ]

        missing_ids = {cid for cid, _ in candidates} - {cid for cid, _, _ in pairs}
        for chunk_id, original_score in candidates:
            if chunk_id in missing_ids:
                chunk_scores.append((chunk_id, original_score * 0.5))

        reranked = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
        return reranked[:top_n]
