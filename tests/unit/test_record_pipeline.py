"""Focused degradation tests for record-pipeline reranking."""

from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record, RecordHit, RecordIdentity, SearchFilters
from searchkernel.search.record_pipeline import RecordSearchConfig, RecordSearchPipeline


def _record(record_id: str, *, title: str, body: str, indexed_text: str | None = None) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="test",
        source_id=record_id,
        title=title,
        body=body,
        indexed_text=indexed_text,
        created_at=timestamp,
        updated_at=timestamp,
    )


class _KeywordStore:
    def index(self, records: list[Record]) -> None:
        del records

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[RecordHit]:
        del query, k, filters
        return [
            RecordHit(RecordIdentity(None, "test", "empty"), 0.8),
            RecordHit(RecordIdentity(None, "test", "indexed"), 0.7),
            RecordHit(RecordIdentity(None, "test", "body"), 0.6),
        ]


@pytest.mark.asyncio
async def test_reranking_bypasses_empty_text_and_preserves_position_and_score() -> None:
    """
    Empty candidates do not consume model input or lose their raw score.
    """
    records = {
        "empty": _record("empty", title="", body=""),
        "indexed": _record(
            "indexed", title="indexed", body="raw body", indexed_text="search text"
        ),
        "body": _record("body", title="body", body="fallback body"),
    }

    class Reranker:
        model_name = "test-reranker"
        max_input_chars = 12

        def __init__(self) -> None:
            self.documents: list[str] = []

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            del query
            self.documents = documents
            return [0.1, 0.9]

    reranker = Reranker()
    pipeline = RecordSearchPipeline(
        keyword_store=_KeywordStore(),
        hydrator=lambda identity: records.get(identity.source_id),
        reranker=reranker,
        config=RecordSearchConfig(rerank_budget=3),
    )

    outcome = await pipeline.async_search("query", limit=3)

    assert [result.record_id for result in outcome.results] == [
        "empty",
        "body",
        "indexed",
    ]
    assert outcome.results[0].score == pytest.approx(1 / 61)
    assert reranker.documents == ["indexed\nsear", "body\nfallbac"]
    assert "reranker_bypassed_empty_text" in outcome.diagnostics


@pytest.mark.asyncio
async def test_reranking_prefers_record_reranker_over_plain_text() -> None:
    """A RecordReranker receives full Records instead of flattened text."""
    records = {
        "empty": _record("empty", title="", body=""),
        "indexed": _record(
            "indexed", title="indexed", body="raw body", indexed_text="search text"
        ),
        "body": _record("body", title="body", body="fallback body"),
    }

    class IdentityAwareReranker:
        model_name = "test-record-reranker"

        def __init__(self) -> None:
            self.received: list[Record] = []

        def rerank_records(self, query: str, records: list[Record]) -> list[float]:
            del query
            self.received = records
            return [0.1, 0.9]

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            del query, documents
            return [0.1, 0.9]

    reranker = IdentityAwareReranker()
    pipeline = RecordSearchPipeline(
        keyword_store=_KeywordStore(),
        hydrator=lambda identity: records.get(identity.source_id),
        reranker=reranker,
        config=RecordSearchConfig(rerank_budget=3),
    )

    await pipeline.async_search("query", limit=3)

    assert [record.source_id for record in reranker.received] == ["indexed", "body"]
