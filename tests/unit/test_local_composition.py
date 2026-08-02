from datetime import UTC, datetime

import pytest

from searchkernel import Record, RecordSearchOutcome, build_local_record_kernel


class _FakeEmbeddingProvider:
    model_name = "fake"
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_public_local_composition_indexes_and_searches_records(tmp_path) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = Record(
        workspace_id="workspace",
        source_kind="note",
        source_id="record-1",
        title="Local record",
        body="canonical local search content",
        created_at=timestamp,
        updated_at=timestamp,
    )
    composition = build_local_record_kernel(
        tmp_path / "records.db",
        embedding_provider=_FakeEmbeddingProvider(),
    )

    composition.keyword_store.index([record])

    outcome = await composition.kernel.search("canonical", limit=1)

    assert isinstance(outcome, RecordSearchOutcome)
    assert [result.record.storage_key for result in outcome.results] == [
        record.storage_key
    ]
    assert outcome.results[0].record.body == record.body
    hydrated = composition.backend.hydrate_record(record.storage_key)
    assert hydrated is not None
    assert hydrated.storage_key == record.storage_key
    assert hydrated.body == record.body
