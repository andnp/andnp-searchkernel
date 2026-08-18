from datetime import UTC, datetime
from pathlib import Path

import pytest

from searchkernel.api import (
    Record,
    RecordSearchOutcome,
    RecordStatus,
    build_local_record_kernel,
)

pytestmark = pytest.mark.e2e


class _DeterministicEmbeddingProvider:
    model_name = "e2e-test"
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _record(source_id: str, status: RecordStatus) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        workspace_id="e2e",
        source_kind="note",
        source_id=source_id,
        title=f"{source_id} rollout note",
        body="restart rollout search journey",
        status=status,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _index_records(db_path: Path) -> None:
    with build_local_record_kernel(
        db_path,
        embedding_provider=_DeterministicEmbeddingProvider(),
    ) as composition:
        composition.keyword_store.index(
            [
                _record("active", RecordStatus.ACTIVE),
                _record("archived", RecordStatus.ARCHIVED),
            ]
        )


@pytest.mark.asyncio
async def test_public_local_kernel_restarts_and_searches_durable_records(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "records.db"
    _index_records(db_path)

    with build_local_record_kernel(
        db_path,
        embedding_provider=_DeterministicEmbeddingProvider(),
    ) as restarted:
        outcome = await restarted.kernel.search(
            "restart rollout",
            filters={"include_inactive": True},
            limit=5,
        )

    assert isinstance(outcome, RecordSearchOutcome)
    assert {result.record.source_id for result in outcome.results} == {
        "active",
        "archived",
    }


@pytest.mark.asyncio
async def test_public_local_restart_journey_applies_status_filter(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "records.db"
    _index_records(db_path)

    with build_local_record_kernel(
        db_path,
        embedding_provider=_DeterministicEmbeddingProvider(),
    ) as restarted:
        outcome = await restarted.kernel.search(
            "restart rollout",
            filters={"statuses": [RecordStatus.ACTIVE.value]},
            limit=5,
        )

    assert isinstance(outcome, RecordSearchOutcome)
    assert [result.record.source_id for result in outcome.results] == ["active"]
