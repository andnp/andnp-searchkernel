from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from searchkernel.ingestion import (
    EmbeddingInput,
    async_embed_and_upsert,
    async_embed_in_batches,
    embed_and_upsert,
    embed_in_batches,
)
from searchkernel.ingestion.embedding import (
    async_iter_embed_batches,
    iter_embed_batches,
)
from searchkernel.ports.embedding import EmbeddingWrite


@dataclass
class _Provider:
    model_name: str = "test-model"

    def __post_init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]


class _Sink:
    def __init__(self, rejected: set[str] | None = None) -> None:
        self.rows: list[dict[str, object]] = []
        self.rejected = rejected or set()

    def upsert(self, **kwargs: object) -> bool:
        self.rows.append(kwargs)
        return str(kwargs["source_id"]) not in self.rejected


class _BatchSink:
    def __init__(self) -> None:
        self.batches: list[list[EmbeddingWrite]] = []

    def upsert_batch(self, writes: Sequence[EmbeddingWrite]) -> Sequence[bool]:
        self.batches.append(list(writes))
        return [True] * len(writes)


def _inputs(count: int) -> list[EmbeddingInput]:
    return [
        EmbeddingInput(
            source_kind="memory",
            source_id=f"memory-{index}",
            text=f"text-{index}",
            workspace_id="workspace",
            source_updated_at=f"version-{index}",
        )
        for index in range(count)
    ]


def test_embed_and_upsert_batches_inputs_and_preserves_source_metadata() -> None:
    provider = _Provider()
    sink = _Sink()

    result = embed_and_upsert(_inputs(5), provider=provider, sink=sink, batch_size=2)

    assert result.attempted == 5
    assert result.stored == 5
    assert result.rejected == 0
    assert result.batches == 3
    assert provider.calls == [["text-0", "text-1"], ["text-2", "text-3"], ["text-4"]]
    assert sink.rows[0] == {
        "source_kind": "memory",
        "source_id": "memory-0",
        "workspace_id": "workspace",
        "model_name": "test-model",
        "embedding": [6.0],
        "source_updated_at": "version-0",
    }


def test_embed_and_upsert_passes_provider_batches_to_batch_sink() -> None:
    sink = _BatchSink()

    result = embed_and_upsert(_inputs(5), provider=_Provider(), sink=sink, batch_size=2)

    assert result.stored == 5
    assert result.rejected == 0
    assert [[write.source_id for write in batch] for batch in sink.batches] == [
        ["memory-0", "memory-1"],
        ["memory-2", "memory-3"],
        ["memory-4"],
    ]
    assert sink.batches[0][0] == EmbeddingWrite(
        source_kind="memory",
        source_id="memory-0",
        workspace_id="workspace",
        model_name="test-model",
        embedding=[6.0],
        source_updated_at="version-0",
    )


def test_embed_and_upsert_counts_rejected_writes() -> None:
    result = embed_and_upsert(
        _inputs(2),
        provider=_Provider(),
        sink=_Sink(rejected={"memory-1"}),
        batch_size=10,
    )

    assert result.stored == 1
    assert result.rejected == 1


def test_embed_and_upsert_empty_input_does_not_call_adapters() -> None:
    provider = _Provider()
    sink = _Sink()

    result = embed_and_upsert([], provider=provider, sink=sink, batch_size=2)

    assert result.attempted == 0
    assert result.stored == 0
    assert result.rejected == 0
    assert result.batches == 0
    assert provider.calls == []
    assert sink.rows == []


def test_embed_and_upsert_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        embed_and_upsert([], provider=_Provider(), sink=_Sink(), batch_size=0)


def test_embed_and_upsert_rejects_provider_count_mismatch_before_writes() -> None:
    class _ShortProvider(_Provider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return []

    sink = _Sink()
    with pytest.raises(ValueError, match="returned 0 vectors for 2 inputs"):
        embed_and_upsert(_inputs(2), provider=_ShortProvider(), sink=sink, batch_size=2)
    assert sink.rows == []


def test_embed_and_upsert_rejects_late_provider_count_mismatch_after_prior_writes() -> None:
    """A later provider mismatch fails without hiding earlier committed writes.

    The operation validates each bounded provider response before pairing it.
    """
    class _LateShortProvider(_Provider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            if len(self.calls) == 2:
                return []
            return [[float(len(text))] for text in texts]

    sink = _Sink()
    with pytest.raises(ValueError, match="returned 0 vectors for 1 inputs"):
        embed_and_upsert(
            _inputs(3),
            provider=_LateShortProvider(),
            sink=sink,
            batch_size=2,
        )
    assert [row["source_id"] for row in sink.rows] == ["memory-0", "memory-1"]


@pytest.mark.parametrize(
    ("texts", "batch_size", "expected_calls"),
    [
        ([], 2, []),
        (["text-0", "text-1"], 2, [["text-0", "text-1"]]),
        (["text-0", "text-1", "text-2"], 2, [["text-0", "text-1"], ["text-2"]]),
    ],
)
def test_embed_in_batches_handles_empty_exact_and_partial_inputs(
    texts: list[str],
    batch_size: int,
    expected_calls: list[list[str]],
) -> None:
    """The synchronous batch API preserves order across batch boundaries.

    Empty input must avoid provider calls, while partial tails remain intact.
    """
    provider = _Provider()

    vectors = embed_in_batches(texts, provider=provider, batch_size=batch_size)

    assert vectors == [[float(len(text))] for text in texts]
    assert provider.calls == expected_calls


def test_embed_in_batches_rejects_invalid_size_and_provider_mismatch() -> None:
    """Invalid batch sizes and short provider responses fail explicitly.

    A mismatch must not be silently truncated into a shorter result.
    """
    with pytest.raises(ValueError, match="batch_size"):
        embed_in_batches(["text"], provider=_Provider(), batch_size=0)

    class _ShortProvider(_Provider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return []

    with pytest.raises(ValueError, match="returned 0 vectors for 1 inputs"):
        embed_in_batches(["text"], provider=_ShortProvider(), batch_size=1)


def test_iter_embed_batches_is_lazy_and_yields_partial_tail() -> None:
    """The lazy iterator defers source consumption until requested.

    Each yielded list is bounded and the final partial batch is preserved.
    """
    provider = _Provider()
    consumed: list[str] = []

    def texts():
        for index in range(5):
            consumed.append(f"text-{index}")
            yield f"text-{index}"

    batches = iter_embed_batches(texts(), provider=provider, batch_size=2)

    assert consumed == []
    assert next(batches) == [[6.0], [6.0]]
    assert consumed == ["text-0", "text-1"]
    assert list(batches) == [[[6.0], [6.0]], [[6.0]]]


@pytest.mark.asyncio
async def test_async_batch_wrappers_match_synchronous_results() -> None:
    """Async batch wrappers produce the same vectors and acceptance counts.

    The wrappers preserve the synchronous contracts while yielding control.
    """
    provider = _Provider()
    inputs = _inputs(3)

    vectors = await async_embed_in_batches(
        [item.text for item in inputs], provider=provider, batch_size=2
    )
    result = await async_embed_and_upsert(
        inputs, provider=provider, sink=_Sink(), batch_size=2
    )

    assert vectors == [[6.0], [6.0], [6.0]]
    assert result.attempted == result.stored == 3
    assert result.rejected == 0
    assert result.batches == 2


@pytest.mark.asyncio
async def test_async_lazy_iterator_validates_batches_and_keeps_tail() -> None:
    """The async iterator mirrors lazy synchronous batch behavior.

    It yields provider results in bounded lists, including a short final list.
    """
    provider = _Provider()

    batches = async_iter_embed_batches(
        (f"text-{index}" for index in range(3)),
        provider=provider,
        batch_size=2,
    )

    assert [batch async for batch in batches] == [
        [[6.0], [6.0]],
        [[6.0]],
    ]


def test_embed_and_upsert_rejects_batch_sink_acceptance_mismatch() -> None:
    """A batch sink must return one acceptance result per write.

    Returning fewer statuses would make stored and rejected counts ambiguous.
    """
    class _MismatchedBatchSink(_BatchSink):
        def upsert_batch(self, writes: Sequence[EmbeddingWrite]) -> Sequence[bool]:
            self.batches.append(list(writes))
            return [True]

    with pytest.raises(ValueError, match="acceptance results for 2 writes"):
        embed_and_upsert(
            _inputs(2),
            provider=_Provider(),
            sink=_MismatchedBatchSink(),
            batch_size=2,
        )
