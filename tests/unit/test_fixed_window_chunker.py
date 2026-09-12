"""Focused unit coverage for the plain-text fixed-window chunking strategy."""

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from searchkernel.chunking import get_chunker
from searchkernel.chunking.fixed_window_chunker import FixedWindowChunker
from searchkernel.domain import Record


@dataclass
class Config:
    min_chunk_chars: int = 1
    max_chunk_chars: int = 20
    overlap_chars: int = 0


def make_record(body: str, *, source_id: str = "ticket:1") -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="ticket",
        source_id=source_id,
        title="Test ticket",
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"issue_key": "ENG-1"},
    )


def chunker(**overrides: int) -> FixedWindowChunker:
    values = Config().__dict__ | overrides
    return FixedWindowChunker(Config(**values))


def test_short_body_returns_a_single_chunk():
    chunks = chunker().chunk_record(make_record("short body"))

    assert [chunk.content for chunk in chunks] == ["short body"]
    assert chunks[0].chunk_id == "ticket:1_chunk_0"
    assert chunks[0].chunk_index == 0
    assert chunks[0].content_hash == chunks[0].compute_content_hash()


def test_long_body_splits_into_windows_no_larger_than_max_chars():
    body = "".join(f"word{i:03d} " for i in range(20))  # 160 chars
    chunks = chunker(max_chunk_chars=50, overlap_chars=0).chunk_record(
        make_record(body)
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 50 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == body
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_overlap_repeats_a_trailing_slice_in_the_next_window():
    body = "0123456789" * 4  # 40 chars
    chunks = chunker(max_chunk_chars=20, overlap_chars=5).chunk_record(
        make_record(body)
    )

    assert chunks[0].content[-5:] == chunks[1].content[:5]


def test_trailing_undersized_window_merges_into_the_previous_one():
    # 45 chars at a 20-char window: windows would be [0:20, 20:40, 40:45] --
    # the last, 5 chars, is below min_chunk_chars and must fold into [20:40].
    body = "x" * 45
    chunks = chunker(max_chunk_chars=20, overlap_chars=0, min_chunk_chars=10).chunk_record(
        make_record(body)
    )

    assert [len(chunk.content) for chunk in chunks] == [20, 25]
    assert "".join(chunk.content for chunk in chunks) == body


def test_metadata_carries_char_offsets_and_source_metadata():
    body = "x" * 30
    chunks = chunker(max_chunk_chars=20, overlap_chars=0).chunk_record(
        make_record(body)
    )

    assert chunks[0].metadata["start_pos"] == 0
    assert chunks[0].metadata["end_pos"] == 20
    assert chunks[0].metadata["issue_key"] == "ENG-1"
    assert chunks[1].metadata["start_pos"] == 20
    assert chunks[1].metadata["end_pos"] == 30


@pytest.mark.parametrize(
    ("max_chunk_chars", "overlap_chars"),
    [(0, 0), (-1, 0), (10, 10), (10, 11)],
)
def test_rejects_a_config_that_cannot_make_forward_progress(
    max_chunk_chars: int, overlap_chars: int
):
    with pytest.raises(ValueError):
        FixedWindowChunker(
            Config(max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars)
        )


def test_factory_selects_fixed_window_chunker_for_plain_text():
    strategy = get_chunker(Config(), strategy="plain_text")

    assert isinstance(strategy, FixedWindowChunker)


def test_factory_defaults_to_markdown_chunker():
    from searchkernel.chunking import HeaderBasedChunker

    strategy = get_chunker(Config())

    assert isinstance(strategy, HeaderBasedChunker)
