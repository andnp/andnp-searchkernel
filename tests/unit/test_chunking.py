"""Focused unit coverage for the markdown chunking subsystem."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from searchkernel.chunking import ChunkingStrategy, HeaderBasedChunker, get_chunker
from searchkernel.domain import Record
from searchkernel.ports.chunking_config import ChunkTuningConfig


@dataclass
class Config:
    min_chunk_chars: int = 1
    max_chunk_chars: int = 1_000
    overlap_chars: int = 0
    parent_chunk_min_chars: int = 10_000
    parent_chunk_max_chars: int = 10_000


def make_record(body: str, *, source_id: str = "note:1") -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="note",
        source_id=source_id,
        title="Test note",
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={"file_path": "notes/test.md", "tag": "fixture"},
    )


def chunker(**overrides: int) -> HeaderBasedChunker:
    values = Config().__dict__ | overrides
    return HeaderBasedChunker(Config(**values))


def test_chunking_strategy_is_abstract_and_factory_uses_header_chunker():
    with pytest.raises(TypeError):
        ChunkingStrategy()  # pyright: ignore[reportAbstractUsage]

    config = Config()
    assert isinstance(config, ChunkTuningConfig)
    assert isinstance(get_chunker(config), HeaderBasedChunker)


def test_atx_headings_build_nested_paths_and_serializable_metadata():
    record = make_record("# Guide\nintro\n## Install\nsteps")

    chunks = chunker().chunk_record(record)

    assert [chunk.metadata["header_path"] for chunk in chunks] == [
        "Guide",
        "Guide > Install",
    ]
    assert chunks[1].content == "Install\nContext: Guide\n\nsteps"
    assert chunks[1].metadata["start_pos"] == len("# Guide\nintro\n")
    assert chunks[1].metadata["modified_time"] == "2026-01-01T00:00:00+00:00"
    assert chunks[1].content_hash == chunks[1].compute_content_hash()
    assert json.dumps(chunks[1].metadata)


def test_setext_headings_are_extracted_with_their_levels():
    record = make_record("Guide\n=====\nintro\n\nInstall\n-------\nsteps")

    chunks = chunker().chunk_record(record)

    assert [chunk.metadata["header_path"] for chunk in chunks] == [
        "Guide",
        "Guide > Install",
    ]
    assert [chunk.content for chunk in chunks] == [
        "Guide\n\nintro",
        "Install\nContext: Guide\n\nsteps",
    ]


def test_atx_heading_levels_extend_nested_paths():
    body = "# A\n## B\n### C\n#### D\n##### E\n###### F\nbody"

    chunks = chunker().chunk_record(make_record(body))

    assert [chunk.metadata["header_path"] for chunk in chunks] == [
        "A",
        "A > B",
        "A > B > C",
        "A > B > C > D",
        "A > B > C > D > E",
        "A > B > C > D > E > F",
    ]


def test_header_offsets_use_characters_after_unicode_prefix():
    body = "Préface 🎉\n# Café\nélève"
    chunks = chunker().chunk_record(make_record(body))

    assert chunks[0].metadata["start_pos"] == len("Préface 🎉\n")
    assert chunks[0].metadata["end_pos"] == len(body)
    assert body[chunks[0].metadata["start_pos"] :] == "# Café\nélève"


def test_many_headers_keep_paths_and_offsets_linear_with_multibyte_content():
    sections = ["# Начало 🎉"]
    for index in range(1, 101):
        sections.extend([f"## Раздел {index} café", f"текст {index} 🧪"])
    body = "\n".join(sections)

    chunks = chunker().chunk_record(make_record(body))

    assert len(chunks) == 101
    assert chunks[0].metadata["header_path"] == "Начало 🎉"
    assert chunks[1].metadata["header_path"] == "Начало 🎉 > Раздел 1 café"
    assert chunks[-1].metadata["header_path"] == "Начало 🎉 > Раздел 100 café"
    assert [body[chunk.metadata["start_pos"] :].split("\n", 1)[0] for chunk in chunks] == [
        "# Начало 🎉",
        *[f"## Раздел {index} café" for index in range(1, 101)],
    ]


def test_header_chunk_output_preserves_expected_content_and_hashes():
    record = make_record("# Guide 🎉\nintro\n## Install 🧪\nsteps")

    chunks = chunker().chunk_record(record)

    assert [
        (chunk.content, chunk.metadata["header_path"], chunk.content_hash)
        for chunk in chunks
    ] == [
        (
            "Guide 🎉\n\nintro",
            "Guide 🎉",
            "6ce23a5ac23bc39869c084d250cd3134751b9c5ebbe0d59e31518561896667c2",
        ),
        (
            "Install 🧪\nContext: Guide 🎉\n\nsteps",
            "Guide 🎉 > Install 🧪",
            "1b373dd4cd63c20d449b11d995e641f734de30b1dc26000a4eb8f2dd067ba92a",
        ),
    ]


def test_plain_text_fallback_keeps_short_content_and_metadata():
    record = make_record("plain text")

    chunks = chunker().chunk_record(record)

    assert len(chunks) == 1
    assert chunks[0].content == "plain text"
    assert chunks[0].metadata["header_path"] == ""
    assert chunks[0].metadata["start_pos"] == 0
    assert chunks[0].metadata["end_pos"] == len(record.body)


def test_plain_text_fallback_splits_paragraphs_at_maximum_size():
    record = make_record("one\n\ntwo\n\nthree")

    chunks = chunker(max_chunk_chars=7).chunk_record(record)

    assert [chunk.content for chunk in chunks] == ["one", "two", "three"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_plain_text_fallback_merges_paragraphs_that_fit():
    chunks = chunker(max_chunk_chars=10).chunk_record(make_record("one\n\ntwo"))

    assert [chunk.content for chunk in chunks] == ["one\n\ntwo"]


def test_small_chunks_merge_without_duplicate_structured_headers():
    record = make_record("# Guide\nx\n## Install\nlong enough")

    chunks = chunker(min_chunk_chars=30).chunk_record(record)

    assert len(chunks) == 1
    assert chunks[0].content == "Guide\n\nx\n\nInstall\nContext: Guide\n\nlong enough"
    assert chunks[0].metadata["header_path"] == "Guide > Install"


def test_small_context_only_chunk_is_replaced_by_nested_child():
    record = make_record("# Guide\n## Install\nsteps")

    chunks = chunker(min_chunk_chars=30).chunk_record(record)

    assert len(chunks) == 1
    assert chunks[0].content == "Install\nContext: Guide\n\nsteps"
    assert chunks[0].metadata["header_path"] == "Guide > Install"


def test_large_chunks_split_paragraphs_into_subchunks():
    record = make_record("# Guide\n\nfirst paragraph\n\nsecond paragraph\n\nthird paragraph")

    chunks = chunker(max_chunk_chars=28).chunk_record(record)

    assert [chunk.chunk_id for chunk in chunks] == [
        "note:1_chunk_0_sub_0",
        "note:1_chunk_0_sub_1",
        "note:1_chunk_0_sub_2",
    ]
    assert all(len(chunk.content) <= 28 for chunk in chunks)
    assert all(chunk.metadata["header_path"] == "Guide" for chunk in chunks)


def test_overlap_applies_only_between_sibling_subchunks():
    record = make_record("# Guide\n\nfirst paragraph\n\nsecond paragraph\n\nthird paragraph")

    chunks = chunker(max_chunk_chars=28, overlap_chars=5).chunk_record(record)

    assert "[...graph]" in chunks[1].content
    assert "[...graph]" in chunks[2].content


def test_zero_overlap_does_not_change_split_content():
    record = make_record("# Guide\n\nfirst paragraph\n\nsecond paragraph")

    chunks = chunker(max_chunk_chars=24, overlap_chars=0).chunk_record(record)

    assert all("[..." not in chunk.content for chunk in chunks)


def test_parent_chunks_are_created_and_children_reference_them():
    record = make_record("# One\nfirst\n# Two\nsecond")

    chunks = chunker(
        parent_chunk_min_chars=10,
        parent_chunk_max_chars=40,
    ).chunk_record(record)

    parents = [chunk for chunk in chunks if "_parent_" in chunk.chunk_id]
    children = [chunk for chunk in chunks if "_parent_" not in chunk.chunk_id]
    assert len(parents) == 1
    assert len(children) == 2
    assert parents[0].content == "One\n\nfirst\n\nTwo\n\nsecond"
    assert all(
        child.metadata["parent_chunk_id"] == parents[0].chunk_id for child in children
    )
    assert parents[0].metadata["header_path"] == "One"


def test_parent_minimum_can_leave_chunks_without_parents():
    record = make_record("# One\nfirst")

    chunks = chunker(parent_chunk_min_chars=100).chunk_record(record)

    assert len(chunks) == 1
    assert "_parent_" not in chunks[0].chunk_id
    assert "parent_chunk_id" not in chunks[0].metadata


def test_parent_maximum_flushes_multiple_parent_groups():
    record = make_record("# One\naaa\n# Two\nbbb\n# Three\nccc")

    chunks = chunker(parent_chunk_min_chars=1, parent_chunk_max_chars=15).chunk_record(
        record
    )

    parents = [chunk for chunk in chunks if "_parent_" in chunk.chunk_id]
    children = [chunk for chunk in chunks if "_parent_" not in chunk.chunk_id]
    assert [parent.content for parent in parents] == [
        "One\n\naaa",
        "Two\n\nbbb",
        "Three\n\nccc",
    ]
    assert [child.metadata["parent_chunk_id"] for child in children] == [
        parent.chunk_id for parent in parents
    ]


def test_header_path_helpers_handle_shared_and_empty_paths():
    strategy = chunker()

    assert strategy._compose_chunk_content("", " body ") == "body"
    assert strategy._combine_header_paths("Guide > Install", "Guide > Usage") == (
        "Guide > Install / Usage"
    )
    assert strategy._combine_header_paths("", "") == ""
    assert strategy._header_path_extends("Guide", "Guide > Install")
    assert not strategy._header_path_extends("Guide > Install", "Guide")


def test_chunk_metadata_preserves_input_values_while_normalizing_datetime():
    record = make_record("body")
    chunk = chunker()._build_chunk(
        chunk_id="chunk",
        record_id=record.source_id,
        content="body",
        metadata={"custom": {"nested": True}},
        chunk_index=0,
        header_path="Guide",
        start_pos=2,
        end_pos=6,
        file_path="notes/test.md",
        modified_time=record.updated_at,
        parent_chunk_id="parent",
    )

    assert chunk.metadata == {
        "custom": {"nested": True},
        "header_path": "Guide",
        "start_pos": 2,
        "end_pos": 6,
        "file_path": "notes/test.md",
        "modified_time": "2026-01-01T00:00:00+00:00",
        "parent_chunk_id": "parent",
    }
