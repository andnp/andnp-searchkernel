from datetime import UTC, datetime

import pytest

from searchkernel.ports import (
    RetrievalFields,
    SourceCapabilities,
    extract_retrieval_fields,
)


def test_extract_retrieval_fields_maps_native_names_without_identity_fields():
    fields = extract_retrieval_fields(
        {
            "headline": "A title",
            "content": "A body",
            "path": "/notes/a",
            "labels": ["one", "two"],
            "keys": "ABC-1",
            "container": "parent-1",
            "changed": "2026-07-31T12:00:00+00:00",
            "trust": 0.75,
            "locale": "en",
            "visibility": ["private"],
        },
        field_map={
            "title": "headline",
            "body": "content",
            "uri": "path",
            "tags": "labels",
            "identifiers": "keys",
            "parent_id": "container",
            "source_timestamp": "changed",
            "authority": "trust",
            "language": "locale",
            "access_labels": "visibility",
        },
    )

    assert fields.title == "A title"
    assert fields.body == "A body"
    assert fields.uri == "/notes/a"
    assert fields.tags == ("one", "two")
    assert fields.identifiers == ("ABC-1",)
    assert fields.parent_id == "parent-1"
    assert fields.source_timestamp == datetime(2026, 7, 31, 12, tzinfo=UTC)
    assert fields.authority == 0.75
    assert fields.language == "en"
    assert fields.access_labels == ("private",)
    assert fields.embedding_text == "A title\nA body\none\ntwo\nABC-1"


def test_retrieval_fields_round_trip_is_json_compatible():
    fields = RetrievalFields(
        title="Title",
        body="Body",
        source_timestamp=datetime(2026, 7, 31, tzinfo=UTC),
        tags=("tag",),
    )

    restored = RetrievalFields.from_mapping(fields.to_dict())

    assert restored == fields


def test_retrieval_field_validation_rejects_non_finite_authority():
    with pytest.raises(ValueError, match="authority"):
        RetrievalFields.from_mapping({"authority": float("nan")})


def test_source_capabilities_are_opt_in():
    assert not SourceCapabilities().supports_hierarchical_retrieval
    assert SourceCapabilities(supports_hierarchical_retrieval=True).hierarchical_retrieval
