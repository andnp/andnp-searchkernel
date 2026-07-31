"""Generic retrieval fields and optional source capabilities."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RetrievalFields:
    """Generic fields extracted by an adapter for retrieval and policy."""

    title: str = ""
    body: str = ""
    uri: str | None = None
    tags: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    parent_id: str | None = None
    source_timestamp: datetime | None = None
    authority: float | None = None
    language: str | None = None
    access_labels: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        field_map: Mapping[str, str] | None = None,
    ) -> RetrievalFields:
        """Extract generic fields from native adapter data."""
        keys = {name: name for name in _FIELD_NAMES}
        keys.update(field_map or {})
        source_timestamp = value.get(keys["source_timestamp"])
        if isinstance(source_timestamp, str):
            source_timestamp = datetime.fromisoformat(source_timestamp)
        if source_timestamp is not None and not isinstance(source_timestamp, datetime):
            raise TypeError("source_timestamp must be a datetime, ISO string, or None")

        authority = value.get(keys["authority"])
        if authority is not None:
            authority = float(authority)
            if not math.isfinite(authority):
                raise ValueError("authority must be finite")

        uri = value.get(keys["uri"])
        if uri is not None:
            uri = str(uri)
        parent_id = value.get(keys["parent_id"])
        if parent_id is not None:
            parent_id = str(parent_id)
        language = value.get(keys["language"])
        if language is not None:
            language = str(language)

        return cls(
            title=str(value.get(keys["title"], "")),
            body=str(value.get(keys["body"], "")),
            uri=uri,
            tags=_string_tuple(value.get(keys["tags"])),
            identifiers=_string_tuple(value.get(keys["identifiers"])),
            parent_id=parent_id,
            source_timestamp=source_timestamp,
            authority=authority,
            language=language,
            access_labels=_string_tuple(value.get(keys["access_labels"])),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for adapter metadata."""
        return {
            "title": self.title,
            "body": self.body,
            "uri": self.uri,
            "tags": list(self.tags),
            "identifiers": list(self.identifiers),
            "parent_id": self.parent_id,
            "source_timestamp": (
                self.source_timestamp.isoformat()
                if self.source_timestamp is not None
                else None
            ),
            "authority": self.authority,
            "language": self.language,
            "access_labels": list(self.access_labels),
        }

    @property
    def embedding_text(self) -> str:
        """Build deterministic text for an adapter-owned embedding request."""
        parts = [self.title.strip(), self.body.strip()]
        parts.extend(value.strip() for value in self.tags if value.strip())
        parts.extend(value.strip() for value in self.identifiers if value.strip())
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """Optional retrieval capabilities advertised by a source adapter."""

    supports_hierarchical_retrieval: bool = False
    supports_embeddings: bool = False
    supports_source_fields: bool = False
    supports_candidate_filtering: bool = False
    has_source_summaries: bool = False

    @property
    def hierarchical_retrieval(self) -> bool:
        """Compatibility alias for capability checks."""
        return self.supports_hierarchical_retrieval


@runtime_checkable
class RetrievalFieldExtractor(Protocol):
    """Adapter contract for extracting generic fields from native values."""

    def extract(self, value: object) -> RetrievalFields:
        ...


def extract_retrieval_fields(
    value: Mapping[str, Any],
    *,
    field_map: Mapping[str, str] | None = None,
) -> RetrievalFields:
    """Extract generic retrieval fields without changing canonical identity."""
    return RetrievalFields.from_mapping(value, field_map=field_map)


_FIELD_NAMES = (
    "title",
    "body",
    "uri",
    "tags",
    "identifiers",
    "parent_id",
    "source_timestamp",
    "authority",
    "language",
    "access_labels",
)


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        raise TypeError("retrieval field collections must be strings or sequences")
    return tuple(str(item) for item in value)
