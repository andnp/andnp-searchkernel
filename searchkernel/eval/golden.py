"""Golden set schema and loader for evaluation.

A golden set is a collection of queries with their ground-truth relevant result IDs.
Used to measure retrieval quality against a benchmark.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GoldenEntry:
    """Single evaluation entry with optional graded and slice metadata."""

    query: str
    """The search query text."""

    relevant_ids: list[str] = field(default_factory=list)
    """Positive-gain result IDs, retained for binary relevance compatibility."""

    relevance: dict[str, float] | None = None
    """Optional graded gains keyed by result ID."""

    query_type: str | None = None
    """Optional query category for evaluation slices."""

    source_kinds: list[str] = field(default_factory=list)
    """Source kinds represented by the expected results."""

    workspace_id: str | None = None
    """Optional workspace identifier for the query."""

    tags: list[str] = field(default_factory=list)
    """Optional evaluation tags."""

    corpus_version: str | None = None
    """Version of the corpus used to create this entry."""

    split: str | None = None
    """Optional dataset split: train, validation, or test."""

    def __post_init__(self) -> None:
        """Normalize mutable inputs and make graded gains authoritative."""
        self.relevant_ids = list(self.relevant_ids)
        if self.relevance is not None:
            self.relevance = dict(self.relevance)
            self.relevant_ids = [
                result_id for result_id, gain in self.relevance.items() if gain > 0
            ]
        self.source_kinds = list(self.source_kinds)
        self.tags = list(self.tags)
        if self.split not in {None, "train", "validation", "test"}:
            raise ValueError("split must be one of train, validation, test, or None")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        data: dict[str, Any] = {
            "query": self.query,
        }
        data["relevant_ids"] = self.relevant_ids
        if self.relevance is not None:
            data["relevance"] = self.relevance
        if self.query_type is not None:
            data["query_type"] = self.query_type
        if self.source_kinds:
            data["source_kinds"] = self.source_kinds
        if self.workspace_id is not None:
            data["workspace_id"] = self.workspace_id
        if self.tags:
            data["tags"] = self.tags
        if self.corpus_version is not None:
            data["corpus_version"] = self.corpus_version
        if self.split is not None:
            data["split"] = self.split
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenEntry":
        """Deserialize from a dictionary."""
        return cls(
            query=data["query"],
            relevant_ids=data.get("relevant_ids") or [],
            relevance=data.get("relevance"),
            query_type=data.get("query_type"),
            source_kinds=data.get("source_kinds", []),
            workspace_id=data.get("workspace_id"),
            tags=data.get("tags", []),
            corpus_version=data.get("corpus_version"),
            split=data.get("split"),
        )


@dataclass
class GoldenSet:
    """A collection of golden entries for evaluation."""

    entries: list[GoldenEntry]
    """List of evaluation entries."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoldenSet":
        """Deserialize from a dictionary."""
        entries = [GoldenEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(entries=entries)

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self.entries)

    def __iter__(self):
        """Iterate over entries."""
        return iter(self.entries)


def load_golden(path: str | Path) -> GoldenSet:
    """Load a golden set from a JSON file.

    Expected JSON format:
    ```json
    {
      "entries": [
        {
          "query": "search query text",
          "relevant_ids": ["result_id_1", "result_id_2"]
        },
        ...
      ]
    }
    ```

    Args:
        path: Path to the JSON file.

    Returns:
        A GoldenSet instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the JSON structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden set file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "entries" not in data:
        raise ValueError("Golden set JSON must contain an 'entries' key with a list value")

    return GoldenSet.from_dict(data)


def save_golden(golden_set: GoldenSet, path: str | Path) -> None:
    """Save a golden set to a JSON file.

    Args:
        golden_set: The GoldenSet to save.
        path: Path to write the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(golden_set.to_dict(), f, indent=2)
