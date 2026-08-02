"""Registry of named SearchableSources for federation.

Lets callers register federated/local sources once and select a subset by
source_kind at query time (search_anything(sources=["local", "memory"])),
instead of every caller wiring the full source list by hand.
"""

from searchkernel.ports.content_source import SearchableSource


class SourceRegistry:
    """Holds SearchableSources keyed by their source_kind."""

    def __init__(self) -> None:
        self._sources: dict[str, SearchableSource] = {}

    def register(self, source: SearchableSource) -> None:
        """Register a source, rejecting duplicate source kinds."""
        if source.source_kind in self._sources:
            raise ValueError(
                f"Source kind {source.source_kind!r} is already registered"
            )
        self._sources[source.source_kind] = source

    def get(self, source_kind: str) -> SearchableSource | None:
        """Look up a single registered source by source_kind."""
        return self._sources.get(source_kind)

    def select(self, source_kinds: list[str] | None = None) -> list[SearchableSource]:
        """Resolve a list of source_kinds to their registered sources.

        Raises:
            KeyError: If any requested source_kind is not registered.
        """
        if source_kinds is None:
            return list(self._sources.values())
        unknown = [kind for kind in source_kinds if kind not in self._sources]
        if unknown:
            raise KeyError(f"Unknown source kind(s): {unknown!r}")
        return [
            self._sources[kind] for kind in source_kinds
        ]

    def all(self) -> list[SearchableSource]:
        """Return every registered source."""
        return list(self._sources.values())
