"""DiscoverStage: file-discovery ingestion stage.

Lifted from `ApplicationContext.discover_files`, the first phase of the
ingestion path (discover -> chunk -> embed -> index ->
dedup/canonicalize -> re-embed/repair). Pure delegate to
`searchkernel.indexing.discovery`'s `discover_files`/
`discover_files_multi_root` -- same inputs, same outputs, same
single-root-vs-multi-root branch `ApplicationContext.discover_files`
made before extraction.
"""

from __future__ import annotations

from pathlib import Path

from searchkernel.indexing.discovery import discover_files, discover_files_multi_root
from searchkernel.pipeline.stage import SearchContext, replace_state

_DOCUMENTS_PATH_KEY = "documents_path"
_DOCUMENTS_ROOTS_KEY = "documents_roots"
_INCLUDE_PATTERNS_KEY = "include_patterns"
_EXCLUDE_PATTERNS_KEY = "exclude_patterns"
_EXCLUDE_HIDDEN_DIRS_KEY = "exclude_hidden_dirs"
_DISCOVERED_FILES_KEY = "discovered_files"


class DiscoverStage:
    """Discover indexable files under one or more documents roots.

    Expects `context.state` to carry `documents_path` (str | Path,
    the single configured root -- used only when `documents_roots` has
    at most one entry, matching `ApplicationContext.discover_files`'s
    own branch exactly rather than deriving one from the other),
    `documents_roots` (list[str | Path]), `include_patterns`
    (list[str] | None), `exclude_patterns` (list[str] | None) and
    `exclude_hidden_dirs` (bool). Writes
    `context.state["discovered_files"]` (list[str]).
    """

    name = "discover"

    def run(self, context: SearchContext) -> SearchContext:
        documents_path: str | Path = context.state.documents_path
        documents_roots: list[str | Path] = context.state.documents_roots
        include_patterns = context.state.include_patterns
        exclude_patterns = context.state.exclude_patterns
        exclude_hidden_dirs = (
            context.state.exclude_hidden_dirs
            if context.state.exclude_hidden_dirs is not None
            else True
        )

        if len(documents_roots) <= 1:
            discovered = discover_files(
                documents_path=documents_path,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                exclude_hidden_dirs=exclude_hidden_dirs,
            )
        else:
            discovered = discover_files_multi_root(
                [str(root) for root in documents_roots],
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                exclude_hidden_dirs=exclude_hidden_dirs,
            )

        metadata = dict(context.state)
        metadata[_DISCOVERED_FILES_KEY] = discovered
        return replace_state(context, metadata)
