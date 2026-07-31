"""ChunkStage: record-chunking ingestion stage.

Lifted from `IndexManager`'s three `self._chunker.chunk_record(record)`
call sites (`index_document`, `index_record`, `reconcile_indices`'s move
detection), the second phase of the ingestion path (discover -> chunk ->
embed -> index -> dedup/canonicalize -> re-embed/repair). Pure delegate to
a `ChunkingStrategy` instance -- same input, same output -- parameterized
over the strategy (like `RetrieveStage` over its searchers) since the
concrete chunker varies with `Config.chunking`.
"""

from __future__ import annotations

from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.pipeline.stage import SearchContext, replace_state, require_state

_RECORD_KEY = "record"
_CHUNKS_KEY = "chunks"


class ChunkStage:
    """Chunk a record with a configured `ChunkingStrategy`.

    Expects `context.state["record"]`. Writes the resulting
    `list[Chunk]` to `context.state["chunks"]`.
    """

    name = "chunk"

    def __init__(self, chunker: ChunkingStrategy):
        self._chunker = chunker

    def run(self, context: SearchContext) -> SearchContext:
        record = require_state(context.state.record, "record")
        chunks = self._chunker.chunk_record(record)

        metadata = dict(context.state)
        metadata[_CHUNKS_KEY] = chunks
        return replace_state(context, metadata)
