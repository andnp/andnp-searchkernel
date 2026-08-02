# Core concepts

## Records are the public search identity

Applications adapt native content into `Record` values. A record is identified
by the composite tuple:

```text
(workspace_id, source_kind, source_id)
```

The kernel serializes that tuple as `RecordIdentity.storage_key`. The complete
identity is carried through indexing, keyword/vector/graph retrieval, fusion,
caches, and hydration.

Do not deduplicate, update, or delete records by a bare `source_id`. Two
sources or workspaces may legitimately reuse the same source ID.

```python
from searchkernel import Record

record = Record(
    workspace_id="team-a",
    source_kind="notes",
    source_id="welcome",
    title="Welcome",
    body="A source-agnostic record.",
)

print(record.identity)
print(record.storage_key)
```

Source-specific fields belong in record metadata. The kernel uses stable
identity, searchable text, lifecycle status, timestamps, and injected policies;
it does not need to know the source's native schema.

## Ingestion and querying are separate paths

The application owns source discovery and lifecycle. A `ContentSource` yields
records or bounded record batches, while a `RecordIngestor` writes those
records to the configured stores. `SearchKernel` connects those pieces:

```text
ContentSource -> SearchKernel.ingest_source -> RecordIngestor -> stores
stores        -> SearchKernel.search       -> RecordSearchOutcome
```

For a local, durable composition, use
`build_local_record_kernel(...)`. For an application-owned composition, use
`SearchKernel.build(...)` with a `record_hydrator` and the stores/providers it
controls. The [getting-started guide](getting-started.md) shows the smallest
local setup.

Ingestion is asynchronous and checkpoint-aware. A checkpoint is advanced only
after the corresponding batch has succeeded. Strict mode raises
`IngestionError` for a failed batch; lenient mode keeps successful records and
returns per-record failures. These guarantees do not form a cross-store
transaction, so a partial index is an expected state during recovery.

## Search outcomes are explicit

`SearchKernel.search(...)` and `SearchOrchestrator.search(...)` return a
`RecordSearchOutcome` rather than a bare list. Its main fields are:

- `results`: hydrated `RecordSearchResult` values with score and provenance;
- `failures`: stage-specific failures when a backend or hydrator degrades; and
- `diagnostics`: additional cache or execution information.

Check `outcome.degraded` when partial execution matters to your application.
Search is read-only: a query does not change source lifecycle, checkpoints,
access state, or supersession state.

## Stores and providers are injected

The record pipeline can combine the following contracts:

- a keyword store for lexical retrieval;
- a vector store plus an embedding provider for semantic retrieval;
- a graph store for related-record expansion;
- a record hydrator for returning current source data; and
- an optional reranker and search policy.

`build_local_record_kernel(...)` supplies SQLite-backed local stores and accepts
an embedding provider. `SearchKernel.build(...)` and `SearchOrchestrator(...)`
accept application-owned implementations. Optional adapters are kept out of
the core import path.

## Optional integrations

Install only the extra required by the adapter you use:

| Extra | Use |
| --- | --- |
| `faiss` | FAISS-backed local vector search |
| `pgvector` | psycopg2-backed Postgres vector storage |
| `pgvector-psycopg3` | psycopg3-backed Postgres vector storage |
| `huggingface` | Hugging Face embedding and reranking providers |
| `ollama` | Ollama HTTP embedding provider |
| `markdown` | Tree-sitter Markdown chunking during ingestion |

Optional providers can still need an external service, model cache, or
database. Importing `searchkernel` itself requires only the core dependencies.
