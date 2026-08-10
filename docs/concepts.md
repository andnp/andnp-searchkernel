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

Each result preserves its raw `score` and exposes a `normalized_score` in
`[0, 1]`. The normalized value is query-relative to the returned result set,
so it is useful for comparing results within one query but is not a
cross-query probability. The pipeline defaults to reciprocal-rank fusion;
`RecordSearchConfig(fusion_mode="calibrated")` opts into per-lane score
calibration before hybrid fusion.

### Eligibility-aware artifact retrieval

Artifact-shaped queries can use a confident keyword ranking to suppress vector
retrieval. Applications with additional downstream eligibility rules can
provide `RecordSearchPolicy.query_candidate_set_eligible` to allow that
shortcut only when the keyword candidate set can satisfy those rules:

```python
from searchkernel import RecordHit, RecordSearchPolicy, RecordSearchQueryContext

def eligible(
    candidates: list[RecordHit], context: RecordSearchQueryContext
) -> bool:
    return all(candidate.identity.workspace_id == context["workspace_id"]
               for candidate in candidates)

policy = RecordSearchPolicy(query_candidate_set_eligible=eligible)
```

The callback receives raw keyword hits and the read-only query context only
after the keyword ranking meets the artifact confidence threshold. Returning
`True` permits vector suppression; returning `False` keeps vector retrieval
enabled while retaining the normal keyword-bounded candidate set. If no
callback is supplied, the existing artifact shortcut remains unchanged. The
callback should be deterministic and account for all application-owned
eligibility rules that affect whether the requested result limit can be met.

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
