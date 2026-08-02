# `andnp-searchkernel`

A domain-agnostic search and indexing kernel for hybrid keyword, vector, and
graph retrieval with pluggable embedding, LLM, and reranker providers.

## Status

**0.5.0, pre-alpha: canonical record architecture.** Record ingestion,
candidate retrieval, fusion, graph expansion, and hydration use the
record-oriented contracts. The package is still evolving, and the validation
limits below are important when assessing production readiness.

## Canonical record contracts

Records are identified by the composite tuple
`(workspace_id, source_kind, source_id)`. The canonical `storage_key` is the
serialized form of that tuple and is the identity used by local and Postgres
stores, fusion, graph expansion, caches, and hydration. A bare `source_id` is
not a safe identity because different sources or workspaces may reuse it.

The search path also preserves these invariants:

- `RecordHit` carries the complete `RecordIdentity` across every backend
  boundary.
- Search is read-only. Source lifecycle, checkpoints, access state, and
  supersession state are not changed by a query.
- Status, workspace, source-kind, and candidate-storage-key filters are
  applied as retrieval constraints. Candidate-ID filtering is an explicit
  adapter capability; an adapter must not silently ignore a requested filter.
- Equal scores are ordered by canonical `storage_key`, so scalar, batch, and
  concurrent execution remain deterministic.
- Record results retain `SearchResultProvenance`, including contributing
  strategies, rank/raw-score details, score adjustments, and parent-expansion
  identity where applicable. Degraded-mode failures are reported explicitly.

## Compose canonical search

Compose a local record pipeline from the record hydrator and store ports:

```python
from searchkernel.api import SearchOrchestrator

search = SearchOrchestrator(
    hydrator=record_hydrator,
    keyword_store=keyword_store,
    vector_store=vector_store,
    graph_store=graph_store,
    embedding_provider=embedding_provider,
)
```

`SearchOrchestrator` is the canonical record query boundary. Source adapters
map native data into `Record`; the core keeps source-specific fields in opaque
metadata and uses injected policy objects for filtering and ranking.

## Ingest canonical records

`SemanticRecordIngestor` is the canonical keyword-and-vector ingestor. It
returns an `IngestionReceipt` with one outcome per record and leaves checkpoint
persistence to the caller. `ResumableSemanticCoordinator` adds bounded source
iteration and persists a source checkpoint only after the batch completes
successfully:

- Strict ingestion stops at the first failed record or stage and the
  coordinator raises `IngestionError`; it does not roll back work already
  committed to another store.
- Lenient ingestion retains successful records, reports failed records with
  stage errors, and does not advance the checkpoint for a failed batch.
- Once a batch fails, later source batches may still be processed in lenient
  mode, but their checkpoints remain blocked until the failed work is retried.

These are partial-failure guarantees, not a cross-store transaction. A source
can therefore be queryable with a partial index; readiness snapshots expose
`indexing`, `partial`, and `ready` states.

## Removed legacy paths

The old chunk-oriented query pipeline and legacy federated query execution are
removed from the supported 0.5.0 architecture. Chunks may still be produced
during ingestion, and migration-only compatibility types may remain at their
explicit seams, but neither is a public query path. New integrations must use
`Record`, `RecordIdentity`, `RecordHit`, the record store ports, and the
canonical record pipeline. The remaining compatibility seams are scheduled
for removal after all backends and callers consume canonical record results;
they must not regain ownership of query execution.

## Optional backends

The core package provides source-agnostic domain models, ports, record search,
and evaluation primitives. Install only the integrations an application uses:

```bash
pip install andnp-searchkernel[pgvector,huggingface,markdown]
```

Available extras are `faiss`, `pgvector`, `huggingface`, `ollama`, and
`markdown`. FAISS and pgvector implement the same record-oriented backend
contracts and can be selected independently. Importing the core does not
require any optional provider or backend.

## Validation limits

The CI quality gate runs Ruff, Pyrefly, import-linter, and the complete
collected suite while excluding `slow` and `real_embeddings` tests. Pyrefly is
the only type checker. This gate does not prove production relevance, latency,
memory use, or parity across every backend. The pgvector tests need Docker or
`SEARCHKERNEL_PG_DSN`; real-embedding tests need locally cached models and are
not part of the default offline gate. Use the benchmark artifacts and [the
performance roadmap](docs/search-performance-roadmap.md) for the measured
scope and remaining limits.

## Releases

Merges to `main` with `feat`, `fix`, or breaking Conventional Commits are
released automatically. The release workflow bumps the SemVer version,
updates `pyproject.toml` and `uv.lock`, pushes a `v*` tag, and dispatches the
PyPI publishing workflow. Documentation, chore, and test-only commits do not
create releases.

## License

MIT License. See [LICENSE](LICENSE) for details.
