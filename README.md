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
from searchkernel.api import SearchKernel

kernel = SearchKernel.build(
    record_hydrator=record_hydrator,
    keyword_store=keyword_store,
    vector_store=vector_store,
    graph_store=graph_store,
    embedding_provider=embedding_provider,
)
```

`SearchKernel.build` creates the canonical `SearchOrchestrator` when record
dependencies are supplied. Callers that already own one may pass
`orchestrator=` instead. Source adapters map native data into `Record`; the
core keeps source-specific fields in opaque metadata and uses injected policy
objects for filtering and ranking.

## Removed legacy paths

The old chunk-oriented query pipeline and legacy federated query execution are
removed from the supported 0.5.0 architecture. Chunks may still be produced
during ingestion, and a few compatibility types remain at migration seams,
but new integrations must use `Record`, `RecordIdentity`, `RecordHit`, the
record store ports, and the canonical record pipeline. Do not build a second
query pipeline around the compatibility surface.

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

The normal quality gate runs the complete collected suite while excluding
`slow` and `real_embeddings` tests. It does not prove production relevance,
latency, memory use, or parity across every backend. The pgvector tests need
Docker or `SEARCHKERNEL_PG_DSN`; real-embedding tests need locally cached
models and are not part of the default offline gate. Use the benchmark
artifacts and [the performance roadmap](docs/search-performance-roadmap.md)
for the measured scope and remaining limits.

## Releases

Merges to `main` with `feat`, `fix`, or breaking Conventional Commits are
released automatically. The release workflow bumps the SemVer version,
updates `pyproject.toml` and `uv.lock`, pushes a `v*` tag, and dispatches the
PyPI publishing workflow. Documentation, chore, and test-only commits do not
create releases.

## License

MIT License. See [LICENSE](LICENSE) for details.
