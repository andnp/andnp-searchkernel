# Search Performance and Retrieval Quality Roadmap

Status: 0.18.0 canonical record architecture with batch-first performance
boundaries and v1 federation contracts

Last reviewed: 2026-08-02

## Purpose and current conclusion

This roadmap tracks the work needed to make `andnp-searchkernel` fast,
predictable, and useful across mixed-source search. Version 0.5.0 establishes
one supported local query architecture: source adapters produce `Record`
values and the canonical record pipeline performs candidate retrieval, fusion,
optional graph expansion, policy application, and hydration. Version 0.6.0
adds a versioned federation port, bounded executor, and HTTP source adapter for
systems that own independent search indexes.

Version 0.7.0 adds batch hydration, compiled filters, single-flight caches,
bulk epoch reads, batch graph and parent boundaries, incremental embedding
persistence, bounded federation workers, and opt-in progressive federation
events. These changes improve execution cost and memory behavior without
turning batch-oriented source APIs into per-record public concurrency.

The repository still has synthetic benchmarks, backend tests, and federation
contract tests; it does not yet have a representative, versioned, labeled
corpus or a cross-backend latency study.

## 1. Canonical architecture

The supported local record flow is:

```text
Record source adapter
  -> RecordIngestor
     -> KeywordStore / VectorStore / GraphStore
        -> SearchOrchestrator
           -> RecordSearchPipeline
              -> RecordHit identities
                 -> fusion and policy
                    -> RecordHydrator
                       -> record results + provenance
```

The composition root is `SearchKernel.build`; applications provide stores,
providers, policies, and a hydrator instead of teaching the kernel about a
source's native schema.

The supported federated flow is:

```text
Independent source
  -> SearchSource (v1 contract)
     -> FederationExecutor
           -> bounded worker pool
           -> rank fusion and identity/URI deduplication
              -> hits + source responses + degradation diagnostics

The existing `search()` method remains batch-final. `FederationExecutor.stream()`
and its event aliases are opt-in and emit source updates, explicitly
non-authoritative provisional results, and one authoritative fused result.
```

Federation is intentionally a separate composition boundary. The executor
does not replace source-owned authorization, filtering, lifecycle, or index
management.

Ingestion uses `SemanticRecordIngestor` for keyword and vector stages and
`ResumableSemanticCoordinator` when source iteration and checkpoints are
needed. Embedding writes are consumed and persisted batch by batch, while the
ingestor returns per-record outcomes. The coordinator persists a source
checkpoint only after the complete batch succeeds, so the checkpoint never
claims progress beyond a failed batch.

## 2. Contracts that every optimization must preserve

### 2.1 Identity

The semantic identity of a record is the composite tuple
`(workspace_id, source_kind, source_id)`. Its canonical serialized
`storage_key` is the equality and deduplication key across SQLite and Postgres.
It must survive retrieval, fusion, graph traversal, caches, and hydration.

Never deduplicate or delete by a bare `source_id`. Two sources or workspaces
may legitimately use the same source ID. Deterministic tie-breaking uses the
canonical storage key, so parallel and scalar implementations return the same
order.

The historical read-cache v1 token is not semantic identity. At its retired
compatibility boundary it may identify cache entries, but it must not be used
for cross-backend equality, stale checks, revision tokens, or restore
validation.

### 2.2 Filters

Filters are retrieval constraints, not post-hoc presentation hints. The
standard dimensions are status, workspace, source kind, and eligible candidate
storage keys. A backend may advertise candidate-ID filtering as an optional
capability, but a requested candidate filter must never be silently ignored.

Every backend optimization must prove that filters are applied before result
materialization where the backend supports it, and must preserve the same
eligible identity set as the local reference implementation. Filter changes
must include tests for collisions across workspaces and source kinds.

### 2.3 Ingestion consistency and partial failure

The keyword and semantic stages may commit independently; ingestion is not a
cross-store transaction and does not roll back a store that succeeded before a
different stage failed. Strict mode stops at the first failed record or stage
and the coordinator raises `IngestionError`. Lenient mode keeps successful
records, reports failed records with stage errors, and continues processing
source batches, but a failed batch blocks subsequent checkpoint advancement
until the failed work is retried.

This means a partial index is an expected serving state, not proof that every
source record is searchable. Public readiness exposes `indexing`, `partial`,
and `ready`; availability snapshots separately describe which lexical, graph,
and semantic capabilities can serve queries.

### 2.4 Provenance and search failures

Record results retain `SearchResultProvenance`: contributing strategies,
rank/raw-score details, score adjustments, and parent-expansion identity.
Graph and parent expansion must retain complete identities, not reconstructed
source IDs. Strict mode raises retrieval or hydration failures; lenient mode
returns explicit degradation diagnostics. Batch, cache, and concurrent paths
must not hide either kind of failure.

### 2.5 Query safety and optional dependencies

Search is read-only. It must not mutate source lifecycle, checkpoints, access
state, or supersession state. Core imports must remain usable without FAISS,
Hugging Face, Postgres, tree-sitter, or an LLM provider. Optional adapters
must fail at their own boundary with actionable diagnostics.

### 2.6 Federation safety and comparability

Federated sources must advertise the `v1` contract and the capabilities they
can enforce. A source that cannot honor requested filters is not queried. The
executor bounds concurrency, per-source deadlines, request sizes, response
sizes, and optional reranking text. Source-native scores are retained but are
not assumed comparable; fusion uses local result ranks and deterministic
identity tie-breaking.

Partial availability is part of the response contract. Successful source
responses remain usable when another source times out or fails, and callers
must inspect `partial`, `degradations`, and `warnings` when completeness
matters.

## 3. Compatibility boundaries

The former chunk-oriented query pipeline was retired before the 0.5.0
architecture. Chunks may still be created as an ingestion artifact, but they
are not the public query identity. Local query boundaries return canonical
record outcomes and carry complete identities through `RecordHit`, graph
expansion, and hydration.

Federated query execution is supported in `0.6.0` through the v1
`SearchSource`, `SearchRequest`, `SearchResponse`, and `FederationExecutor`
contracts. Federation carries complete source identities, uses source-local
rank for reciprocal-rank fusion, bounds per-source work, and reports timeout,
unavailability, partial-source, and reranker degradation explicitly. The
repository provides the transport-neutral contracts and HTTP client adapter;
source services remain responsible for their own authorization and filtering.

## 4. Validation baseline and limits

The CI quality gate runs Ruff, Pyrefly, import-linter, and the complete
collected pytest suite with `--strict-markers -m "not slow and not
real_embeddings"`. Pyrefly is the only type checker. The latest local
validation run on 2026-08-07 selected 1,148 tests and deselected 16 slow or
real-embedding tests:

- Python 3.13.14 with the locked environment;
- 1,148 passing tests;
- 12 PostgreSQL pool deprecation warnings;
- `uv lock --check` passing;
- the locked environment and the CI static checks passing.

The CI coverage floor is intentionally 75%. It is a regression floor for the
whole package, not a claim that every optional adapter or failure branch is
equally covered. Raising it requires adding durable tests rather than merely
excluding difficult modules.

The baseline does not establish:

- relevance on a representative labeled corpus;
- p50/p95/p99 latency under production concurrency;
- memory or index-size limits at production scale;
- parity between every local, FAISS, and pgvector configuration;
- real-embedding quality or provider availability.

Pgvector integration tests require Docker or `SEARCHKERNEL_PG_DSN`. Real
embedding tests require locally cached models and remain outside the default
offline gate. Skips caused by unavailable external services must be visible in
CI output and must not be reported as backend coverage.

## 5. Roadmap

### R0 — Keep the quality gate honest (current)

- Run the complete safe test collection, rather than only selected directories.
- Enforce strict marker handling and a 75% package coverage floor.
- Check supported Python versions and installability of selected optional
  extras from the locked dependency graph.
- Keep Pyrefly as the only type checker and use `uv sync --locked`.

### R1 — Complete identity migration

- Keep all store and retrieval seams returning `RecordHit` directly.
- Add cross-backend fixtures proving canonical bytes, storage keys, cache keys,
  stale checks, and deletion behavior agree.
- Verify that graph expansion, parent expansion, and hydration preserve the
  same identity object through every boundary.

### R2 — Prove filter parity

- Define the shared filter vocabulary and adapter capability reporting.
- Add local-vs-Postgres eligible-set parity tests for status, workspace, source,
  and candidate filters.
- Measure filtered ANN under-return and configure safe overfetch from evidence.
- Add query-plan checks showing backend filters run before expensive scoring or
  result materialization where possible.

### R3 — Make provenance an evaluation input

- Add labeled fixtures for keyword, vector, graph, parent, and rerank
  contributions.
- Report degradation reasons and provenance completeness in evaluation output.
- Test deterministic provenance under concurrency, batching, cache hits, and
  strict/lenient failures.

### R4 — Establish representative relevance and performance evidence

- Version a small labeled corpus with query types, source/workspace slices,
  graded relevance, and a reproducible environment fingerprint.
- Report recall@k, nDCG@k, MRR, empty-result rate, source coverage, and latency
  percentiles by slice.
- Add cold/warm, serial/concurrent, and index-build measurements before tuning
  candidate budgets, fusion weights, cache sizes, or ANN thresholds.
- Keep synthetic 1k/10k/100k scale checks as engineering signals, not product
  performance claims.

### R5 — Optimize only after parity is measured

- Measure local FTS filtered retrieval and keep filter work out of Python row
  loops where the schema can support it.
- Compact local vectors and cache normalized norms without changing model
  fingerprint semantics.
- Compare scalar and batch graph, parent, hydration, and version-provider
  output exactly across local and PostgreSQL adapters.
- Add cache epochs for all mutation lanes, including graph changes, with
  invalidation tests.

### R6 — Preserve migration boundaries

- Keep storage migrations isolated from the query contracts.
- Keep documentation aligned with the canonical record-only query path.
- Keep source-specific behavior in adapters and injected policies, not in core
  domain models or a transitional query path.

## 6. Evidence policy

Every performance or relevance change should include the corpus version,
backend, embedding model/dimension, filter set, configuration fingerprint,
Python version, and whether the run was cold or warm. A single synthetic
benchmark or a successful integration test is useful evidence for a narrow
contract; neither is evidence of production readiness by itself.
