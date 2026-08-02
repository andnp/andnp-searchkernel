# Search Performance and Retrieval Quality Roadmap

Status: 0.5.0 canonical record architecture; performance work is still in
validation

Last reviewed: 2026-08-02

## Purpose and current conclusion

This roadmap tracks the work needed to make `andnp-searchkernel` fast,
predictable, and useful across mixed-source search. Version 0.5.0 establishes
one supported query architecture: source adapters produce `Record` values and
the canonical record pipeline performs candidate retrieval, fusion, optional
graph expansion, policy application, and hydration.

The architecture is ready for focused contract testing, but it is not a
production performance or relevance claim. The repository has synthetic
benchmarks and backend tests; it does not yet have a representative,
versioned, labeled corpus or a cross-backend latency study.

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

Federated callers may fan out across registered sources and fuse their ranked
references, but the record-oriented path is the target for new integrations.
The composition root is `SearchKernel.build`; applications provide stores,
providers, policies, and a hydrator instead of teaching the kernel about a
source's native schema.

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

The legacy read-cache v1 token is not semantic identity. It may remain a
cache/deduplication token at its compatibility boundary, but it must not be
used for cross-backend equality, stale checks, revision tokens, or restore
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

### 2.3 Provenance and failures

Record results retain `SearchResultProvenance`: contributing strategies,
rank/raw-score details, score adjustments, and parent-expansion identity.
Graph and parent expansion must retain complete identities, not reconstructed
source IDs. Strict mode raises retrieval or hydration failures; lenient mode
returns explicit degradation diagnostics. Batch, cache, and concurrent paths
must not hide either kind of failure.

### 2.4 Query safety and optional dependencies

Search is read-only. It must not mutate source lifecycle, checkpoints, access
state, or supersession state. Core imports must remain usable without FAISS,
Hugging Face, Postgres, tree-sitter, or an LLM provider. Optional adapters
must fail at their own boundary with actionable diagnostics.

## 3. Legacy deletion boundary

The old chunk-oriented query pipeline and legacy federated query execution are
removed from the supported 0.5.0 architecture. Chunks may still be created
as an ingestion artifact, but they are not the public query identity.

Compatibility types and adapters that accept tuple-shaped or flat legacy
results may remain temporarily at migration seams. They are not a second
pipeline and must not regain ownership of query execution. The deletion goal
is to remove those seams after all backends and source adapters return
canonical `RecordHit` values and all callers consume complete identities.

## 4. Validation baseline and limits

The safe default gate is the complete collected pytest suite with
`--strict-markers -m "not slow and not real_embeddings"`. The local baseline
on 2026-08-02 is:

- Python 3.13.7 with the locked environment;
- 1,199 passing tests;
- 79% total line coverage;
- `uv lock --check` passing;
- Ruff, Pyrefly, and import-linter available as separate static gates.

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
- Keep Pyrefly as the type checker and use `uv sync --locked`; do not add ty or
  pyright back to the project.

### R1 — Complete identity migration

- Change remaining store and federation seams to return `RecordHit` directly.
- Remove tuple-only compatibility paths once callers have migrated.
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

- Replace local lexical full scans with an indexed implementation while
  preserving filter and identity parity.
- Compact local vectors and cache normalized norms without changing model
  fingerprint semantics.
- Batch graph and hydration operations and compare scalar/batch output exactly.
- Add cache epochs for all mutation lanes, including graph changes, with
  invalidation tests.

### R6 — Retire migration-only surfaces

- Delete legacy flat-result and tuple adapters after R1-R3 acceptance gates
  pass.
- Remove documentation that suggests a legacy query bridge is supported.
- Keep source-specific behavior in adapters and injected policies, not in core
  domain models or a replacement compatibility pipeline.

## 6. Evidence policy

Every performance or relevance change should include the corpus version,
backend, embedding model/dimension, filter set, configuration fingerprint,
Python version, and whether the run was cold or warm. A single synthetic
benchmark or a successful integration test is useful evidence for a narrow
contract; neither is evidence of production readiness by itself.
