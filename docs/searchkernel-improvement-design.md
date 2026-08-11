# SearchKernel improvement design

Status: proposed

Date: 2026-08-11

Scope: `andnp-searchkernel` 0.22.x and the next pre-1.0 development cycle

## Summary

This design turns the library review into an ordered improvement program. The
first priority is to make indexing and evaluation trustworthy at scale. The
second is to remove retrieval-quality cliffs and backend divergence. Only
after those contracts are measured should fusion, candidate budgets, or ANN
parameters be tuned.

The program preserves the current record-oriented architecture:

```text
Record source
  -> ingestion and checkpointing
     -> keyword / vector / graph stores
        -> query router and candidate acquisition
           -> fusion, policy, graph and parent expansion
              -> hydration and provenance
```

The design does not introduce a Rust extension, replace SQLite, make injected
providers library-owned, or collapse backend benchmarks into full-pipeline
quality benchmarks. SQLite FTS5, NumPy, FAISS, and pgvector already provide
native-backed paths; the immediate evidence points to bounded-query design,
contract clarity, and evaluation quality as higher-value work.

## Current evidence

The current repository has strong foundations:

- canonical composite identity through `RecordIdentity.storage_key`;
- local SQLite keyword, vector, and graph stores;
- optional FAISS and pgvector integrations;
- hybrid keyword/vector/graph retrieval with routing and provenance;
- checkpointed ingestion, caches, reranking, federation, and evaluation tools;
- Ruff, Pyrefly, and import-linter all pass;
- the safe test suite passed 1,211 tests with 83.25% coverage during the
  review.

The review also established these constraints and risks:

| Area | Evidence | Design implication |
| --- | --- | --- |
| Vector updates | `LocalRecordBackend.upsert()` skips vector writes when `Record.embedding` is `None`; an old vector remains searchable | Define missing-embedding semantics and test local/Postgres parity |
| Batch scale | `_records_have_vectors()` and `SQLiteEmbeddingCache.get_many()` build unbounded `IN` clauses; 32,767 keys fail with SQLite's variable limit | Every key-list query must be bounded |
| FTS fallback | No-FTS fallback refuses corpora over 10,000 rows | Degrade with complete, observable behavior |
| ANN filters | FAISS and pgvector use bounded overfetch/scan rounds and can under-return after filtering | Make under-return a measured and explicit serving policy |
| Fusion | `fusion_mode="calibrated"` currently uses per-lane min-max normalization; a tied weak lane maps to 1.0 | Separate relative normalization from absolute calibration |
| PostgreSQL lexical search | Search reparses weighted vectors and uses `plainto_tsquery`; SQLite and Postgres semantics differ | Align the contract before optimizing SQL |
| Evaluation | Backend-level FTS evaluation is useful but does not exercise the full pipeline; the scale gate checks evidence presence, not regressions | Maintain layered benchmarks and validate metadata compatibility |
| Test hygiene | The suite emits unclosed SQLite resource warnings; the roadmap reports stale test counts | Close owned resources and keep evidence documentation current |

## Goals

1. Preserve canonical identity, filter, provenance, ownership, and failure-mode
   contracts across all backends.
2. Make indexing correct under updates, partial failures, retries, large
   batches, restarts, and model migration.
3. Eliminate silent empty-result or under-return cliffs where a complete result
   is still possible.
4. Make benchmark artifacts reproducible and resistant to comparing unrelated
   corpora, backends, models, or configurations.
5. Establish representative labeled evidence before selecting retrieval
   algorithms or fixed thresholds.
6. Improve hybrid retrieval quality without making every query pay for every
   optional capability.
7. Keep optional integrations optional and preserve application ownership of
   injected resources.

## Non-goals

- A broad rewrite of the indexing or search architecture.
- A Rust/PyO3 implementation before native-backed paths and database plans are
  profiled at representative scale.
- Treating backend-specific FTS/ANN benchmarks as substitutes for a
  full-pipeline benchmark.
- Making every optional backend a mandatory dependency of the core package.
- Adding speculative compatibility layers without a demonstrated caller.
- Changing source authorization, lifecycle, or federation ownership semantics.

## Invariants

These invariants must remain true throughout the work.

### Identity

The semantic identity of every result is
`(workspace_id, source_kind, source_id)`. Stores, caches, fusion, graph
traversal, parent expansion, hydration, and evaluation must use the canonical
storage key or the complete `RecordIdentity`, never a bare source ID.

### Index consistency

For each `(storage_key, encoder_namespace, dim)`, the persisted vector must
either represent the current indexed record revision or be absent. A record
update must not silently expose an embedding from an older revision.

### Filter correctness

Filters are retrieval constraints. A backend may return fewer than `k` only
when fewer than `k` eligible records exist or when the response explicitly
reports bounded approximation/degradation. It must not silently ignore a
requested candidate, workspace, source, metadata, or status filter.

### Ownership

The library closes only resources it created. Injected database managers,
providers, pools, and source adapters remain application-owned.

### Evidence

Every quality or performance comparison identifies the corpus, split, `k`,
backend, model, vector dimension, routing/fusion configuration, environment,
and collection date. A report from an incompatible setup cannot pass a
baseline comparison.

## Proposed design

### 1. Make vector lifecycle semantics explicit

The `VectorStore.upsert()` protocol currently describes records with populated
embeddings, while the concrete stores accept optional embeddings. This allows a
caller to update record metadata or text and accidentally retain an obsolete
vector.

Adopt the following contract:

- `VectorStore.upsert()` requires an embedding for every record in its input;
- missing embeddings raise a deterministic validation error before mutation;
- keyword-only record updates use `KeywordStore.index()` and do not alter
  vectors;
- explicit model-vector removal uses `delete_for_model()` or an equivalent
  model-scoped operation;
- semantic ingestion materializes the new vector before calling vector upsert;
- vector rows carry a record revision token or content fingerprint so stale
  writes can be detected during migration and restore.

The recommended resolution is to reject missing embeddings at the vector-store
boundary and keep keyword-only record updates separate. If Phase 0 instead
chooses explicit model-vector deletion, the same write-path separation and
parity tests remain required. The first implementation can use the existing
`updated_at` plus a stable content fingerprint. A future schema migration may
add `record_revision` to both local and Postgres vector tables. The vector
namespace and dimension remain part of the key and may not be mixed.

Required tests:

- reject an `upsert()` batch containing a missing embedding without changing
  records or vectors;
- update a record with a new embedding and verify the old result disappears;
- update keyword fields through `index()` and verify the existing vector is
  preserved;
- compare local, FAISS, pgvector, and psycopg3 behavior;
- retry a failed batch and verify no stale vector becomes visible.

### 2. Bound all variable-length persistence queries

Introduce one shared bounded-key utility for SQLite operations. It should:

- deduplicate keys while preserving deterministic order;
- use a conservative parameter budget below the runtime SQLite limit;
- yield chunks for `IN` queries and bulk deletes;
- preserve empty-input fast paths;
- expose the chosen batch size in diagnostics only when tracing is enabled.

Use it in:

- `LocalRecordBackend._records_have_vectors()`;
- `SQLiteEmbeddingCache.get_many()`;
- hydration, deletion, candidate filters, and any future key-list query.

Do not assume the compile-time SQLite variable limit is identical across
platforms. The default should be conservative, with an optional measured limit
for environments that need larger batches. Add a scale test that runs above
the default and above the host's discovered limit.

The operation must remain set-oriented. A loop that performs one SQL query per
key is not an acceptable fallback. Start with the confirmed SQLite failures;
audit PostgreSQL's existing `ANY(array)` paths separately and chunk them only
when driver, packet, or measured statement limits require it. Do not impose
SQLite placeholder assumptions on PostgreSQL without evidence.

### 3. Replace silent FTS fallback cliffs with explicit capabilities

When FTS5 is available, retain the indexed path. When it is unavailable,
provide a bounded streaming scan that:

- scans rows in deterministic storage-key order;
- evaluates filters before expensive token scoring;
- maintains only the top `k` hits using a heap;
- reports scan count, limit, and whether the scan was complete;
- supports corpora larger than 10,000 rows without returning an unexplained
  empty list.

If a deployment requires a hard scan limit, return a degraded outcome or
diagnostic that distinguishes “no match” from “scan budget exhausted.” The
public keyword-store port may continue returning a list for compatibility, but
the record pipeline must preserve the degradation signal.

Improve fallback tokenization in a separate step:

- use Unicode-aware token extraction for Latin and accented text;
- add language-specific handling for scripts without whitespace tokenization;
- retain artifact-query behavior and avoid fuzzy expansion for quoted or
  identifier-like queries;
- remove the fixed first-256-token false-negative cliff, or replace it with an
  indexed token candidate phase that is bounded by document size and query
  terms.

### 4. Make chunk and reranker degradation observable

Chunk retrieval currently groups chunk hits under a parent. If the parent
cannot be hydrated, the matching chunks are dropped. Resolve this in Phase 0.
The recommended default is to omit the unhydrated chunk to preserve the
parent-oriented public result contract, while returning structured degradation
with the missing parent identity. An opt-in chunk-preserving mode may return
the chunk with complete provenance. In either case, choose one documented
behavior:

- return the chunk as a degraded result with complete chunk provenance; or
- omit it but add a structured `parent_unavailable` degradation and include the
  missing parent in the outcome.

Reranking should construct text using title, `indexed_text`, and body fallback.
An empty candidate must not disable reranking for every other candidate. Do not
feed an artificial placeholder into a cross-encoder. Exclude empty candidates
from the reranker batch, retain their pre-rerank score and deterministic
position, and record a structured `reranker_bypassed_empty_text` event. The
retained score is the raw pre-rerank candidate score in the same scale used by
the pre-rerank ordering; it must not be mixed with a normalized lane score.
Truncate assembled text to the reranker's declared token or character budget
and test the boundary.

Fallback and parent degradation should extend the existing
`RecordSearchDiagnostics` and `RecordSearchOutcome` contracts with structured
events and fields such as `scan_complete`, `ann_under_returned`, and
`missing_parent_ids`; do not introduce a parallel `SearchResult` response
model.

Required tests cover parent hydration returning `None`, parent hydration
raising in strict and lenient modes, empty title/body, indexed-text-only
records, and reranker ordering with mixed candidate text quality.

### 5. Separate score normalization, calibration, and ranking fusion

Use distinct names and contracts:

- `normalize_scores`: query-relative min-max or rank normalization; suitable
  only for comparing candidates within one lane and one query;
- `calibrate_scores`: maps a lane's raw score to an estimated probability or
  confidence using parameters learned from labeled data;
- `fuse_reciprocal_rank`: rank-only fusion when score scales are unknown;
- `fuse_calibrated_scores`: combines calibrated scores only when the lane
  calibration fingerprint matches the active configuration.

The first safe change is to rename or document the current min-max behavior so
callers do not mistake it for absolute confidence. Then add a calibration
experiment:

1. collect keyword, vector, graph, and reranker scores with graded labels;
2. fit monotonic Platt/logistic or isotonic calibration per lane and query
   family;
3. evaluate calibration error, nDCG, recall, and empty-result behavior;
4. compare against ordinary RRF and the current relative normalization;
5. persist a calibration version and reject incompatible cached candidates.

No learned model becomes the default until it beats RRF on held-out slices and
does not regress exact-identifier queries.

### 6. Make filtered ANN behavior adaptive and explicit

For FAISS and pgvector, candidate filtering happens after or alongside ANN
candidate generation. Fixed overfetch values cannot provide stable recall for
both broad and highly selective filters.

Add filter-selectivity-aware search evidence:

- broad, 10%, 1%, and 0.1% eligible-set slices;
- exact-vs-ANN recall at each `k`;
- returned count, scan rounds, scan bound, latency, and memory;
- separate results by backend and ANN configuration.

Then implement the smallest validated policy:

- estimate eligible density from indexed counts where cheap;
- increase overfetch within a configured ceiling for selective filters;
- use iterative HNSW scan where supported;
- fall back to exact search when the filter is highly selective and the exact
  path is within its resource budget;
- otherwise return an explicit under-return/degraded diagnostic.

Persist separate fingerprints with FAISS artifacts. The index-build
fingerprint covers model namespace, dimension, metric, search strategy, HNSW
`M`, and construction parameters. The query-policy fingerprint covers
`efSearch`, overfetch, scan rounds, and scan bounds. Query-policy changes must
invalidate query-result evidence and caches, but must not force a full index
rebuild. Build-parameter changes require rebuild or a compatible artifact
migration.

### 7. Align PostgreSQL and SQLite lexical contracts before optimizing SQL

The PostgreSQL path currently stores an unweighted tsvector and reparses
weighted title/body vectors during search. A direct switch to the stored vector
would change ranking semantics.

Use a two-step migration:

1. Define the canonical lexical contract: language configuration, phrase and
   prefix behavior, identifier behavior, title/body weights, and tie-breaking.
2. Add a versioned shadow column such as `tsvector_weighted_v1`, populate it
   with the same weighted representation used by ranking, and dual-write it
   during the migration. Backfill existing rows, switch readers only after
   parity verification, and retain the previous column until rollback is no
   longer required.

If PostgreSQL must continue using `plainto_tsquery`, document it as a plain
natural-language backend and test its intentional divergence. Otherwise add a
safe query parser that maps the supported SQLite syntax to PostgreSQL's
`phraseto_tsquery`, prefix, and boolean forms without accepting arbitrary SQL.

Parity tests must compare eligible identity sets, ordering for ties, phrase and
identifier behavior, status/workspace/metadata filters, and empty queries.

### 8. Build layered evaluation evidence

Maintain three distinct benchmark layers.

#### Backend contract benchmarks

These isolate SQLite FTS, exact vectors, FAISS, pgvector, and federation
adapters. They answer whether a backend meets its port contract and expose
backend-specific latency or recall behavior.

#### Full-pipeline retrieval benchmarks

These call `RecordSearchPipeline` or the public kernel and exercise routing,
candidate budgets, fusion, graph expansion, parent aggregation, hydration,
reranking, caching, and degradation reporting.

#### Capacity and reliability benchmarks

These measure cold/warm behavior, serial/concurrent latency, index build/load,
RSS, index size, restart, partial failure, retry, and model migration.

Every artifact must include:

```json
{
  "corpus_version": "...",
  "split": "test",
  "k": 10,
  "backend": "...",
  "model_fingerprint": "...",
  "vector_dimension": 384,
  "indexing_fingerprint": "...",
  "ann_build_fingerprint": "...",
  "ann_query_policy_fingerprint": "...",
  "routing_fingerprint": "...",
  "fusion_fingerprint": "...",
  "environment": "..."
}
```

`compare_report()` must reject incompatible metadata rather than comparing only
aggregate metrics. The local scale gate should support explicit p95 latency,
RSS, index-size, build-time, and QPS thresholds. Machine-sensitive absolute
limits remain opt-in. The default shared-CI gate should keep wall-clock latency
and QPS non-blocking because runner contention is variable; dedicated,
controlled performance environments may enable relative latency gates. Quality,
index-size, query-count, and bounded memory/allocation checks are better
candidates for deterministic blocking gates.

`indexing_fingerprint` must include chunk size, chunk overlap, tokenizer/parser
version, and other candidate-boundary parameters. A matching model fingerprint
alone is not enough to compare full-pipeline retrieval runs.

The checked-in labeled fixture should grow from a smoke corpus into a versioned
small corpus containing:

- exact identifiers, natural-language concepts, synonyms, abbreviations, and
  negative queries;
- workspace and source-kind collisions;
- graded relevance and hard negatives;
- multilingual and artifact-query slices;
- vector-only, keyword-only, hybrid, graph, parent, and reranker cases;
- explicit filter-selectivity slices.

Metrics should include recall@k, nDCG@k, MRR, AP, empty-result rate, duplicate
IDs, source coverage, calibration error where applicable, p50/p95/p99 latency,
and degradation rates. Report both record-level and parent-level retrieval
quality for chunked data.

### 9. Improve lifecycle and test hygiene

Audit all test fixtures and benchmark paths for owned SQLite managers and close
them deterministically. Keep injected-manager ownership tests proving that the
backend does not close caller-owned resources.

Update documentation from the observed 1,211-test run and distinguish skipped
optional integrations from exercised coverage. CI should have at least one
scheduled or containerized job that exercises pgvector and one job that
exercises FAISS ANN quality; the default offline gate may remain dependency
light, but it must report those gaps explicitly.

Coverage increases should target behavior currently below the quality bar,
especially vector vocabulary state, indexing bootstrap/coordinator failure
branches, reranker adapters, and ingestion failure paths. Coverage percentage
alone is not an acceptance criterion.

## Dependency-ordered delivery plan

### Phase 0 — Contract decisions and evidence schema

1. Decide and document missing-embedding behavior.
2. Define lexical parity and PostgreSQL migration semantics.
3. Define the three benchmark layers and artifact fingerprints.
4. Add failing tests for stale vectors, oversized SQL key lists, chunk parent
   loss, empty-text reranking, and metadata-incompatible evidence.

Exit criteria: contracts are written, tests demonstrate current failures, and
no implementation change depends on an unresolved ownership decision.

### Phase 1 — Correctness, bounded operations, and lexical compatibility

1. Implement vector lifecycle validation or explicit model deletion semantics.
2. Add bounded SQLite key-list operations.
3. Replace the no-FTS empty-result cliff with complete streaming fallback or an
   explicit degraded outcome.
4. Fix chunk and reranker degradation handling.
5. Add Unicode and long-document fallback tests.
6. Close owned resources and eliminate ResourceWarnings.
7. Complete a compatibility-preserving PostgreSQL weighted-tsvector migration,
   or explicitly pin and document backend lexical divergence before recording
   cross-backend baselines.
8. Run an oversized PostgreSQL `ANY(array)` sanity check in the supported
   drivers to establish empirical array/wire limits without importing SQLite's
   placeholder limit into the PostgreSQL design.

Exit criteria: focused tests pass at oversized batches, no stale vector is
servable, and fallback behavior is complete or explicitly degraded.

### Phase 2 — Trustworthy evaluation

1. Add artifact compatibility validation and gate thresholds.
2. Separate backend and full-pipeline runners.
3. Expand the labeled corpus and add vector/hybrid/graph/reranker slices.
4. Add mandatory scheduled backend parity and ANN evidence.

Exit criteria: a report cannot pass using a different corpus, backend, model,
or configuration; full-pipeline quality is measured independently from backend
microbenchmarks.

### Phase 3 — Retrieval quality experiments

1. Establish an RRF baseline on the expanded corpus.
2. Compare calibrated fusion, relative normalization, and learned monotonic
   calibration.
3. Measure adaptive ANN overfetch by filter selectivity.
4. Measure candidate starvation caused by lane budgets, graph expansion, and
   parent aggregation.
5. Tune only parameters supported by held-out evidence.

Exit criteria: every default change has a before/after report, slice-level
quality evidence, and a rollback configuration.

### Phase 4 — Backend performance and scale

1. Optimize snapshot construction and FAISS persistence based on measured
   contention and memory profiles.
2. Add controlled concurrency and capacity thresholds.
3. Reassess whether Python overhead is material after database and native paths
   are profiled.

Exit criteria: production-shaped capacity evidence exists for each supported
   deployment path, with known limits and degradation behavior.

## Verification matrix

| Change | Focused proof | Broader proof |
| --- | --- | --- |
| Vector lifecycle | stale update, retry, model deletion | local/FAISS/Postgres parity |
| Bounded SQL | batch above variable limit | 100k ingestion and cache reuse |
| FTS fallback | forced no-FTS corpus over 10k | latency and completeness slices |
| Chunk handling | missing parent, strict/lenient hydration | parent-vs-record metrics |
| Reranking | indexed-text-only and empty candidates | reranker adapter integration |
| Fusion | tied lane, weak singleton, calibration version | held-out labeled quality |
| ANN filtering | selectivity/under-return matrix | backend capacity benchmark |
| PostgreSQL lexical parity | phrase, prefix, weights, filters | psycopg2/psycopg3 integration |
| Evidence gates | incompatible metadata and threshold regressions | release-readiness artifact |
| Lifecycle | owned and injected close behavior | restart and concurrent tests |

The ANN selectivity matrix must include compound filters, not only isolated
workspace or source-kind filters. Verification must also include concurrent
reads during vector revision migration, fallback scans during active SQLite
transactions, and reranker input at model context limits.

The standard verification command remains:

```bash
uv run ruff check searchkernel tests benchmarks
uv run pyrefly check
uv run lint-imports
uv run pytest -q --strict-markers -m 'not slow and not real_embeddings' \
  --cov=searchkernel --cov-report=term-missing
```

Never use `ruff format` as part of this work.

## Rollout and rollback

- Keep existing RRF and exact-vector behavior as the default until experiments
  demonstrate an improvement.
- Gate new calibration, adaptive ANN, and lexical parser behavior behind
  configuration fingerprints.
- Version any persisted FAISS metadata, calibration artifact, or PostgreSQL
  vector representation.
- Roll back by selecting the prior configuration/model namespace, not by
  deleting shared storage.
- If a migration partially completes, leave the old serving namespace intact
  until validation and an atomic active-model switch succeed.
- Publish degradation diagnostics with every bounded fallback so operators can
  distinguish unavailable evidence from genuine empty results.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Deleting stale vectors breaks callers that intentionally preserve them | Make the contract explicit and provide a separate keyword-only update path |
| Larger fallback scans increase latency | Heap-based top-k, bounded budgets, metrics, and explicit degradation |
| Calibration overfits the small corpus | Held-out splits, slice reporting, and RRF fallback |
| PostgreSQL lexical migration changes ranking | Backfill a versioned weighted-vector column, compare parity, retain rollback schema/configuration |
| Blocking latency gates flap on shared CI | Keep them non-blocking by default and run them in a dedicated environment |
| Adaptive ANN increases tail latency | Per-query ceilings, exact fallback budget, p95 and under-return monitoring |
| More CI dependencies reduce reliability | Keep offline gate separate from scheduled optional-backend evidence |
| Broad plan creates unreviewable changes | Deliver by dependency-ordered, independently valid slices |

## Open decisions

1. Should a missing embedding be rejected at the `VectorStore` boundary or
   interpreted as an explicit model-vector deletion?
2. Should chunk results be allowed in the public outcome when parent hydration
   fails? The recommended default is parent-only with structured degradation;
   chunk-preserving output should be opt-in.
3. Which lexical query syntax is the supported cross-backend contract: plain
   natural language only, or a restricted phrase/prefix/identifier subset?
4. Which deployment environments are stable enough for blocking latency gates?
5. What minimum corpus size and language mix are required before learned
   calibration can be considered for the default?
