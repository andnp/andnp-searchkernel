# Search Performance and Retrieval Quality Roadmap

Status: implementation complete; performance validation remains scoped
Last reviewed: 2026-08-01
Applies to: `andnp-searchkernel` 0.2.x pre-alpha

## Implementation status

Milestones 0-9 are implemented in the current code and committed contract
tests. The canonical record path now owns query execution, with FTS5 keyword
retrieval, compact local vectors, optional FAISS, batched retrieval, caching,
query routing, filtered pgvector support, storage integrity, source diversity,
hierarchical retrieval, and the legacy chunk query pipeline removed.

The implementation status does not imply production performance parity. The
committed evidence and remaining validation limits are recorded in
[Committed evidence and limits](#committed-evidence-and-limits).

## 1. Purpose

This document records the implementation and validation plan for making
`andnp-searchkernel` fast, lightweight, and effective for AI-driven search
across diverse data sources. It preserves the architecture, constraints, and
evidence requirements behind the completed milestones.

The main conclusion of the review is that the largest near-term gains do not
require adding complex retrieval methods. The repository already contains many
of the right primitives, but they are split between the canonical record path
and an older chunk-oriented path. The first objective is to make the canonical
record path use the efficient indexing, caching, batching, routing, and
evaluation mechanisms that already exist elsewhere in the package.

## 2. Target outcomes

The completed roadmap should provide:

- fast local hybrid search without scanning every record for every query;
- bounded memory use and compact on-disk vector storage;
- predictable latency under concurrent queries;
- high recall across natural language, identifiers, file paths, code symbols,
  and relationship queries;
- consistent behavior across local SQLite/FAISS and Postgres/pgvector backends;
- source-aware ranking without putting source-specific logic in core domain
  models;
- incremental indexing that reuses embeddings and avoids unnecessary writes;
- meaningful relevance, latency, throughput, memory, and index-size gates;
- one canonical record-oriented search path instead of two drifting pipelines.

The package should remain lightweight. SQLite, NumPy, and the standard library
remain sufficient for the default local backend. FAISS, Hugging Face, pgvector,
and source-specific parsers remain optional extras.

## 3. Current architecture

### 3.1 Canonical record path

The supported record-oriented flow is:

```text
SearchKernel.search_anything
  -> federation.search_anything
     -> SearchableSource.search (concurrent across sources)
        -> LocalSearchSource
           -> SearchOrchestrator
              -> RecordSearchPipeline
                 -> KeywordStore
                 -> EmbeddingProvider + VectorStore
                 -> GraphStore
                 -> RecordHydrator
```

Important files:

- `searchkernel/kernel.py`: composition root and federation entry point;
- `searchkernel/runtime/federation.py`: source fanout, RRF, and one optional
  cross-source rerank;
- `searchkernel/runtime/local.py`: adapts record results to federation;
- `searchkernel/search/orchestrator.py`: canonical local orchestrator;
- `searchkernel/search/record_pipeline.py`: record retrieval, graph expansion,
  policy application, and hydration;
- `searchkernel/indices/local.py`: canonical SQLite-backed local stores;
- `searchkernel/adapters/stores/pgvector.py`: Postgres record stores.

### 3.2 Older chunk-oriented path

The richer older query path built around `Chunk`, `VectorIndex`, `KeywordIndex`,
`GraphStore`, and declarative pipeline stages has been retired. The chunk
indices and indexing transforms remain only where they serve ingestion and
indexing contracts. The canonical record path now owns query execution.

The remaining indexing compatibility files include:

- `searchkernel/indices/keyword.py`;
- `searchkernel/indices/vector.py`;
- `searchkernel/indices/graph.py`;
- `searchkernel/indexing/core.py`;
- `searchkernel/indexing/stages.py`.

Do not build a third pipeline. Reuse pure indexing primitives behind the
record-oriented ports while keeping query execution on the canonical path.

### 3.3 Existing primitives that must be reused

Before adding a new mechanism, check these existing implementations:

- `searchkernel/runtime/query_embedding_cache.py`: bounded query-embedding LRU
  with TTL and leader/follower miss coalescing;
- `searchkernel/runtime/cache.py`: epoch-aware cache keys;
- `searchkernel/search/result_cache.py`: legacy query-result cache key shape;
- `searchkernel/indexing/embedding_cache.py`: durable, encoder-namespaced SQLite
  embedding cache;
- `searchkernel/indexing/semantic.py`: canonical embedding text, encoder
  fingerprints, content-hash identities, deduplication, and work planning;
- `searchkernel/search/fusion.py`: RRF primitive;
- `searchkernel/search/adaptive_limit.py`: bounded adaptive result limits;
- `searchkernel/eval/`: golden-set loader, metrics, reports, and gates;
- `searchkernel/runtime/fanout.py`: per-source concurrency and timeouts.

## 4. Current bottlenecks and risks

### 4.1 Local keyword search is an O(N) scan

`LocalRecordBackend.search_keyword` reads every row from `local_records`,
lowercases `title + body`, counts each query token as a substring, sorts all
matches, and truncates to `k`.

Consequences:

- query time grows linearly with corpus size;
- every query allocates large temporary strings;
- substring counts produce weak ranking and false matches;
- indexed workspace/status/source filters are applied in Python after rows are
  loaded, so the SQLite indexes provide little benefit;
- the backend does not reuse the repository's existing FTS5/BM25 logic.

### 4.2 Local vector search is an O(N * D) JSON scan

`LocalRecordBackend.search_vector` fetches all vectors for a model/dimension,
parses each JSON vector, converts it to NumPy, recomputes its norm, scores it,
sorts every candidate, and only then returns `k` results.

Consequences:

- JSON is much larger and slower to decode than packed float32 data;
- vector norms are recalculated for every query;
- SQL filters are applied in Python;
- a process-wide `RLock` is held during the scan;
- synchronous CPU and database work can block the async search loop;
- the misleading `FAISSVectorStore` alias has been removed; callers must choose
  the explicit `FAISSLocalVectorStore` backend.

### 4.3 The canonical record pipeline performs N+1 operations

`RecordSearchPipeline` currently:

- waits for keyword retrieval before generating/querying the vector lane;
- calls `GraphStore.neighbors` once per graph seed;
- hydrates one record at a time;
- constructs provenance by scanning each strategy list for every fused
  candidate.

The last item is avoidable quadratic work. The graph and hydration calls are
more important because remote adapters can turn them into many network or
database round trips.

### 4.4 Existing caches are not wired into the canonical path

The query embedding cache is tested but the canonical record pipeline and
`PGVectorIndex` call embedding providers directly. The general cache utilities
and backend epochs are also not used by `RecordSearchPipeline`.

The local epoch currently changes during record/vector writes, but graph writes
do not bump it. A future cache based on that epoch would therefore return stale
graph-expanded results. Record-plus-vector upserts can also increment the epoch
more than once for one logical batch.

### 4.5 Filtered ANN search can under-return

The pgvector query joins the vector table to `records` and applies status,
workspace, and source filters. HNSW can produce too few post-filter results if
the initial ANN scan does not visit enough eligible rows. The legacy
`PGVectorIndex` also uses a fixed `top_k * 2` overfetch for file exclusions and
then filters in Python, which can still return fewer than `top_k` results.

The record pipeline can provide `candidate_ids`, but all stores must explicitly
support that filter. Silently ignoring it changes retrieval semantics.

### 4.6 Evaluation is not strong enough to guide tuning

The current evaluation runner calls the search function once per query. It has
no warmup, repeats, concurrency, cold/warm split, throughput measurement, or
memory/index-size measurement. Its percentile calculation selects an array
index directly and is not a standard interpolated or nearest-rank percentile.

Golden entries store only a query and binary relevant IDs. They cannot express
graded relevance, query type, source, tenant/workspace, difficulty, temporal
split, or corpus version. Aggregate results can therefore hide regressions for
specific sources and query classes.

### 4.7 Storage is duplicated and can drift

`ChunkHashStore` loads its complete JSON map and a reverse map into memory. It
scans all chunk IDs for document deletes and lookups, and persists the complete
map after changes. Local graph edges have no foreign keys to records and graph
writes do not bump the search epoch. Deleting a record can leave graph edges
behind.

## 5. Design constraints and invariants

Every implementation must preserve these rules:

1. **Canonical identity**
   
   `RecordIdentity` and `storage_key` remain the cross-backend identity. Never
   deduplicate solely by `source_id`, since two sources or workspaces may use
   the same ID.

2. **Source-agnostic core**
   
   Source-specific fields and policies enter through adapters or injected
   policy objects. Core domain models must not gain fields for Markdown, Jira,
   Git, Obsidian, or any other one source.

3. **Deterministic ordering**
   
   Equal scores must use a stable identity tie-breaker. Batch and concurrent
   implementations must return the same order as scalar implementations.

4. **Read-only search**
   
   Query execution must not mutate source lifecycle, checkpoints, access
   state, or supersession state.

5. **Strict and lenient failures**
   
   Existing strict mode raises stage errors. Lenient mode returns explicit
   degradation diagnostics. Batch and cache layers must not hide failures.

6. **Model-safe vectors**
   
   Vector identities include the full encoder fingerprint: model, version,
   normalization, instructions, dimension, and truncation settings. Model name
   alone is insufficient for durable caches.

7. **Incremental ingestion safety**
   
   A checkpoint advances only after durable successful writes. Partial failures
   must remain retryable.

8. **Optional heavy dependencies**
   
   Importing the core package must not require FAISS, Hugging Face, Postgres,
   tree-sitter, or an LLM provider.

9. **Backward-compatible migrations**
   
   Existing SQLite/Postgres data must be migrated or rebuilt explicitly. Never
   reinterpret old JSON vectors as a new binary format without a schema/version
   check.

10. **Measure before changing defaults**
    
    Candidate counts, ANN thresholds, RRF weights, score cutoffs, cache sizes,
    and rerank budgets must be selected using the evaluation harness, not by
    intuition alone.

## 6. Implementation sequence

The milestones are ordered deliberately. Complete evaluation first, then make
backend and pipeline changes one at a time so each improvement can be measured.

| Order | Milestone | Primary outcome | Dependency |
|---:|---|---|---|
| 0 | Evaluation and benchmark foundation | Trustworthy comparisons | None |
| 1 | Canonical SQLite FTS5 keyword index | Remove local lexical full scan | M0 |
| 2 | Compact local vector engine | Remove JSON vector scan overhead | M0 |
| 3 | Batched and concurrent record pipeline | Remove N+1 latency | M0 |
| 4 | Canonical caching and epochs | Avoid repeated expensive work | M1-M3 |
| 5 | Query routing and adaptive fusion | Improve speed and relevance | M0-M4 |
| 6 | pgvector filtered-search parity | Reliable remote ANN behavior | M0, M4 |
| 7 | Storage integrity and compaction | Lower memory and prevent drift | M1-M4 |
| 8 | Source diversity and hierarchical retrieval | Better mixed-source results | M0, M5 |
| 9 | Pipeline consolidation | Reduce weight and maintenance cost | M1-M8 |

## 7. Milestone 0: evaluation and benchmark foundation

### Goal

Make performance and relevance changes measurable and safe to compare.

### Required changes

1. Extend `GoldenEntry` without breaking existing JSON fixtures. Add optional:

   - `relevance: dict[str, float]` for graded gains;
   - `query_type: str | None`;
   - `source_kinds: list[str]`;
   - `workspace_id: str | None`;
   - `tags: list[str]` for slices such as `identifier`, `temporal`, `hard`, or
     `cross-source`;
   - `corpus_version: str | None` and `split: train|validation|test|None`.

   Continue accepting `relevant_ids`. If both fields exist, `relevance` is the
   authoritative graded map and `relevant_ids` is its positive-gain ID set.

2. Correct percentile calculation. Choose one documented method and test small
   samples explicitly. Prefer a standard library implementation or linear
   interpolation rather than a custom array index.

3. Add a benchmark runner configuration with:

   - configurable warmup count;
   - configurable measured repetitions;
   - serial and concurrent runs;
   - cold-start and warm-cache reports;
   - p50, p95, p99, mean, min, max, and QPS;
   - per-stage latency when the search implementation exposes a trace;
   - process RSS before/after index load and during peak query load;
   - index build time and total on-disk size.

4. Add slice reporting by source, query type, and tags. At minimum report:

   - recall@k;
   - nDCG@k with graded gains;
   - MRR;
   - mean average precision;
   - empty-result rate;
   - source coverage in the top-k;
   - per-source recall;
   - latency percentiles.

5. Add deterministic synthetic corpora at approximately 1k, 10k, and 100k
   records. Keep the smallest corpus in normal tests; mark larger runs as
   benchmark/slow so routine unit tests remain fast.

6. Add an A/B output that records configuration fingerprints, corpus version,
   backend, model fingerprint, and environment metadata. A report without this
   context is not reproducible.

### Acceptance criteria

- Existing golden JSON continues to load.
- Graded gains affect nDCG as expected.
- Percentiles match documented examples for one, two, three, and large sample
  sets.
- Warmup calls do not appear in measured latency.
- Repeated runs report both per-query distributions and aggregates.
- Concurrent runs verify output determinism.
- A baseline report is checked into a non-test-artifact location or attached to
  the implementation PR before M1 begins.

### Narrow verification

```bash
uv run pytest -q tests/unit/test_eval_golden.py \
  tests/unit/test_eval_metrics.py \
  tests/unit/test_eval_runner.py \
  tests/unit/test_eval_gates.py
uv run ruff check searchkernel/eval tests/unit/test_eval_*.py
uv run pyright searchkernel/eval
```

### Committed evidence and limits

`benchmarks/milestone-0-baseline.json` records a synthetic 1k-record,
single-process cold/warm baseline with 16 queries, one warmup, and three
measured repetitions. It is a harness baseline, not a production or
cross-backend performance comparison.

`benchmarks/local_scale_gate.py` with
`benchmarks/local_scale_gate.json` defines the reproducible local scale-gate
scope: synthetic 1k, 10k, and 100k record corpora; seed 0; 32-dimensional
vectors; top-10 retrieval; one warmup; three serial repetitions; and optional
FAISS recall@10 with a configured 0.9 minimum when FAISS is available. No
committed scale-gate report exists for those runs. In particular, this
roadmap makes no unrun claims about 10k/100k scaling, ANN recall, keyword
scaling, pgvector latency, or production-corpus relevance.

## 8. Milestone 1: canonical SQLite FTS5 keyword index

### Goal

Replace the O(N) local keyword scan with indexed BM25 retrieval while keeping
SQLite as the only required storage dependency.

### Design

Use an FTS5 table tied to `local_records`. Prefer an external-content FTS table
to avoid duplicating full bodies. `local_records` is currently a rowid table
despite its text primary key, so its internal integer rowid can be used as the
FTS rowid. If a migration changes it to `WITHOUT ROWID`, add an explicit integer
primary key first.

Initial indexed fields:

- title;
- body;
- URI/path;
- normalized tags or source-provided keywords.

Keep `storage_key`, workspace, source kind, and status in `local_records` and
join/filter them in SQL. Do not rely on post-query Python filtering.

Use the existing functions in `indices/keyword_scoring.py` for artifact-query
detection, query sanitization, and field-aware boosts. RRF uses rank order, so
backend BM25 scores need to be internally consistent but do not need to match
vector scores.

### Required changes

1. Add a schema version and migration/rebuild path for the FTS table.
2. Keep record and FTS updates in the same transaction.
3. On upsert, remove/update the prior FTS row before inserting the new text.
4. On delete, remove the FTS row in the same transaction as the record.
5. Implement SQL filters for:

   - status/statuses/include-inactive;
   - workspace;
   - source kind(s);
   - candidate storage keys.

6. Use bounded overfetch only where field-aware reranking or artifact matching
   occurs. Make the multiplier configurable and measure it.
7. Retain a safe fallback for builds of SQLite without FTS5. The fallback can
   use the current scan for very small/test corpora, but it must expose a
   diagnostic that indexed lexical search is unavailable.
8. Add an integrity check/rebuild operation so external-content rows cannot
   silently drift from `local_records`.

### Tests

- exact token, phrase, prefix, filename, URI, and code-symbol queries;
- case handling and query sanitization;
- workspace/source/status/candidate filters;
- update and delete consistency;
- migration from the existing schema;
- corruption/rebuild behavior;
- deterministic tie order;
- equivalence of scalar and batched ingestion;
- benchmark showing sublinear query growth from 1k to 100k records.

### Acceptance criteria

- `search_keyword` does not call `SELECT * FROM local_records`.
- filters are present in the SQL query plan before result materialization.
- deleted or inactive records do not appear.
- an identifier-heavy golden slice does not regress.
- at 100k records, lexical p95 is materially lower than the scan baseline and
  stays within the agreed benchmark gate.

## 9. Milestone 2: compact local vector engine

### Goal

Provide a fast default exact-search path for small corpora and an optional ANN
path for larger corpora without requiring a standalone vector database.

### Storage format

Replace JSON vectors with packed, little-endian float32 bytes. Validate finite
values and dimensions at ingestion. Normalize vectors once at ingestion so
cosine similarity becomes a dot product. Record the storage format version,
model fingerprint, dimension, and normalization policy.

Suggested table shape:

```sql
CREATE TABLE local_vectors_v2 (
    storage_key TEXT NOT NULL,
    encoder_namespace TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (storage_key, encoder_namespace),
    FOREIGN KEY (storage_key) REFERENCES local_records(storage_key)
        ON DELETE CASCADE
);
```

Do not silently migrate malformed vectors. Decode old JSON, validate, write the
new row, and only then remove or mark the old representation migrated.

### Exact engine for small corpora

Maintain an immutable in-memory snapshot per encoder namespace:

- contiguous `float32` matrix;
- parallel array of `RecordIdentity` or storage keys;
- epoch used to build the snapshot;
- optional filter metadata arrays.

On search, normalize the query once, select eligible rows, use a matrix-vector
dot product, and use `argpartition` plus a deterministic final sort instead of
sorting every score. Rebuild the snapshot lazily after an epoch change. Never
hold the SQLite write lock during NumPy scoring.

For corpora too large for one matrix, support block-wise exact scoring or a
memory-mapped matrix before introducing a mandatory ANN dependency.

### Optional ANN engine

Use the existing optional FAISS dependency above a measured/configured record
threshold. The exact threshold must come from M0 benchmarks.

- use explicit stable IDs, such as `IndexIDMap2`, rather than positional IDs;
- maintain an ID-to-`RecordIdentity` mapping with the same epoch/version as the
  index;
- either remove IDs safely or use tombstones with adaptive overfetch;
- rebuild when the tombstone ratio crosses a measured threshold;
- save index and mapping atomically;
- fall back to exact search if the ANN index is absent, stale, or corrupt;
- verify ANN recall against exact cosine search.

Use names that reveal the engine, such as `SQLiteExactVectorStore` and
`FAISSLocalVectorStore`; the misleading compatibility alias is removed.

### Filtering

Apply workspace, source, status, and candidate filters before expensive scoring
when possible. For exact snapshots, maintain compact filter arrays or map a SQL
result set to vector positions. For FAISS, use selector support where practical
or adaptive overfetch with a hard bound and diagnostics.

### Tests and gates

- binary round-trip and dimension/finite-value validation;
- migration from JSON vectors;
- exact score parity with the current cosine implementation;
- deterministic ties;
- filtering before top-k truncation;
- index reload, corruption, deletion, and tombstone rebuild;
- ANN recall@10 >= the configured gate against exact search;
- memory and disk usage lower than the JSON baseline;
- no event-loop blocking in the async composition path.

## 10. Milestone 3: batched and concurrent record pipeline

### Goal

Remove avoidable serial waits, database round trips, and quadratic bookkeeping.

### New optional capabilities

Add narrow capability protocols rather than forcing every adapter to implement
batch methods immediately:

```python
class BatchRecordHydrator(Protocol):
    def hydrate_records(
        self, identities: Sequence[RecordIdentity]
    ) -> Mapping[str, Record | None] | Awaitable[Mapping[str, Record | None]]: ...

class BatchGraphStore(Protocol):
    def neighbors_many(
        self,
        identities: Sequence[RecordIdentity],
        *,
        depth: int,
    ) -> Mapping[str, Sequence[GraphNeighbor]] | Awaitable[...]: ...
```

Exact names may change, but keys must use canonical storage keys and results
must preserve identity.

### Retrieval scheduling

1. If vector acquisition does not depend on keyword candidate IDs:

   - start keyword search;
   - start query embedding generation;
   - start vector search as soon as the embedding completes;
   - await both lanes before fusion.

2. If `policy.vector_candidate_ids` is configured, keyword acquisition remains
   a dependency. Still generate the query embedding concurrently with keyword
   search so only vector lookup waits.

3. Synchronous adapters must not run slow database/model work directly in the
   event loop. Prefer explicit async adapter wrappers using `asyncio.to_thread`
   at the composition boundary. Do not blindly offload tiny pure functions.

### Graph expansion

- prefer `neighbors_many` for all seeds in one call;
- otherwise issue scalar async calls with a bounded semaphore;
- preserve the existing max seeds, max neighbors, depth, discounts, provenance,
  and deterministic order;
- avoid the current second lookup by plain `source_id` when a canonical identity
  lookup succeeds; legacy fallback should be explicit and observable.

### Hydration

- request all selected identities in one batch when supported;
- map returned records back to candidate order;
- report missing IDs in the existing outcome field;
- in lenient mode, preserve per-stage errors without discarding successful
  records;
- do not hydrate candidates discarded by the adaptive result limit.

### Provenance complexity

Before building candidates, build one map per strategy:

```text
storage_key -> (rank, original_score, identity)
```

Candidate provenance can then be assembled in O(strategies * candidates)
instead of scanning each entire ranking for every candidate.

### Acceptance criteria

- scalar and batch implementations return identical outcomes;
- query embedding overlaps keyword I/O when candidate gating is disabled;
- one remote graph query and one hydration query are made per search when batch
  capabilities exist;
- strict/lenient behavior remains unchanged;
- cancellation propagates to outstanding tasks;
- deterministic output holds under concurrent completion order;
- stage timings demonstrate reduced graph and hydration latency.

## 11. Milestone 4: canonical caching and epoch semantics

### Goal

Avoid repeated embedding, retrieval, and hydration work without serving stale
or cross-tenant results.

### 11.1 Query embedding cache

Reuse `QueryEmbeddingCache`, but extend or wrap it safely for async providers.
The current thread `Event.wait()` must not block an asyncio event loop.

Cache key:

```text
(encoder_namespace, normalized_query_text)
```

Query normalization should strip outer whitespace and normalize repeated
whitespace. Preserve case because case may matter for identifiers and code. The
encoder namespace must include all model settings described in M2.

Required behavior:

- bounded LRU;
- TTL;
- sync and async single-flight miss coalescing;
- defensive copies or immutable vector storage;
- failures and cancellations are never cached;
- hit, miss, eviction, coalesced-waiter, and compute-time metrics.

Wire it once at the embedding-provider/composition boundary so local,
pgvector, record search, legacy compatibility adapters, and evaluation all use
the same behavior.

### 11.2 Candidate-result cache

Cache the small unhydrated result, not full record bodies:

```text
[(RecordIdentity, fused_score, provenance), ...]
```

Key fields:

- normalized query;
- canonical, stable serialization of filters;
- requested/acquisition/adaptive limits;
- routing and fusion configuration fingerprint;
- encoder namespace;
- keyword, vector, and graph epochs;
- policy version supplied by the application when policy can change results.

Do not use `repr()` of arbitrary objects as a durable cross-process cache key.
Reject or bypass caching when a filter/policy cannot be represented stably.

Use separate lane epochs. A vector-only update should not invalidate pure
keyword cache entries. A graph write must invalidate graph-expanded candidates.

### 11.3 Hydration cache

Use a small optional cache keyed by canonical identity plus a record version,
updated timestamp, or backend record epoch. Never cache authorization decisions
unless the authorization policy/version is also in the key. Do not cache a
missing record for long; ingestion may make it available immediately.

### 11.4 Epoch corrections

- one logical record/vector batch increments relevant epochs once after commit;
- graph edge writes and deletes increment the graph epoch;
- failed/rolled-back writes do not increment epochs;
- epoch reads are cheap and can be fetched together;
- old cache entries are lazily ignored and periodically pruned rather than
  scanned on every write.

### Acceptance criteria

- identical concurrent query misses cause one embedding computation;
- repeated warm queries avoid embedding and retrieval work;
- changing any relevant filter, model setting, policy version, or lane epoch
  causes a miss;
- graph changes cannot return stale graph-expanded results;
- cache failures degrade to normal computation;
- cache memory is bounded and metrics are exposed.

## 12. Milestone 5: query routing and adaptive fusion

### Goal

Run only useful retrieval work for each query and improve ranking across query
types without making the default pipeline heavy.

### Query plan

Introduce a small immutable `QueryPlan` produced by the router. Suggested
fields:

- query type/signals;
- enabled lanes: keyword, vector, graph;
- candidate budget per lane;
- fusion weight per lane;
- whether vector candidates are keyword-bounded;
- graph depth/seed budget;
- rerank budget;
- optional expansion strategy.

Reuse `search/classifier.py` as the starting heuristic. Extend it using measured
signals rather than replacing it with an LLM router.

### Recommended routing behavior

1. **Artifact/identifier queries**

   Examples: `RecordSearchPipeline`, `foo_bar`, `ABC-123`, file paths, quoted
   names, versions, and symbols containing `_`, `:`, `-`, `/`, or dots.

   - run keyword first;
   - use exact/prefix/path boosts;
   - if enough high-confidence keyword results exist, return without generating
     a query embedding;
   - otherwise fall through to hybrid retrieval.

2. **Natural-language factual questions**

   - run keyword and vector concurrently;
   - use hybrid weighted RRF;
   - enable a small rerank budget when configured.

3. **Exploratory/semantic queries**

   - give vector retrieval a larger budget;
   - retain keyword candidates for precise terms;
   - allow an adaptive result limit.

4. **Relationship/navigation queries**

   - enable graph expansion only when the query or initial candidates contain
     relationship signals;
   - keep graph depth and fanout bounded.

5. **Ambiguous or low-recall queries**

   - optionally run RAG-Fusion, HyDE, or another expansion only after the cheap
     first pass is weak;
   - keep this behind a feature flag and a strict latency/cost budget.

### Candidate budgets

The current default is `limit * 5`, with a minimum of one. Replace the single
multiplier with lane-specific budgets. Do not hard-code a new minimum such as
50 until benchmarks establish its recall/latency tradeoff. Ensure filtering and
deduplication still leave enough candidates for the requested result count.

### Weighted RRF

Add a backwards-compatible weighted RRF primitive:

```text
score(item) = sum(strategy_weight / (rrf_k + rank))
```

Keep plain RRF as the default compatibility behavior. Use the query plan and
optional per-source reliability weights to supply weights. Preserve original
scores in provenance, but do not add raw vector, BM25, and graph scores together
without calibration.

### Reranking

- federate and deduplicate first;
- select a bounded top-M candidate set;
- rerank once across sources;
- pass compact title + best snippet + useful source metadata rather than
  unbounded full bodies;
- fall back to RRF deterministically when candidate text or the reranker fails.

### Acceptance criteria

- strong artifact queries can complete without an embedding call;
- natural-language queries retain or improve recall/nDCG;
- graph storage is not touched when graph is disabled;
- weighted RRF has deterministic ties and full provenance;
- optional expansion cannot exceed its latency/call budget;
- routing decisions and skip reasons appear in query diagnostics.

## 13. Milestone 6: pgvector filtered-search parity

### Goal

Make Postgres search reliable under workspace, source, lifecycle, candidate,
and file filters while sharing canonical caching and embedding semantics.

### Required changes

1. Support every canonical vector filter explicitly, including
   `candidate_ids`. Add contract tests that run the same filter suite against
   local and pgvector stores.

2. Add relational indexes based on measured query plans. A likely starting
   point is a composite index covering workspace, status, source kind, and
   record ID. Use `EXPLAIN (ANALYZE, BUFFERS)` on representative filtered
   queries before finalizing indexes.

3. Enable pgvector iterative HNSW scans when the installed extension supports
   them. Feature-detect the server extension version and use a bounded adaptive
   overfetch fallback on older servers. Do not assume the Python client package
   version equals the server extension version.

4. Make `hnsw.ef_search`, iterative scan mode, and scan bounds query/config
   settings with safe validated limits. Include them in benchmark metadata.

5. Move path/project/document filtering into SQL. For frequently filtered JSON
   metadata, consider generated columns or expression indexes rather than a
   broad JSON scan.

6. Replace `PGVectorIndex.search`'s fixed `top_k * 2` file-exclusion overfetch
   with SQL filtering or iterative bounded overfetch until `top_k` eligible
   records are found.

7. Wire the canonical query embedding cache into `PGVectorIndex`.

8. `get_embedding_for_chunk` must fetch the stored vector for the active model
   instead of re-embedding body text. Re-embedding is slower and currently does
   not reproduce the header-plus-content text used at ingestion.

9. `expand_query` must not silently be a no-op while claiming backend parity.
   Either move expansion above the backend into the canonical pipeline or mark
   the capability unsupported and route around it.

### Acceptance criteria

- local and pgvector contract tests return equivalent eligible identities;
- heavily filtered ANN queries return `k` results when at least `k` eligible
  records exist within configured scan bounds;
- stored embeddings round-trip without model calls;
- repeated queries hit the shared embedding cache;
- query plans use expected relational and HNSW indexes;
- ANN recall and filtered recall pass evaluation gates.

## 14. Milestone 7: storage integrity and compaction

### Goal

Reduce duplicated state, prevent dangling records, and keep write/invalidation
semantics transactional.

### Hash storage

Replace the full-memory JSON `ChunkHashStore` with a SQLite-backed store or fold
the fields into the canonical record database.

Required indexes:

- chunk/storage key -> content hash;
- document identity -> chunk keys;
- content hash -> chunk keys for move/dedup detection.

Batch set/delete operations and commit them with related manifest/index changes
where possible. Provide an explicit one-time JSON migration and corruption
recovery path.

### Graph integrity

- use canonical storage keys at both ends;
- add source and target indexes, plus `(source, edge_type)` if filtering uses it;
- use foreign keys/cascades where graph nodes are guaranteed to be records;
- otherwise perform explicit transactional edge cleanup on record deletion;
- increment graph epoch after committed edge mutations;
- add integrity checks for dangling edges and malformed identities.

### SQLite tuning

Keep WAL and `synchronous=NORMAL` defaults. Make these measured/configurable
rather than hard-coded globally:

- busy timeout;
- page/cache size;
- memory map size;
- temp store;
- checkpoint policy.

Avoid holding Python locks during CPU scoring or network/model calls. SQLite
transactions should cover only the required database mutation/read snapshot.

### Acceptance criteria

- startup memory no longer scales with a duplicated hash map;
- document hash lookup/delete uses an index instead of scanning all chunks;
- deleting a record cannot leave query-visible graph edges;
- each logical batch has one atomic commit and correct epoch increments;
- corruption and interrupted migration tests recover or fail explicitly.

## 15. Milestone 8: source diversity and hierarchical retrieval

### Goal

Improve mixed-source usefulness without embedding application-specific policy in
the core.

### Source-aware fields

Define an adapter-owned extraction result such as:

```text
title, body, uri/path, tags, identifiers, parent_id, source_timestamp,
authority, language, access labels
```

These are generic retrieval concepts, not fields for one product. Each source
adapter maps its native data into them. Keyword stores can weight fields;
vector adapters can build canonical embedding text; policy hooks can use
authority and time without changing core storage contracts.

### Source-balanced top-k

After fusion and before expensive reranking/hydration, apply an optional
diversity policy:

- cap results per document/entity;
- optionally cap or reserve slots per source;
- use a light maximal-marginal-relevance pass when embeddings are already
  available;
- never force diversity when it would replace clearly relevant results with
  irrelevant ones.

Measure source coverage and per-source recall. Do not tune only aggregate
recall.

### Hierarchical retrieval

For large structured sources, support two representations:

- coarse document/entity summary record;
- fine chunk/section records linked to the parent.

Retrieve coarse candidates first, then search or promote fine children within
the best parents. This reduces fine-grained candidate volume and improves
context assembly. Keep flat retrieval available for small or unstructured
sources.

Do not introduce LLM-generated summaries into the default ingestion path. Use
source-provided summaries or deterministic truncation first; make generated
summaries an optional enrichment tier.

### Acceptance criteria

- mixed-source golden sets report per-source and aggregate improvements;
- one large source cannot crowd out all other relevant sources;
- parent/child results preserve canonical identity and provenance;
- hierarchical retrieval stays behind a capability/config flag;
- default lightweight installs do not require an LLM.

## 16. Milestone 9: pipeline consolidation

### Goal

Make the record-oriented path the only supported search architecture and remove
duplicate behavior that increases package size and maintenance cost.

### Migration process

1. Inventory every stage in `pipeline/default_query_spec.py`.
2. For each stage, classify it as:

   - required in the canonical record path;
   - application policy;
   - optional advanced retrieval;
   - obsolete or source-specific.

3. Port required behavior behind record ports/policies, reusing pure functions
   where possible.
4. Add cross-path golden tests during the transition.
5. Migrate all in-repo adapters and documented callers.
6. Deprecate old public imports with clear replacements.
7. Remove the legacy execution path only after downstream consumers have moved
   and parity gates pass.

Likely mappings:

| Legacy capability | Canonical destination |
|---|---|
| routing/effective top-k | `QueryPlan` + `RecordSearchConfig` |
| retrieve | record keyword/vector stores |
| graph/tag expansion | graph capability + application policy |
| fusion/calibration | record fusion module |
| project/source filters | canonical backend filters/policy |
| dedup/doc limit | post-fusion candidate policy |
| rerank | federation one-pass reranker |
| parent expansion | hierarchical record capability |
| hydrate | batch record hydrator |
| provenance | `SearchResultProvenance` |

### Acceptance criteria

- README and public examples use only the canonical record path;
- backend contract tests cover local and pgvector implementations;
- no behavior is removed without a documented replacement or deliberate
  deprecation decision;
- optional dependency imports remain lazy;
- package import time and installed size do not regress;
- obsolete aliases and duplicate caches/pipelines are removed after migration.

## 17. Cross-cutting observability

Add a record-oriented equivalent of `QueryExecutionStats`. It should be cheap
when disabled and contain no full query text or record body by default.

Recommended fields:

- query-plan type and enabled lanes;
- cache hits/misses and single-flight waits;
- embedding, keyword, vector, graph, fusion, policy, hydration, rerank, and total
  milliseconds;
- requested limit and per-lane acquisition limits;
- candidates returned, filtered, fused, expanded, hydrated, missing, reranked,
  and emitted;
- backend names and index epochs;
- degradation/fallback reasons;
- ANN settings and number of overfetch/iterative scan rounds.

Use structured data returned to an application-owned sink. Logging is a
fallback, not the primary metrics API.

## 18. Suggested atomic commit sequence

The following is a guide, not permission to implement all milestones in one
change. Before each milestone, agree on its exact commit plan and keep every
commit independently testable.

The implementation sequence through legacy-pipeline removal is landed in the
current history. This list is retained as an audit trail, not pending work.

1. `fix(eval): use standard latency percentiles`
2. `feat(eval): support graded labeled golden entries`
3. `feat(eval): add repeated concurrent benchmark runs`
4. `feat(eval): report relevance and latency slices`
5. `feat(local): add record FTS5 schema migration`
6. `perf(local): route keyword search through FTS5`
7. `test(local): add keyword scale and filter contracts`
8. `feat(local): add binary normalized vector storage`
9. `perf(local): add exact vector snapshot search`
10. `feat(local): add optional FAISS vector engine`
11. `test(local): gate ANN recall and storage size`
12. `feat(ports): add batch hydration capability`
13. `feat(ports): add batch graph capability`
14. `perf(search): overlap independent retrieval work`
15. `perf(search): batch graph expansion and hydration`
16. `perf(search): precompute provenance rank maps`
17. `feat(cache): support async query embedding coalescing`
18. `perf(search): cache canonical query embeddings`
19. `feat(cache): add lane-epoch candidate caching`
20. `fix(local): make graph and record epochs transactional`
21. `feat(search): add canonical query plans`
22. `feat(search): add weighted reciprocal rank fusion`
23. `perf(search): add lexical fast path`
24. `feat(search): add bounded conditional expansion`
25. `perf(pgvector): enable filtered iterative ANN scans`
26. `fix(pgvector): apply canonical vector filters`
27. `perf(pgvector): reuse stored and query embeddings`
28. `refactor(storage): move chunk hashes into SQLite`
29. `fix(storage): enforce graph edge lifecycle`
30. `feat(search): add optional source diversity policy`
31. `feat(search): add hierarchical record retrieval`
32. `refactor(search): migrate remaining legacy stages`
33. `refactor(search): remove deprecated chunk pipeline`

Tests that define a changed contract belong with the contract-changing commit.
Large benchmark fixtures or additive test suites can be separate follow-up test
commits.

## 19. Rollout and rollback strategy

Use feature flags or backend configuration for changes that alter retrieval:

- `local_keyword_engine = scan|fts5` during migration;
- `local_vector_engine = exact|faiss|auto`;
- `query_routing_enabled`;
- `weighted_rrf_enabled`;
- `candidate_cache_enabled`;
- `pgvector_iterative_scan_enabled`;
- `source_diversity_enabled`;
- `hierarchical_retrieval_enabled`.

For each rollout:

1. build/migrate the new index without deleting the old state;
2. run shadow queries and compare identity overlap, rank changes, failures, and
   latency;
3. enable for a small application/workspace slice;
4. monitor cache correctness, missing hydration, empty results, and p95/p99;
5. make the new path default only after gates pass;
6. retain a simple rollback until the next durable index snapshot is verified;
7. delete old state only in a later explicit migration.

Search fallbacks must be visible in diagnostics. Silent fallback can hide an
index that is permanently corrupt or never used.

## 20. Completion status and remaining validation

The code and contract portion of the definition of done is satisfied:

- local keyword search uses FTS5 rather than a table scan;
- local vectors are compact, normalized once, and searched through exact or
  optional ANN engines;
- filters are applied before expensive scoring wherever possible;
- record hydration and graph lookup are batched;
- independent retrieval work overlaps safely;
- query embedding and candidate caches use stable keys and lane epochs;
- query routing, fusion, reranking, diversity, and hierarchical retrieval are
  implemented behind the canonical record path;
- pgvector filtered retrieval, indexed storage, and graph lifecycle contracts
  are covered by current code and tests;
- one canonical record pipeline remains;
- default installation remains SQLite/NumPy-only and optional integrations stay
  optional.

The benchmark gate itself remains an evidence task rather than a completed
performance claim. The harness supports relevance, latency, throughput,
memory, disk-size, and optional ANN-recall checks, but the committed artifacts
currently provide only the 1k synthetic baseline described above. A future
scale-gate report must be produced before making comparative 1k/10k/100k,
ANN, or production-performance claims.

## 21. Baseline verification from the review

The focused pre-change baseline passed:

```text
56 passed in 3.51s
```

Covered suites:

- `tests/unit/search/test_record_pipeline.py`;
- `tests/unit/test_local_record_backend.py`;
- `tests/unit/test_query_embedding_cache.py`;
- `tests/unit/test_eval_runner.py`;
- `tests/unit/test_eval_metrics.py`.

This is a correctness baseline, not a performance result. Milestone 0 must
produce the first reproducible performance and relevance baseline before any
new default algorithm is selected.
