# FAISS local vector search

`FAISSLocalVectorStore` is an optional local vector store. Install the
`faiss` extra before constructing it:

```bash
pip install "andnp-searchkernel[faiss]"
```

## Search strategies

The strategy is explicit at construction time:

```python
store = FAISSLocalVectorStore(
    backend,
    index_path=Path(".searchkernel/faiss"),
    search_strategy="exact",
    hnsw_m=32,
    hnsw_ef_construction=40,
    hnsw_ef_search=16,
    overfetch_multiplier=4.0,
    max_scan_rounds=4,
    max_scan_candidates=100_000,
)
```

The HNSW settings apply to `approximate` indexes. All settings are validated
at construction time and are included in the persisted configuration
fingerprint. Changing any setting causes the persisted index to be rebuilt.

`exact` is the default. It uses FAISS inner-product flat search, scans the
complete indexed corpus, applies the canonical vector filters to the stored
candidate metadata, and sorts by descending score with the storage key as the
deterministic tie-breaker. Filtered exact search therefore returns the same
eligible top-k set as the local exact vector backend, subject to normal
floating-point score representation.

`approximate` uses an HNSW index and bounded scan expansion. It is a latency
option: `overfetch_multiplier` controls the initial scan size,
`max_scan_rounds` bounds expansion, and `max_scan_candidates` is a hard cap on
the total candidates inspected for one query. Approximate search can miss
eligible candidates or return fewer than `k` results even when more eligible
records exist. Use `verify_recall(...)` against the local exact backend to
measure the observed overlap for a representative workload.

The strategy and complete ANN configuration are stored with a persisted index.
Reopening an index with a different strategy or ANN setting causes it to be
rebuilt rather than treating an incompatible index as reusable. Corrupt
persisted indexes are rebuilt from the durable local vectors. If FAISS
execution itself fails, the store falls back to the local exact vector backend
and reports the fallback reason in `last_search_diagnostics`.

Filters are evaluated from metadata captured when the FAISS state is built;
search does not issue one SQLite validation query per candidate. A vector
epoch change invalidates the cached state, so updates and deletes are not
served from an old candidate snapshot. Every search exposes execution details
through `last_search_diagnostics`, including scan rounds, candidate budget,
under-return, persistence rebuilds, and exact-fallback status.
