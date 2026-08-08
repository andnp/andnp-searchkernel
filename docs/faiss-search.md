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
)
```

`exact` is the default. It uses FAISS inner-product flat search, scans the
complete indexed corpus, applies the canonical vector filters to the stored
candidate metadata, and sorts by descending score with the storage key as the
deterministic tie-breaker. Filtered exact search therefore returns the same
eligible top-k set as the local exact vector backend, subject to normal
floating-point score representation.

`approximate` uses an HNSW index and bounded scan expansion. It is a latency
option: `overfetch_multiplier` controls the initial scan size and
`max_scan_rounds` bounds expansion. Approximate search can miss eligible
candidates or return fewer than `k` results even when more eligible records
exist. Use `verify_recall(...)` against the local exact backend to measure the
observed overlap for a representative workload.

The strategy is stored with a persisted index. Reopening an index with a
different strategy causes it to be rebuilt rather than treating an exact
index as approximate or vice versa. Corrupt or unavailable optional indexes
fall back to the local exact vector backend.

Filters are evaluated from metadata captured when the FAISS state is built;
search does not issue one SQLite validation query per candidate. A vector
epoch change invalidates the cached state, so updates and deletes are not
served from an old candidate snapshot.
