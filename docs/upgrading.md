# Upgrading

Breaking changes by release, and what each one requires of a consumer.

## 0.25.0 to 1.0.0

Four contract changes. Two of them invalidate data already on disk; the kernel
detects that itself, but it cannot rebuild for you.

### Stored indexes must be rebuilt

Chunk records used to derive their identity from the parent's whole storage
key, so a chunk's key embedded the parent's serialized identity inside its own
and repeated the workspace and source kind:

```
before  record:["ws","notes","record:[\"ws\",\"notes\",\"doc-a\"]#chunk:0"]
after   record:["ws","notes","doc-a#chunk:0"]
```

Separately, header chunking no longer emits parent chunks alongside their
children. The original record already serves as the parent — chunks point at it
through `_chunk_parent_storage_key`, and the pipeline groups on that — so the
intermediate layer was indexed but never retrieved through, and it skewed the
lexical corpus statistics that BM25 depends on.

Both change what is written to storage, so an index built by an earlier version
cannot be served correctly.

**Only applications that opted into chunking are affected.** `RecordIngestor`
takes `chunker=None` by default and `build_local_record_kernel` wires no
chunker, so an application that never supplied one stores no chunk records and
needs no rebuild.

**Detecting it.** A local composition reports the condition:

```python
composition = build_local_record_kernel(db_path, embedding_provider=provider)
if composition.requires_rebuild:
    reindex_everything()          # your ingestion path
    composition.mark_rebuilt()
```

A stale store still opens and still answers queries — it is reported, not
refused, because refusing to open would be a worse failure than serving a
clearly flagged index. Nothing is deleted for you.

Manifest-based consumers need no code change: `CURRENT_MANIFEST_SPEC_VERSION`
moved to `2.0.0` and the existing `should_rebuild` check turns that into a
rebuild automatically.

Embedding revisions invalidate themselves, because
`record_embedding_revision` hashes the storage key — changed chunk keys
therefore produce changed revisions, and reconciliation re-embeds them without
any extra work.

### `EmbeddingProvider` now requires `embed_query`

The sync `EmbeddingProvider` protocol gained `embed_query(text) -> Vector`,
which `AsyncEmbeddingProvider` already had.

Most embedding models are asymmetric: a query takes an instruction prefix that
a document does not. Embedding a query as a document does not fail — it returns
a plausible vector pointed the wrong way relative to the document space, which
shows up as a mediocre model rather than as a mistake. The sync port had no way
to express the distinction, so a provider could satisfy the contract and still
get it wrong.

**If you implement `EmbeddingProvider`,** add `embed_query`. If your model is
symmetric, delegating is correct and explicit:

```python
def embed_query(self, text: str) -> Vector:
    return self.embed([text])[0]
```

`OllamaEmbeddingProvider` does exactly that by default, since the adapter
serves arbitrary models and cannot know which need a prefix. Pass
`query_prefix=` for one that does — `nomic-embed-text` wants
`search_query:` / `search_document:`.

### `ChunkTuningConfig` lost two fields

`parent_chunk_min_chars` and `parent_chunk_max_chars` are gone with the parent
chunk layer. Remove them from any config object you pass; no strategy reads
them.

### `SearchResultProvenance` lost two fields

`community_boost` and `project_uplift` are removed. Neither was ever assigned,
so both could only ever serialize as absent.

### Not breaking, but worth knowing

Retrieval thresholds are now expressed in calibrated confidence rather than raw
lane scores. `adaptive_graph_min_seed_score` and
`artifact_confidence_threshold` keep their `0.75` defaults, but those defaults
now mean something consistent across lanes: previously a raw BM25 score was
compared against the same number as a cosine similarity, so the gates fired on
nearly every keyword-bearing query. They are now genuinely selective.

If you tuned either threshold against the old behavior, retune it. The new
scale is documented by `searchkernel.search.lane_confidence`, and
`RecordSearchConfig.keyword_saturation_k` (default `10.0`) sets the raw lexical
score that reads as even odds.
