# Retrieval algorithm design notes

Design notes for the retrieval, ranking, and chunking algorithms searchkernel
could offer. Each note states what the algorithm changes, where it plugs into
the existing architecture, whether it should be opinionated or pluggable, what
it costs, and what evidence would show it worked.

Nothing here is committed work. These are proposals to choose from.

## Design position

Searchkernel is a building-block library. It should be opinionated where a
second live implementation would produce inconsistent identity, provenance, or
diagnostics — and pluggable where the right answer depends on the application
domain, the corpus, or the model.

That position is already expressed in the codebase through four extension
idioms. New algorithms should extend these rather than invent a fifth.

| Idiom | Use it for | Existing examples |
| --- | --- | --- |
| **Port** (`Protocol`) | Swappable infrastructure or models, where implementations differ in their dependencies | `EmbeddingProvider`, `Reranker`, `VectorStore`, `KeywordStore`, `GraphStore`, `ChunkingStrategy` |
| **Policy callable** (`RecordSearchPolicy`) | Decisions that depend on application-domain knowledge the kernel cannot have | `candidate_filter`, `score_adjuster`, `query_expander`, `post_process` |
| **Config knob** (`RecordSearchConfig`) | Tuning a single canonical algorithm with a scalar that depends on the corpus | `rrf_k`, `keyword_saturation_k`, `graph_depth` |
| **Capability marker** | Optional adapter features, discovered structurally | `CandidateFilterSupport` |

### Sockets exist; batteries mostly do not

`RecordSearchPolicy` already carries fourteen hooks. A consumer can implement
MMR today through `post_process`, or HyDE through `query_expander`. What the
library does not ship is a reference implementation of either.

This is the useful lesson from the modules removed in the cleanup — `dedup`,
`calibration`, `community`, `variance`. Those were batteries written without
sockets: never referenced by the pipeline, never exported, never reachable by a
consumer. Unreachable code is not a building block. A building block is an
implementation behind a named extension point, exported and documented.

Most of what follows is therefore "ship a battery for a socket that exists",
not "add a stage to the pipeline".

### The default stance

Every pluggable algorithm below ships **off** unless it is strictly better than
what it replaces. Silent reordering of results is a bad surprise in a library
whose current selling point is deterministic, explainable retrieval. Opting in
is one argument at composition time.

## Prerequisite: make the evaluation harness able to settle arguments

`eval/` already has `recall@k`, `nDCG@k`, `MRR`, and average precision, plus
golden-corpus and gate machinery. What it cannot do is tell a real improvement
from noise: there is no per-query paired comparison and no significance test.

Every algorithm below changes ranking. On a golden set of realistic size, a
1–3 point nDCG move is entirely consistent with chance. Without paired
statistics, each proposal becomes an argument about plausibility rather than
evidence — which is the failure mode these notes exist to avoid.

- **Plugs into** `eval/`, no new ports. Extend `runner.py` to retain per-query
  scores, add a paired bootstrap or permutation test and an effect size.
- **Stance** Opinionated. One canonical harness.
- **Cost** ~200 lines.
- **Risk** Low.

Recommend doing this first regardless of what else is chosen.

## Ranking and fusion

### Isotonic lane calibration

The `lane_confidence` module added in the score-semantics work maps each lane's
raw score onto `[0, 1]` with fixed transfer functions — cosine rescaled,
lexical saturated at `raw / (raw + k)`. That is a *parametric* calibrator with
one tunable constant, and it was a large improvement over comparing raw BM25 to
an absolute threshold. It is still a guess about the shape of the curve.

Isotonic regression fits the actual curve from labeled data: given
`(raw_score, relevant)` pairs from the golden corpus, pool-adjacent-violators
produces the monotone step function that best maps raw score to
`P(relevant | score)`. Monotonicity is guaranteed by construction, so within-lane
ranking can never be altered — the same invariant the current transfer functions
hold.

- **Plugs into** A new `LaneCalibrator` port with `confidence(lane, raw) -> float`.
  Ship `SaturatingCalibrator` (current behavior, needs no data, stays the default)
  and `IsotonicCalibrator` (fitted, persists its curve alongside the index).
- **Stance** The *contract* is canonical — every lane reports `[0, 1]`
  confidence, and thresholds are expressed in those units. The *fitting* is
  pluggable, because it depends on having labels for your corpus.
- **Cost** ~150 lines. PAVA is ~40 lines of numpy; no scikit-learn dependency
  needed.
- **Risk** Overfits small label sets. Needs a minimum-sample guard that falls
  back to the parametric calibrator.
- **Evidence** Calibration error (Brier score or reliability diagram) against a
  held-out split, plus confirmation that ranking is unchanged within each lane.

This is the most natural follow-on from the score-semantics work and the best
fit for the building-block framing: same contract, better-informed
implementation, consumer chooses based on whether they have labels.

### Score-distribution fusion

Reciprocal rank fusion discards score margins entirely: a vector hit ten times
closer in embedding space contributes the same increment as one that barely
made the cut. `fusion_mode="calibrated"` exists as an alternative but uses
per-query min-max normalization, which is unstable for sparse lanes.

With a calibrator in place, a third mode becomes available: fuse the calibrated
confidences directly as a weighted convex combination. Margins survive, and the
values mean the same thing across queries.

- **Plugs into** `RecordSearchConfig.fusion_mode`, a third value.
- **Stance** Pluggable mode; `"rrf"` stays the default.
- **Cost** ~60 lines once a calibrator exists.
- **Risk** Sensitive to calibration quality — worth having only after the above.

### Diversity: MMR

Nothing currently prevents the top ten from being ten near-identical chunks of
one document. Maximal marginal relevance greedily trades relevance against
similarity to what is already selected:

```
MMR = argmax [ λ · Rel(q, d) − (1 − λ) · max Sim(d, dⱼ) ]
```

- **Plugs into** `RecordSearchPolicy.post_process`, which already exists. Ship a
  factory — `mmr_post_process(embedding_of, lambda_=0.7)` — returning a callable
  that satisfies the existing hook. No new port required.
- **Stance** Pluggable, off by default. `λ` is genuinely domain-dependent: a
  documentation corpus wants diversity, a code-symbol lookup usually does not.
- **Cost** ~80 lines. Vectorized with one normalized matmul, not the pairwise
  Python loop the removed `dedup` module used.
- **Risk** Low, and entirely opt-in.
- **Evidence** `nDCG@10` roughly flat while distinct-source count and subtopic
  recall rise. If nDCG drops materially, λ is wrong for that corpus.

### Cascaded reranking

Reranking is currently one `Reranker` and a `rerank_budget`. A cascade runs a
cheap cross-encoder over a wide candidate set and escalates to an expensive
generative reranker only for the top few, and only when the stage-one
confidence gap is narrow enough that the ordering is genuinely in doubt.

- **Plugs into** Nothing new. `CascadingReranker(fast, slow, escalate_when=…)`
  implements the existing `Reranker` protocol and composes two of them.
- **Stance** Pure composition, ships as a building block, changes no default.
- **Cost** ~80 lines.
- **Risk** Low.

The cheapest genuine win in these notes, and the clearest illustration of the
building-block philosophy: no new extension point, just an implementation that
composes existing ones.

## Vector retrieval

### Binary quantization with a float rescore cascade

Local vector search is an exact `float32` matmul over an in-memory snapshot.
Above the snapshot caps it falls back to a block scan that re-decodes every
stored vector on every query with no early termination — so cost grows with
corpus size exactly where an approximate index is needed.

The standard answer is a two-stage cascade: pack each embedding to one bit per
dimension by sign, rank a wide candidate set by Hamming distance, then rescore
the survivors with the original `float32` vectors. Memory drops roughly 32×,
which also means corpora that previously fell off the snapshot cliff now fit in
it — this fixes the block-scan pathology as a side effect, not just the
throughput.

`numpy >= 2.0` is already required, so `np.packbits` and `np.bitwise_count` are
available without a compiled extension.

- **Plugs into** The `vector_engine` argument on `build_local_record_kernel`,
  which already accepts `"exact"` and `"faiss"`. Add `"binary"`.
- **Stance** Pluggable. `"exact"` stays the default — it is correct and fast
  enough at small scale, and a library should not silently trade recall for
  speed.
- **Cost** ~250 lines, plus a rescore oversampling knob (default ~10× the
  requested `k`).
- **Risk** Recall loss, which is the whole tradeoff. Must be measured, not
  assumed.
- **Evidence** `recall@10` against the exact engine on a fixed corpus, with
  memory and p99 latency alongside. Publish the recall cost in the docs so the
  choice is informed.

Easier to land cleanly after the `LocalRecordBackend` decomposition, which
gives the vector engine a real seam instead of a method on a 2,900-line class.

## Query understanding

### Query expansion batteries

`RecordSearchPolicy.query_expander` exists and is unused by any shipped
implementation. Two batteries are worth having: a synonym expander needing no
model, and a HyDE expander that generates a hypothetical answer document
through the existing `LLM` port and embeds that instead of the raw query.

- **Plugs into** The existing hook and the existing `LLM` port.
- **Stance** Pluggable, off by default. `expansion_timeout_s` already exists to
  bound the latency.
- **Cost** ~100 lines.
- **Risk** Latency, already bounded. HyDE can hurt on keyword-shaped queries —
  the existing router already declines to expand artifact queries.

### Replacing the lexical heuristic ladder

`indices/keyword_scoring.py` is ~250 lines of hand-tuned constants (80, 24, 112,
120, 56 …) added directly onto BM25, where they outweigh it by one to two orders
of magnitude. It also encodes filesystem and markdown assumptions — path
separators, file extensions, `>` header breadcrumbs — inside the generic SQLite
store.

Two separable moves, and only the first is an algorithm choice at all:

1. **Architectural, do regardless.** Move the heuristics out of the storage
   layer and behind a policy-supplied scorer. The store becomes domain-neutral,
   which is what a source-agnostic kernel requires. This is really a Phase 5
   refactor.
2. **Algorithmic, optional.** Once there is a scorer seam, alternatives become
   pluggable: the current ladder as `ArtifactAwareScorer`, a learning-to-rank
   model trained on the golden corpus, or a learned-sparse (SPLADE-style)
   representation stored in an integer postings table.

- **Cost** (1) ~150 lines of movement. (2) Large, and needs training data.
- **Risk** (1) is behavior-preserving if done carefully and is well covered by
  existing tests. (2) is a research project.

## Graph retrieval

### Personalized PageRank expansion

Graph expansion is one hop with static per-edge-type discounts. Transitive
relationships — a caller of a caller, a document two links away — are invisible.
Personalized PageRank (random walk with restart, seeded from the retrieved
candidates) scores nodes by structural relevance to the whole seed set and
naturally reaches further.

- **Plugs into** A new `GraphExpansionStrategy` port. Ship
  `BoundedTypedExpansion` (current behavior, default) and
  `PersonalizedPageRankExpansion`.
- **Stance** Pluggable, one-hop stays the default: it is cheap and predictable,
  and multi-hop value depends entirely on how rich the consumer's graph is.
- **Cost** ~200 lines. Can run in Python over a bounded subgraph, or as a
  SQLite recursive CTE that keeps the walk in the storage engine.
- **Risk** Cost explodes on dense graphs. Needs hard caps on visited nodes and
  a score floor for early termination — non-negotiable, not tuning.

## Chunking

### Token-aware sizing

Chunk sizing is measured in characters. The number that actually matters is
tokens, and the ratio between them varies by roughly 4× between English prose,
CJK, and source code — so a fixed character budget systematically over-fills
some corpora and under-fills others relative to the embedding model's context.

- **Plugs into** An optional token-counter callable on the chunk config, or a
  `TokenAwareChunker` strategy behind the existing `ChunkingStrategy` port.
- **Stance** Pluggable. Character-based stays the default because token
  counting requires a tokenizer, which is model-specific — precisely the kind of
  choice that belongs to the consumer.
- **Cost** ~120 lines.
- **Risk** Low.

### Contextual chunk prefixes

Chunks currently carry a header breadcrumb (`Architecture > Storage`), which is
weak context for a chunk whose text says "it uses connection pooling". Passing
each chunk plus its document through a small model at ingestion time to generate
a one-sentence situating prefix, prepended before embedding, is the reported
large recall win in this area.

- **Plugs into** The existing `LLM` port and an ingestion-time enricher stage.
- **Stance** Pluggable, off by default. Costs one model call per chunk at
  ingest.
- **Cost** ~150 lines.
- **Risk** Ingestion cost and time scale with corpus size. Needs caching keyed
  by chunk content hash — which `Chunk.content_hash` already provides.

### Late chunking

Embed the whole document in one forward pass, then mean-pool token embeddings
per chunk span, so each chunk vector retains document-wide context without any
generation cost at ingest.

- **Plugs into** Requires a new capability — the `EmbeddingProvider` port
  returns one vector per text and cannot express per-token output. Would need a
  `TokenEmbeddingSupport` marker following the `CandidateFilterSupport` idiom.
- **Stance** Pluggable, and dependent on model support.
- **Cost** ~300 lines, plus real constraints on which models qualify.
- **Risk** Highest in these notes. Model support varies, document length is
  capped by the backbone's context, and the capability leaks into the embedding
  port's contract.
- **Verdict** Defer. Contextual prefixes reach a similar goal with fewer model
  constraints and no port change.

## Refactors that are not choices

These are behavior-preserving and provable, unlike everything above. They mostly
matter here because they decide how cleanly the algorithms can land.

- **Decompose `async_search`** — ~480 lines of imperative flow that re-fuses
  three times and mutates its working state throughout. Turning it into an
  ordered list of stages is what gives diversity, calibration, and expansion a
  clean insertion point instead of another conditional in the middle. Do this
  before any pipeline-stage algorithm.
- **Split `LocalRecordBackend`** — 2,900 lines covering schema, migrations,
  lexical search, vector search, graph traversal, and hydration. The vector
  engine seam it creates is what makes the binary engine a new class rather than
  another branch.
- **Async-first with an explicit sync wrapper** — the current
  `asyncio.get_running_loop()` probe is fragile under foreign event loops.

## Suggested order

Sequenced so each step makes the next cheaper or more measurable.

1. **Evaluation harness with significance testing** — otherwise nothing below
   can be judged.
2. **Decompose `async_search`** — unblocks clean stage insertion.
3. **Cascaded reranking** — pure composition, no new extension point, no
   default change.
4. **MMR diversity battery** — fills an existing socket.
5. **Isotonic calibrator** — extends the confidence contract already in place.
6. **Move lexical heuristics out of the store** — architectural debt with a
   known shape.
7. **Split `LocalRecordBackend`**, then the **binary vector engine**.
8. **Query expansion batteries**.
9. **PPR graph expansion**.
10. **Token-aware chunking**, then **contextual prefixes**.
11. **Late chunking** — revisit only if the earlier work leaves a gap it fills.

Items 1–5 are the highest confidence-to-cost ratio. Items 9–11 are worth
deferring until there is a corpus whose measured behavior justifies them.
