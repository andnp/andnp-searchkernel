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
MMR today through `post_process`. What the library does not ship is a reference
implementation.

One correction, found by building against these hooks rather than by reading
them. This document originally claimed HyDE was equally expressible through
`query_expander`. It is not. `_normalize_query_expansion` keeps only the first
`synonym_expansion_max_terms` words of a returned string — three by default —
and the expanded query feeds `_candidate_acquirer.keyword` alone, never a
re-embedding. So the hook carries *synonym-shaped* expansion, and a
hypothetical answer document arrives truncated to a few words in the lexical
lane, which is not HyDE in any meaningful sense. Real HyDE needs the expanded
query to reach the vector lane, which no current hook allows. What ships is `hypothetical_answer_expander`, named for what it does rather
than for HyDE, because a name promising retrieval behaviour the hook cannot
deliver is worse than no battery at all.

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

## Make the evaluation harness able to settle arguments

`eval/` already has `recall@k`, `nDCG@k`, `MRR`, and average precision, plus
golden-corpus and gate machinery. What it cannot do is tell a real improvement
from noise: there is no per-query paired comparison and no significance test.

Most algorithms below change ranking, and the size of the golden set decides
whether a result means anything. At ~50 queries, a 1–3 point nDCG move is
entirely consistent with chance; at ~500 queries with a consistent direction
across them, the same move is solidly significant. Without paired statistics
there is no way to tell those two situations apart, and each proposal becomes an
argument about plausibility rather than evidence.

**This gates ranking changes, not everything.** Behavior-preserving refactors
and pure-composition batteries — decomposing `async_search`, splitting the
backend, `CascadingReranker` — alter no ranking and should not wait on it.
Applying the gate to them is process friction, not rigor.

- **Plugs into** `eval/`, no new ports. Extend `runner.py` to retain per-query
  scores, add a paired bootstrap or permutation test and an effect size.
- **Stance** Opinionated. One canonical harness.
- **Cost** ~200 lines.
- **Risk** Low.

Do this before any ranking change, and alongside — not before — the structural
refactors, which need no evidence to justify themselves.

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
`P(relevant | score)`.

One caveat that matters more than it first appears. PAVA is non-decreasing, so
it can never *invert* two scores — but it is piecewise constant, so it can
**tie** scores that were previously distinct. Within a single lane that is
harmless if ties fall back to the raw score. Across lanes it is not: a plateau
in one lane hands every tie-break inside that range to the other lane, quietly
shifting the fusion balance. Either interpolate linearly between block
midpoints, or retain the raw score as an explicit secondary sort key.

- **Plugs into** Open question. A `LaneCalibrator` port with
  `confidence(lane, raw) -> float` is one option; a policy callable is the other,
  and a one-method Protocol is barely more than a callable. The argument for a
  port is lifecycle rather than signature: a fitted calibrator has state that
  must be fitted, persisted beside the index, and invalidated when the corpus
  changes — which a bare callable has nowhere to put. Decide when building it.
  Either way, ship `SaturatingCalibrator` (current behavior, needs no data,
  stays the default) and `IsotonicCalibrator` (fitted).
- **Stance** The *contract* is canonical — every lane reports `[0, 1]`
  confidence, and thresholds are expressed in those units. The *fitting* is
  pluggable, because it depends on having labels for your corpus.
- **Cost** ~150 lines. PAVA is ~40 lines of numpy; no scikit-learn dependency
  needed.
- **Risk** Overfits small label sets badly. Needs an explicit sample-size gate —
  on the order of 50 positive and 50 negative examples per lane — falling back to
  the parametric calibrator below it, rather than fitting whatever is available.
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
for d in R \ S:  MMR(d) = λ · Rel(q, d) − (1 − λ) · max Sim(d, dⱼ)
                                                    dⱼ ∈ S
```

**Both terms must be on the same scale**, or λ means nothing. This is the same
trap the score-semantics work just closed: if `Rel` is a raw RRF score (~0.02)
and `Sim` is cosine similarity (~0.8), the diversity penalty swamps relevance at
every λ below about 0.98, and the knob appears inert until it suddenly isn't.
Feed MMR the calibrated `[0, 1]` confidence, not the fused score.

- **Plugs into** `RecordSearchPolicy.post_process`, which already exists. Ship a
  factory — `mmr_post_process(embedding_of, lambda_=0.7)` — returning a callable
  that satisfies the existing hook. No new port required.
- **Stance** Pluggable, off by default. `λ` is genuinely domain-dependent: a
  documentation corpus wants diversity, a code-symbol lookup usually does not.
- **Cost** ~80 lines. The candidate-similarity matrix is one normalized matmul
  rather than the removed `dedup` module's pairwise Python loop; the greedy
  selection itself is still a loop over `k`, which is fine at result-set sizes.
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
the survivors with the original `float32` vectors. Memory drops 32× exactly
(a 384-dim `float32` vector is 1536 bytes; packed, 48).

That **moves the block-scan cliff out by 32×; it does not remove it.** A corpus
large enough still falls off the snapshot into the same unindexed sequential
scan. Fixing that pathology properly means giving the fallback an actual index
(IVF or HNSW over the packed vectors), which is separate work. Worth being
precise about, because "quantization fixes our scale problem" is the kind of
claim that gets believed until the corpus doubles again.

`numpy >= 2.0` is already required, so `np.packbits` and `np.bitwise_count` are
available without a compiled extension.

- **Plugs into** The `vector_engine` argument on `build_local_record_kernel`,
  which already accepts `"exact"` and `"faiss"`. Add `"binary"`.
- **Stance** Pluggable. `"exact"` stays the default — it is correct and fast
  enough at small scale, and a library should not silently trade recall for
  speed.
- **Cost** ~250 lines, plus a rescore oversampling knob (default ~10× the
  requested `k`). One implementation note: the obvious
  `np.bitwise_count(matrix ^ query).sum(axis=-1)` allocates an `(N, dim/8)`
  temporary on every query. Reuse a preallocated buffer via the `out=` argument.
- **Risk** Recall loss, which is the whole tradeoff. Must be measured, not
  assumed.
- **Evidence** `recall@10` against the exact engine on a fixed corpus, with
  memory and p99 latency alongside. Publish the recall cost in the docs so the
  choice is informed.

Easier to land cleanly after the `LocalRecordBackend` decomposition, which
gives the vector engine a real seam instead of a method on a 2,900-line class.

## Query understanding

### Query expansion batteries

**Shipped** as `searchkernel/search/expansion.py`: `synonym_expander` (no model
required) and `hypothetical_answer_expander` (any prompt-in/text-out callable, so
no LLM client becomes a dependency). Both satisfy the existing `query_expander` hook and
neither is wired on by default.

Building them exposed a limit of the hook that this document had assumed away.
`_normalize_query_expansion` keeps only the first `synonym_expansion_max_terms`
words of a returned string — three by default — and the result is appended to
the query and sent to `_candidate_acquirer.keyword`, never re-embedded. The hook
is therefore shaped for synonyms, not for rewriting.

- **Consequence for HyDE** A generated answer document reaches the lexical lane
  as a handful of words. That is not useless, but it is not HyDE. Delivering the
  real thing requires the expanded query to reach the vector lane — a pipeline
  change, not a battery. Raising `synonym_expansion_max_terms` widens the
  lexical half but cannot supply the embedding half.
- **Stance** Pluggable, off by default. `expansion_timeout_s` bounds latency,
  and the pipeline already declines to expand artifact-shaped queries.
- **Remaining work** Decide whether the vector lane should accept an expanded
  query at all. That is the actual HyDE task, and it is not small.

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
   pluggable: the current ladder as `ArtifactAwareScorer`, or a learned-sparse
   representation stored in an integer postings table.

**Learned sparse is closer than it looks.** SPLADE-style models
(`naver/splade-v3`, `bge-m3` sparse) are pre-trained and zero-shot — used
exactly like a dense encoder, no labels required. They produce per-term weights
that go into a `(term, storage_key, weight)` table and score by integer dot
product, which SQLite does natively. Grouping them with learning-to-rank was a
category error: LTR genuinely needs labeled training data for *your* corpus,
learned sparse does not. They belong in different tiers of this plan, and the
sparse option deserves evaluating well before any LTR work.

- **Cost** (1) ~150 lines of movement. (2) Learned sparse: a new keyword-store
  adapter plus an ingestion-time encoder, moderate. LTR: large, needs labels.
- **Risk** (1) is behavior-preserving and well covered by existing tests.
  (2) Learned sparse adds a model dependency at ingest and grows the index;
  measure index size before committing.

**Field boosting.** Moving the heuristics out leaves consumers with no
first-class way to say "title matters 3× more than body" — today that weighting
is baked into the `bm25(fts, 5.0, 1.0, 4.0, 2.0)` call. Whatever replaces the
ladder should expose per-field weights as configuration, or the move trades a
bad abstraction for a missing one.

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

- **Plugs into** An additive capability marker following the
  `CandidateFilterSupport` idiom — `SpanEmbeddingSupport` with
  `embed_spans(text, spans) -> list[Vector]`. Providers that cannot do it simply
  do not declare it. This is a smaller change than it first appeared: the
  existing `EmbeddingProvider` contract is untouched, not rewritten.
- **Stance** Pluggable, dependent on model support.
- **Cost** ~300 lines.
- **Risk** Model support varies and document length is capped by the backbone's
  context window (typically 8k tokens), so long documents still need splitting.
- **Verdict** Revised — worth doing, and arguably *before* contextual prefixes.

The cost comparison is lopsided in a way the first draft of this document got
backwards. Contextual prefixes need one **generative** LLM call per chunk: a
10k-document corpus at ~10 chunks each is 100k generations, which is hours of
local GPU time or real API spend, repeated whenever content changes. Late
chunking needs one **embedding forward pass per document** — work the pipeline
already does — plus mean-pooling over token spans, which is numpy slicing.

Same goal, one costs a model inference budget and the other costs a protocol
extension. The port-contract objection was doing more argumentative work than it
could support.

## Asymmetric query and document embedding

Modern embedding models are asymmetric: queries take an instruction prefix,
documents do not. Getting this wrong does not fail loudly — it silently
misaligns query and document representations and costs recall that looks like
a bad model rather than a bad call.

The `EmbeddingProvider` port has no query variant. `embed()` is documented as
embedding documents, and the asymmetry currently lives inside the HuggingFace
adapter, which says so directly in a comment: *"The EmbeddingProvider port
itself has no query variant yet — that asymmetry lives here."* Any provider
written against the port without reading that adapter will get it wrong.

- **Plugs into** The `EmbeddingProvider` port, by formalizing `embed_query` next
  to `embed`. `AsyncEmbeddingProvider` already has `embed_query`, so the sync
  and async boundaries currently disagree with each other.
- **Stance** Opinionated. This is a contract defect, not a choice — a provider
  should not be able to satisfy the port while silently embedding queries as
  documents.
- **Cost** ~60 lines plus adapter updates.
- **Risk** Touches a public port; existing implementations need a default.

Not an algorithm, and worth doing before any retrieval-quality measurement:
otherwise a recall number may be measuring this instead of whatever is under
test.

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

Sequenced so each step makes the next cheaper or more measurable. Work that
changes no ranking comes first, because it needs no evidence to justify it.

**Ungated — no ranking change, start immediately**

1. **Decompose `async_search`** — gives every later stage a clean insertion
   point instead of another conditional.
2. **Split `LocalRecordBackend`** — gives the vector engine a real seam.
3. **Cascaded reranking** — pure composition, no new extension point.

**Foundations for judging anything**

4. **Fix asymmetric query embedding** — otherwise later recall numbers may be
   measuring this defect.
5. **Evaluation harness with significance testing** — gates everything below.

**Gated on evidence**

6. **MMR diversity battery** — fills an existing socket; feed it calibrated
   confidence, not fused score.
7. **Isotonic calibrator** — extends the confidence contract already in place.
8. **Move lexical heuristics out of the store**, with field weights as
   configuration.
9. **Binary vector engine** — and be honest that it moves the scale cliff
   rather than removing it.
10. **Query expansion batteries**.
11. **Learned sparse retrieval** — zero-shot, so reachable once the scorer seam
    from step 8 exists.
12. **Late chunking**, then **token-aware chunking**.
13. **PPR graph expansion** — value depends entirely on graph richness.
14. **Contextual prefixes** — only if late chunking leaves a gap, given the
    ingestion cost.

Steps 1–7 are the highest confidence-to-cost. Steps 13–14 deserve a corpus whose
measured behavior justifies them.

## Considered and not proposed

**ColBERT / PLAID late interaction.** Per-token vectors with MaxSim scoring
genuinely address what single-vector embeddings lose — exact symbol names,
version strings, configuration flags — and there is a fair argument that the
lexical heuristic ladder exists precisely to bandage that weakness. It is
excluded here on cost, not merit: it multiplies index size by roughly the token
count per chunk, needs a centroid-plus-residual index to be tractable, and would
be the largest single addition in this document.

Learned sparse retrieval (step 11) targets the same weakness at a fraction of
the cost and fits the existing inverted-index storage. If that lands and
token-level matching is still the top failure mode in evaluation, revisit this
with evidence in hand.
