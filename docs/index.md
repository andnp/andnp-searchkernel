# Searchkernel documentation

This documentation is split by reader and task so the project README can stay
short.

## For application users

- [Getting started](getting-started.md) — install the package, create a local
  composition, index records, and run a query.
- [End-to-end checks](e2e.md) — run the deterministic local restart journey and
  understand optional environment gates.
- [Core concepts](concepts.md) — understand record identity, source adapters,
  ingestion, search outcomes, and optional integrations.
- [Custom content sources](guides/custom-content-source.md) — adapt a native
  source to checkpointed record ingestion.
- [Lifecycle and ownership](lifecycle.md) — close local compositions and
  distinguish owned from injected resources.
- [Upgrading](upgrading.md) — breaking changes by release and what each one
  requires, including when a stored index must be rebuilt.
- [Retrieval evaluation](evaluation.md) — metric semantics, duplicate result
  reporting, and labeled corpus contracts.
- [Federated search](guides/federated-search.md) — combine compatible local or
  HTTP search sources and handle bounded partial results.

## For contributors and operators

- [Search performance and retrieval roadmap](search-performance-roadmap.md) —
  current validation evidence, invariants, performance limits, and future
  engineering work.
- [SearchKernel improvement design](searchkernel-improvement-design.md) —
  dependency-ordered design for indexing correctness, retrieval quality,
  backend parity, evaluation, and scale.
- [Retrieval algorithm design notes](retrieval-algorithm-design.md) — proposed
  ranking, vector, graph, and chunking algorithms, where each plugs into the
  existing extension points, and which should be opinionated or pluggable.

## Version scope

These pages describe the current `1.x` API. The supported query model is
record-oriented: source adapters produce `Record` values, stores return
complete `RecordIdentity` values, and query results are
`RecordSearchOutcome` values. Public APIs may change between minor releases
when the upgrading guide calls out a contract change.

## Where to look for an API detail

The stable application-facing imports are exposed from `searchkernel` and
`searchkernel.api`. The main composition choices are:

- `build_local_record_kernel(...)` for a durable local SQLite composition;
- `SearchKernel.build(...)` when an application owns its stores, source
  adapters, or ingestor;
- `FederationExecutor(...)` when an application owns several compatible
  `SearchSource` implementations; and
- `HttpSearchSource(...)` from `searchkernel.adapters.federation` for a v1
  HTTP/JSON source.

Use the focused guides for working examples. The implementation modules and
tests remain the reference for advanced adapter contracts that are not yet
covered by a formal API reference.
