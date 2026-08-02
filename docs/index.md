# Searchkernel documentation

This documentation is split by reader and task so the project README can stay
short.

## For application users

- [Getting started](getting-started.md) — install the package, create a local
  composition, index records, and run a query.
- [Core concepts](concepts.md) — understand record identity, source adapters,
  ingestion, search outcomes, and optional integrations.
- [Federated search](guides/federated-search.md) — combine compatible local or
  HTTP search sources and handle bounded partial results.

## For contributors and operators

- [Search performance and retrieval roadmap](search-performance-roadmap.md) —
  current validation evidence, invariants, performance limits, and future
  engineering work.

## Version scope

These pages describe the `0.6.0` API. The supported query model is
record-oriented: source adapters produce `Record` values, stores return
complete `RecordIdentity` values, and query results are
`RecordSearchOutcome` values. The package is still pre-1.0, so public APIs can
change between minor releases.

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
