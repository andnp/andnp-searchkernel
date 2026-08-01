# `andnp-searchkernel`

A domain-agnostic search/indexing kernel for building hybrid vector + keyword + graph search systems with pluggable embedding, LLM, and reranker providers.

## Status

**Pre-alpha, extraction in progress.** This library is being extracted from [`mcp-markdown-ragdocs`](https://github.com/andnp/mcp-markdown-ragdocs) to enable reuse across arbitrary content sources and search backends.

## Optional backends

The core package provides the domain models, ports, record search pipeline, and
evaluation primitives. Install only the integrations required by an
application:

```bash
pip install andnp-searchkernel[pgvector,huggingface,markdown]
```

Available extras are `faiss`, `pgvector`, `huggingface`, and `markdown`.
FAISS and pgvector implement the same record-oriented backend contracts; they
can be selected independently or used together during migrations.

## Canonical search composition

Compose local search from the record-oriented ports:

```python
from searchkernel.api import SearchKernel

kernel = SearchKernel.build(
    record_hydrator=record_hydrator,
    keyword_store=keyword_store,
    vector_store=vector_store,
    graph_store=graph_store,
    embedding_provider=embedding_provider,
)
```

`SearchKernel.build` registers a canonical `SearchOrchestrator` for these
dependencies. Callers that already own one may pass `orchestrator=` instead.
The deprecated chunk-oriented execution path has been removed; migrate callers
to this record composition or use `LegacyLocalOrchestratorAdapter` only when
wrapping an existing legacy orchestrator.

## Integration tests

The pgvector integration tests automatically start a temporary
`pgvector/pgvector:pg17` Docker container when `SEARCHKERNEL_PG_DSN` is not
set. Docker must be running:

```bash
uv run pytest tests/integration
```

To use an existing PostgreSQL instance instead, set
`SEARCHKERNEL_PG_DSN` to its connection string. The database must allow the
`vector` extension to be created.

## Releases

Merges to `main` with `feat`, `fix`, or breaking Conventional Commits are
released automatically. The release workflow bumps the SemVer version,
updates `pyproject.toml` and `uv.lock`, pushes a `v*` tag, and dispatches the
PyPI publishing workflow. Documentation, chore, and test-only commits do not
create releases.

## License

MIT License. See [LICENSE](LICENSE) for details.
