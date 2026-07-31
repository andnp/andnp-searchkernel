# `andnp-searchkernel`

A domain-agnostic search/indexing kernel for building hybrid vector + keyword + graph search systems with pluggable embedding, LLM, and reranker providers.

## Status

**Pre-alpha, extraction in progress.** This library is being extracted from [`mcp-markdown-ragdocs`](https://github.com/andnp/mcp-markdown-ragdocs) to enable reuse across arbitrary content sources and search backends.

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
