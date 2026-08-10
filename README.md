# `andnp-searchkernel`

`andnp-searchkernel` is a source-agnostic Python library for building hybrid
keyword, vector, and graph search systems. Applications map their native data
to canonical `Record` values, choose the storage and provider adapters they
need, and keep source-specific lifecycle logic outside the kernel.

The current release is `0.22.0`. It supports canonical record search,
checkpointed record ingestion, optional local/Postgres/FAISS/provider
integrations, and bounded federation across compatible search sources. The
API is still evolving before the first stable major release.

## Install

The core package supports Python 3.13 and newer:

```bash
pip install andnp-searchkernel
```

Install optional integrations only when you need them:

```bash
pip install "andnp-searchkernel[faiss]"
pip install "andnp-searchkernel[pgvector]"
pip install "andnp-searchkernel[pgvector-psycopg3]"
pip install "andnp-searchkernel[huggingface]"
pip install "andnp-searchkernel[ollama]"
pip install "andnp-searchkernel[markdown]"
```

## First local search

The local composition helper creates durable SQLite-backed stores. This small
example uses a deterministic provider so it can run without downloading a
model; replace it with a real provider for semantic retrieval.

```python
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from searchkernel import Record, build_local_record_kernel


class DemoEmbeddingProvider:
    model_name = "demo"
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


async def main() -> None:
    timestamp = datetime.now(UTC)
    composition = build_local_record_kernel(
        Path("records.db"),
        embedding_provider=DemoEmbeddingProvider(),
    )
    record = Record(
        workspace_id="demo",
        source_kind="notes",
        source_id="welcome",
        title="Welcome",
        body="Canonical records can be searched locally.",
        created_at=timestamp,
        updated_at=timestamp,
    )
    composition.keyword_store.index([record])

    outcome = await composition.kernel.search("canonical records", limit=5)
    for result in outcome.results:
        print(result.record.title, result.score)


asyncio.run(main())
```

The result is a `RecordSearchOutcome`. Each result retains the complete
workspace/source identity and search provenance; degraded execution is
reported through `outcome.failures` and `outcome.diagnostics`.

## Documentation

Start with the [documentation map](docs/index.md) to choose the right guide:

- [Getting started](docs/getting-started.md) — install the package and build a
  first local index.
- [Core concepts](docs/concepts.md) — records, identity, ingestion, querying,
  stores, providers, and readiness.
- [Federated search](docs/guides/federated-search.md) — combine local or HTTP
  search sources with bounded concurrency and explicit partial-result
  diagnostics.
- [Performance and retrieval roadmap](docs/search-performance-roadmap.md) —
  developer-facing validation evidence, constraints, and future work.

## Optional integrations

The core import surface does not require optional providers or backends. The
available extras are `faiss`, `pgvector`, `pgvector-psycopg3`, `huggingface`,
`ollama`, and `markdown`. See the [getting-started guide](docs/getting-started.md)
for the selection rule and the [federation guide](docs/guides/federated-search.md)
for the HTTP source adapter.

## Validation and releases

CI runs Ruff, Pyrefly, import-linter, the safe test suite, supported Python
versions, and selected optional-import checks. Pgvector integration tests need
Docker or `SEARCHKERNEL_PG_DSN`; real-embedding tests are outside the default
offline gate. The performance roadmap records what those checks do and do not
prove.

Merges to `main` with release-worthy Conventional Commits are released by
the repository workflows. The package and runtime `__version__` are checked
against each other in CI and before semantic release.

## License

MIT. See [LICENSE](LICENSE).
