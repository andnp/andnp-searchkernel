# Getting started

This guide builds a small local index and runs a query without requiring an
external database or embedding service. It is intended to verify the package
installation and show the shape of the public API.

## 1. Install the package

Create an environment and install the core package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install andnp-searchkernel
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead.

The core package supports Python 3.13 and newer. Add an optional extra when
you need one of the provider or storage adapters:

```bash
python -m pip install "andnp-searchkernel[ollama]"
```

See [Core concepts](concepts.md#optional-integrations) for the complete extra
list.

## 2. Build a local composition

`build_local_record_kernel` creates SQLite-backed keyword, vector, and graph
stores. It requires an embedding provider because the same composition can
serve semantic queries. This example uses a deterministic provider so it has
no model download or service dependency:

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
    composition = build_local_record_kernel(
        Path("records.db"),
        embedding_provider=DemoEmbeddingProvider(),
    )
    timestamp = datetime.now(UTC)
    records = [
        Record(
            workspace_id="demo",
            source_kind="notes",
            source_id="searchkernel",
            title="Searchkernel",
            body="A source-agnostic search and indexing kernel.",
            created_at=timestamp,
            updated_at=timestamp,
        ),
        Record(
            workspace_id="demo",
            source_kind="notes",
            source_id="other",
            title="Another note",
            body="A second searchable record.",
            created_at=timestamp,
            updated_at=timestamp,
        ),
    ]

    # Direct store indexing is useful for a small local application or a
    # one-time bootstrap. Use SearchKernel.ingest_source for live sources.
    composition.keyword_store.index(records)

    outcome = await composition.kernel.search("search and indexing", limit=5)
    for result in outcome.results:
        print(result.record.source_id, result.score)


asyncio.run(main())
```

The SQLite database is created at `records.db`. The helper also exposes the
underlying stores through `composition.keyword_store`, `composition.vector_store`,
and `composition.graph_store` when an application needs explicit indexing or
maintenance operations.

## 3. Use a real embedding provider

The demo provider makes every vector identical, so it is only a wiring check.
For semantic retrieval, use a provider that returns one stable vector per
input and declares its `model_name` and `dim`.

For example, the built-in Ollama adapter uses a local Ollama daemon:

```bash
python -m pip install "andnp-searchkernel[ollama]"
```

```python
from searchkernel.api import OllamaEmbeddingProvider

embedding_provider = OllamaEmbeddingProvider("nomic-embed-text")
```

Pass that provider to `build_local_record_kernel`. The adapter may pull the
model automatically; configure `auto_pull=False` when model installation must
be managed outside the application. Close the provider when the application
stops, or use it as a context manager.

## 4. Move from direct indexing to source ingestion

Direct store indexing is deliberately small, but it does not discover source
changes or persist source checkpoints. For a live source, implement
`ContentSource` and provide a `RecordIngestor` to `SearchKernel.build`:

```python
from searchkernel import SearchKernel

kernel = SearchKernel.build(
    content_sources=[notes_source],
    ingestor=record_ingestor,
)

receipt = await kernel.ingest_source(
    "notes",
    batch_size=100,
    checkpoint_store=checkpoint_store,
    failure_mode="strict",
)
```

The source must yield `Record` values through an asynchronous iterator or
bounded `SourceBatch` iterator. A successful batch advances its checkpoint.
Strict mode raises `IngestionError` on a failed batch; lenient mode returns
successful and failed record outcomes and keeps later checkpoint advancement
bounded by the failed work.

For a complete adapter skeleton, see the
[custom content source guide](guides/custom-content-source.md).

## 5. Inspect results and failures

Search results are not anonymous dictionaries:

```python
outcome = await kernel.search(
    "search and indexing",
    filters={"statuses": ["active"]},
    limit=10,
)

for result in outcome.results:
    print(result.record.identity, result.score)

if outcome.degraded:
    for failure in outcome.failures:
        print(failure.stage, failure.message)
```

Use `RecordIdentity.storage_key` when persisting or correlating results. Do
not use a bare source ID as a cross-source or cross-workspace key.

## Next steps

- Read [Core concepts](concepts.md) for the identity and lifecycle model.
- Read [Federated search](guides/federated-search.md) to combine compatible
  search sources.
- Read the [performance and retrieval roadmap](search-performance-roadmap.md)
  before treating the test suite or synthetic benchmarks as production
  performance evidence.
