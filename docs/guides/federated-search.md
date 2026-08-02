# Federated search

Federation combines several independent search sources behind one bounded
query. Each source implements the versioned `SearchSource` contract, and
`FederationExecutor` calls eligible sources concurrently, fuses their ordered
hits, deduplicates equivalent results, and reports degradation explicitly.

Use federation when source systems own their own indexes or authorization
boundaries. Use the local record pipeline when one application owns the
stores and ingestion lifecycle.

## Compose an executor

An in-process source needs two methods:

```python
class SearchSource(Protocol):
    async def search(self, request: SearchRequest) -> SearchResponse: ...

    def capabilities(self) -> SourceCapabilities: ...
```

Register each source with its stable `SourceIdentity`:

```python
import asyncio

from searchkernel import (
    FederationConfig,
    FederationExecutor,
    RegisteredSearchSource,
    SearchRequest,
    SourceIdentity,
)
from searchkernel.adapters.federation import HttpSearchSource


async def main() -> None:
    notes = HttpSearchSource(
        "https://notes.example",
        SourceIdentity(
            source_kind="notes",
            source_id="primary",
            workspace_id="team-a",
        ),
        headers={"Authorization": "Bearer ..."},
    )
    executor = FederationExecutor(
        [RegisteredSearchSource(notes.source_identity, notes)],
        config=FederationConfig(
            max_concurrency=8,
            per_source_timeout_s=5.0,
        ),
    )

    response = await executor.search(
        SearchRequest(
            "incident review",
            top_k=10,
            source_selection=("notes",),
            request_id="request-123",
            trace_id="trace-123",
        )
    )
    for hit in response.hits:
        print(hit.title, hit.identity, hit.provenance)
    if response.degraded:
        for diagnostic in response.degradations:
            print(diagnostic.status, diagnostic.source, diagnostic.message)


asyncio.run(main())
```

`source_selection` matches a registered source's `source_kind`, `source_id`,
or `workspace_id`. An empty selection queries every registered source. A
source can also be registered directly when it exposes `identity` or
`source_identity`; `RegisteredSearchSource` makes the routing identity
explicit and is preferred for adapters.

## Request and response contracts

Federation contracts use `FEDERATION_CONTRACT_VERSION == "v1"`. The public
models are JSON-serializable with `to_dict()`, `from_dict()`, `to_json()`, and
`from_json()`.

`SearchRequest` contains:

- `query`, limited to 4,096 characters;
- `top_k`, from 1 through 1,000;
- JSON-compatible `filters` owned by the source contract;
- optional `source_selection` values;
- optional caller authorization context;
- optional absolute deadline and cancellation ID; and
- request and trace IDs for diagnostics.

Each `SearchResponse` identifies its source and returns ordered `SearchHit`
values. A hit includes source kind, source ID, title, snippet, local source
rank, optional workspace and URI, optional native score, lifecycle, metadata,
and provenance. `partial` and `warnings` let a source describe its own
degraded result.

The source's local rank is the input to federation rank fusion. Native scores
are retained as metadata but are not assumed to be comparable across sources.

## HTTP source adapter

`HttpSearchSource` is an async HTTP/JSON implementation of `SearchSource`.
Install the adapter's HTTP dependency with:

```bash
python -m pip install andnp-searchkernel httpx
```

The adapter imports `httpx` only when it is used, so applications that already
manage `httpx` can install it through their own dependency set. The package's
`ollama` extra also provides `httpx`, but it is intended for the Ollama
embedding adapter rather than federation specifically.

The source service exposes these v1 endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/search/capabilities` | Advertise contract features and bounds |
| `GET` | `/v1/health` | Return a JSON health document |
| `POST` | `/v1/search` | Accept a serialized `SearchRequest` and return `SearchResponse` |

The adapter validates the contract version, response source identity, JSON
schema, request/response sizes, and timezone-aware deadlines. It forwards
`X-Request-ID` and `X-Trace-ID` when those fields are present. Authentication
is supplied through the adapter's `headers`; authorization decisions remain
owned by the source service and its `CallerAuthorizationContext` handling.

The default HTTP bounds are:

- 5 seconds per request;
- 256 KiB maximum request body; and
- 4 MiB maximum response body.

Override them in `HttpSearchSource(...)` when the source contract and
deployment limits justify different values.

## Capabilities and source selection

Before searching, the executor checks that a source advertises `v1`. If a
request has filters, the source must advertise `supports_filters`; unsupported
sources are skipped with an `unavailable` diagnostic rather than receiving a
request they cannot honor.

`SourceCapabilities` also reports whether a source supports source selection,
rerank text, partial results, and cancellation, plus its maximum `top_k` and
rerank text length. The executor lowers each source request to the source's
advertised maximum and to `FederationConfig.per_source_top_k` when configured.

## Bounds, fusion, and degradation

`FederationConfig` defaults to eight concurrent source calls and a five-second
per-source timeout. A caller deadline can shorten that timeout. A failed or
timed-out source does not discard successful results from other sources.

The response reports:

- `hits`: fused, deduplicated hits limited to the requested `top_k`;
- `source_responses`: responses that completed successfully;
- `partial`: whether any source or optional reranker degraded;
- `degradations`: typed `FederationDiagnostic` values; and
- `warnings`: source warnings plus diagnostic summaries.

Fusion uses reciprocal rank with deterministic tie-breaking. Equivalent hits
are deduplicated by canonical record identity and normalized URI where
available. Optional reranking is limited to the configured candidate count
and text length; reranker failures are returned as `rerank` diagnostics.

Cancellation of the caller's federation task cancels outstanding source tasks
and is re-raised. This is different from a source timeout, which is isolated
and returned as a timeout diagnostic.

## Implementing a source service

This repository defines the transport-neutral contracts and the client-side
HTTP adapter. It does not provide a generic HTTP server or authorization
middleware. A source service should:

1. map its native results to complete `SearchHit` identities;
2. return local ranks starting at 1 and stable source metadata;
3. advertise only capabilities it can enforce;
4. honor request deadlines and cancellation where supported; and
5. return `partial` and `warnings` instead of silently omitting degraded
   behavior.

Keep source-specific filtering, authorization, index epochs, and lifecycle
rules on the source side. The federator can route and fuse results, but it
cannot infer those rules from opaque metadata.
