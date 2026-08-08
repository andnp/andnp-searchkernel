# End-to-end checks

The public local restart journey is deterministic and offline. It uses the
SQLite-backed `build_local_record_kernel` API with an in-test embedding
provider, so it does not require Docker, Postgres, a model download, or a
network service.

Run it with:

```bash
uv run pytest tests/e2e -m e2e
```

Postgres-backed checks are integration tests and are separately gated. They
use `SEARCHKERNEL_PG_DSN` when supplied; otherwise the integration fixture
attempts to start a Docker `pgvector/pgvector:pg17` container and skips with a
visible reason when Docker is unavailable.

Real embedding tests are also opt-in through the `real_embeddings` marker and
require the documented local model cache. The deterministic e2e journey is
the readiness smoke check for environments without those optional services.
