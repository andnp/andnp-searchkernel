"""Stable identities for persisted record embeddings."""

from __future__ import annotations

import hashlib
import json

from searchkernel.domain import Record
from searchkernel.indexing.semantic import semantic_input_for_record


def record_embedding_revision(record: Record, model_namespace: str, dim: int) -> str:
    """Return a deterministic revision for one stored record embedding."""
    payload = {
        "storage_key": record.storage_key,
        "text": semantic_input_for_record(record).text,
        "model_namespace": model_namespace,
        "dimension": dim,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()
