"""PGVectorIndex: a VectorIndex-shaped adapter over the PGVectorStore port.

The live query/ingestion path (`IndexManager`, `SearchOrchestrator`,
`ChunkHydrator`, the pipeline stages) calls a richer surface than the
narrow `VectorStore` port exposes: `VectorStore` only models embedding
upsert/search/delete, while the live path also needs by-id chunk
hydration (content + metadata), per-document chunk enumeration, and
in-place chunk-path renames.

Rather than widen the port for one adapter (which would leak FAISS/pgvector
concerns into code that only needs `upsert`/`search`/`delete`), this class
presents the exact method surface the live path actually calls -- the same
set `searchkernel.indices.vector.VectorIndex` implements for the FAISS
backend -- and backs it with `PGVectorStore` plus direct reads of the
`records` table `PGVectorStore` already writes (chunk bookkeeping such as
doc_id/header_path/file_path/parent_chunk_id/project_id rides in
`records.metadata`, since the port itself has no by-id/by-document lookup).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from psycopg2 import sql

from searchkernel.adapters.stores.pgvector import (
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_ITERATIVE_SCAN,
    DEFAULT_HNSW_MAX_SCAN_TUPLES,
    DEFAULT_HNSW_SCAN_MEM_MULTIPLIER,
    PGVectorStore,
    PostgresConnection,
    _create_schema,
)
from searchkernel.domain import (
    Chunk,
    Record,
    RecordHit,
    RecordStatus,
    Vector,
    canonical_storage_key,
)
from searchkernel.runtime import QueryEmbeddingCache, get_query_embedding_cache
from searchkernel.search.types import SearchResultDict

logger = logging.getLogger(__name__)

# Chunks are indexed as Records under this fixed source_kind so this
# adapter's document/chunk bookkeeping queries never collide with other
# record kinds (e.g. git commits) that might share the same `records` table.
_SOURCE_KIND = "chunk"


class _Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: list[str]) -> list[Vector]: ...
    def embed_query(self, text: str) -> Vector: ...


def _default_embedder(model_name: str, truncate_dim: int | None = None) -> _Embedder:
    from searchkernel.adapters.embedding import HuggingFaceEmbeddingProvider

    return HuggingFaceEmbeddingProvider(model_name=model_name, truncate_dim=truncate_dim)


def _embedding_text(chunk: Chunk) -> str:
    header_path = chunk.metadata.get("header_path", "")
    return f"{header_path}\n\n{chunk.content}" if header_path else chunk.content


def _parse_stored_vector(raw: Any, dim: int) -> list[float]:
    if isinstance(raw, str):
        try:
            values = json.loads(raw)
        except json.JSONDecodeError:
            values = [value for value in raw.strip("[]").split(",") if value]
    elif hasattr(raw, "tolist"):
        values = raw.tolist()
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        raise TypeError("stored vector has an unsupported representation")
    if not isinstance(values, list) or len(values) != dim:
        raise ValueError(
            f"stored vector dimension mismatch: expected {dim}, got "
            f"{len(values) if isinstance(values, list) else 'unknown'}"
        )
    return [float(value) for value in values]


def _parse_modified_time(raw: Any) -> datetime:
    """Parse a chunk's `metadata["modified_time"]` (ISO text) back to a datetime.

    Chunk metadata must stay JSON-serializable (it's also spread into
    vector/keyword/graph index payloads), so `modified_time` is stored as
    ISO text rather than a raw `datetime`.
    """
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str) and raw:
        return datetime.fromisoformat(raw)
    return datetime.now(UTC)


def _chunk_to_record(chunk: Chunk, workspace_id: str | None = None) -> Record:
    header_path = chunk.metadata.get("header_path", "")
    file_path = chunk.metadata.get("file_path", "")
    modified_time = _parse_modified_time(chunk.metadata.get("modified_time"))
    metadata = {
        **chunk.metadata,
        "doc_id": chunk.record_id,
        "chunk_index": chunk.chunk_index,
        "header_path": header_path,
        "file_path": file_path,
        "parent_chunk_id": chunk.metadata.get("parent_chunk_id"),
        "project_id": chunk.metadata.get("project_id"),
    }
    return Record(
        source_kind=_SOURCE_KIND,
        source_id=chunk.chunk_id,
        workspace_id=workspace_id
        if workspace_id is not None
        else chunk.metadata.get("workspace_id"),
        title=header_path or file_path,
        body=chunk.content,
        created_at=modified_time,
        updated_at=modified_time,
        metadata=metadata,
        uri=file_path,
        status=RecordStatus.ACTIVE,
    )


def _row_to_result(
    record_id: str,
    body: str,
    metadata: dict[str, Any],
    score: float,
) -> SearchResultDict:
    doc_id = metadata.get("doc_id")
    if not isinstance(doc_id, str):
        raise TypeError(f"Missing string doc_id metadata for record {record_id}")

    header_path = metadata.get("header_path", "")
    if not isinstance(header_path, str):
        header_path = ""

    file_path = metadata.get("file_path", "")
    if not isinstance(file_path, str):
        file_path = ""

    project_id = metadata.get("project_id")
    if project_id is not None and not isinstance(project_id, str):
        project_id = None

    return {
        "chunk_id": record_id,
        "doc_id": doc_id,
        "score": score,
        "header_path": header_path,
        "file_path": file_path,
        "project_id": project_id,
        "content": body,
        "metadata": metadata,
    }


class PGVectorIndex:
    """Postgres + pgvector backed replacement for the live `VectorIndex`.

    Presents `add_chunk(s)`/`search`/`remove(_chunk)`/`update_chunk_path`/
    `prune_document`/`persist`/`load`/`is_ready`/`clear`/`get_chunk_by_id`/
    `get_chunk_ids_for_document`/`get_document_ids`/`get_parent_content`/
    `get_embedding_for_chunk`/`expand_query` -- the exact set the live
    query/ingestion path calls on `VectorIndex` today.
    """

    query_expansion_supported = False

    def __init__(
        self,
        pg_dsn: str,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        embedder: _Embedder | None = None,
        *,
        truncate_dim: int | None = None,
        workspace_id: str | None = None,
        query_embedding_cache: QueryEmbeddingCache | None = None,
        encoder_namespace: str | None = None,
        hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH,
        hnsw_iterative_scan: str = DEFAULT_HNSW_ITERATIVE_SCAN,
        hnsw_max_scan_tuples: int = DEFAULT_HNSW_MAX_SCAN_TUPLES,
        hnsw_scan_mem_multiplier: float = DEFAULT_HNSW_SCAN_MEM_MULTIPLIER,
        overfetch_multiplier: float = 2.0,
        max_scan_rounds: int = 4,
    ):
        self._conn_pool = PostgresConnection(pg_dsn)
        _create_schema(self._conn_pool)
        self._store = PGVectorStore(
            self._conn_pool,
            hnsw_ef_search=hnsw_ef_search,
            hnsw_iterative_scan=hnsw_iterative_scan,
            hnsw_max_scan_tuples=hnsw_max_scan_tuples,
            hnsw_scan_mem_multiplier=hnsw_scan_mem_multiplier,
            overfetch_multiplier=overfetch_multiplier,
            max_scan_rounds=max_scan_rounds,
        )
        self._embedder = embedder or _default_embedder(embedding_model_name, truncate_dim=truncate_dim)
        self._workspace_id = workspace_id
        self._model_name = self._embedder.model_name
        self._dim = self._embedder.dim
        self._query_embedding_cache = (
            query_embedding_cache or get_query_embedding_cache()
        )
        self._encoder_namespace = (
            encoder_namespace
            or getattr(self._embedder, "encoder_namespace", None)
            or f"{self._model_name}|dim={self._dim}"
        )

    def warm_up(self) -> None:
        """No-op: the embedding model loads eagerly in `__init__`."""

    def add_chunk(self, chunk: Chunk) -> None:
        self.add_chunks([chunk])

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        vectors = self._embedder.embed([_embedding_text(c) for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        records = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self._dim:
                raise ValueError(
                    f"embedding dimension mismatch for {chunk.chunk_id}: "
                    f"expected {self._dim}, got {len(vector)}"
                )
            record = _chunk_to_record(
                chunk,
                self._effective_workspace_id(chunk),
            )
            record.embedding = vector
            records.append(record)
        self._store.upsert(records, self._model_name, self._dim)

    def search(
        self,
        query: str,
        top_k: int = 10,
        excluded_files: set[str] | None = None,
        docs_root: Path | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResultDict]:
        if not query.strip():
            return []

        query_vector = self._query_embedding_cache.get_or_compute(
            encoder_namespace=self._encoder_namespace,
            query=query,
            compute=lambda: self._embedder.embed_query(query),
        )
        search_filters = dict(filters or {})
        search_filters["source_kind"] = _SOURCE_KIND
        if self._workspace_id is not None:
            search_filters["workspace_id"] = self._workspace_id
        if excluded_files:
            existing_exclusions = search_filters.get("excluded_files", ())
            search_filters["excluded_files"] = sorted(
                set(existing_exclusions) | set(excluded_files)
            )
        hits = cast(
            list[RecordHit],
            self._store.search(
            query_vector,
            top_k,
            model_name=self._model_name,
            dim=self._dim,
            filters=search_filters,
            ),
        )
        if not hits:
            return []

        rows = self._fetch_records([hit.storage_key for hit in hits])

        results: list[SearchResultDict] = []
        for hit in hits:
            row = rows.get(hit.storage_key)
            if row is None:
                continue
            source_id, body, metadata = row

            results.append(_row_to_result(source_id, body, metadata, hit.score))
            if len(results) >= top_k:
                break

        return results

    def get_chunk_by_id(self, chunk_id: str) -> SearchResultDict | None:
        row = self._fetch_records([chunk_id]).get(chunk_id)
        if row is None:
            return None
        source_id, body, metadata = row
        return _row_to_result(source_id, body, metadata, 1.0)

    def get_parent_content(self, parent_chunk_id: str) -> str | None:
        chunk = self.get_chunk_by_id(parent_chunk_id)
        return chunk.get("content") if chunk else None

    def get_embedding_for_chunk(self, chunk_id: str) -> list[float] | None:
        row = self._fetch_records([chunk_id]).get(chunk_id)
        if row is None or not row[1]:
            return None
        record_key = self._storage_key(chunk_id)
        conn = self._conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            table_name = self._own_vector_table_name(cursor)
            if table_name is None:
                return None
            cursor.execute(
                sql.SQL(
                    "SELECT v.embedding FROM {table} v "
                    "JOIN records r ON r.record_id = v.record_id "
                    "WHERE v.record_id = %s AND r.source_kind = %s;"
                ).format(table=sql.Identifier(table_name)),
                (record_key, _SOURCE_KIND),
            )
            result = cursor.fetchone()
            if result is None:
                return None
            return _parse_stored_vector(result[0], self._dim)
        finally:
            if cursor is not None:
                cursor.close()
            self._conn_pool.put_connection(conn)

    def get_chunk_ids_for_document(self, doc_id: str) -> list[str]:
        conn = self._conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            table_name = self._own_vector_table_name(cursor)
            if table_name is None:
                return []
            cursor.execute(
                sql.SQL(
                    "SELECT r.source_id FROM records r "
                    "JOIN {table} v ON v.record_id = r.record_id "
                    "WHERE r.source_kind = %s AND r.metadata->>'doc_id' = %s "
                    "AND r.workspace_id IS NOT DISTINCT FROM %s;"
                ).format(table=sql.Identifier(table_name)),
                (_SOURCE_KIND, doc_id, self._workspace_id),
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            if cursor is not None:
                cursor.close()
            self._conn_pool.put_connection(conn)

    def get_document_ids(self) -> list[str]:
        conn = self._conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            table_name = self._own_vector_table_name(cursor)
            if table_name is None:
                return []
            cursor.execute(
                sql.SQL(
                    "SELECT DISTINCT r.metadata->>'doc_id' FROM records r "
                    "JOIN {table} v ON v.record_id = r.record_id "
                    "WHERE r.source_kind = %s "
                    "AND r.workspace_id IS NOT DISTINCT FROM %s;"
                ).format(table=sql.Identifier(table_name)),
                (_SOURCE_KIND, self._workspace_id),
            )
            return [row[0] for row in cursor.fetchall() if row[0] is not None]
        finally:
            if cursor is not None:
                cursor.close()
            self._conn_pool.put_connection(conn)

    def _own_vector_table_name(self, cursor) -> str | None:
        """Resolve this instance's `(model_name, dim)` vector table, if any.

        Enumeration queries (`get_document_ids`/`get_chunk_ids_for_document`)
        join through this table so they only ever see chunks embedded under
        this model -- `records` is shared Postgres state, and other models
        (or unrelated corpora on the same DSN) must not leak in.
        """
        cursor.execute(
            "SELECT table_name FROM vector_tables WHERE model_name = %s AND dim = %s;",
            (self._model_name, self._dim),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def expand_query(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.5,
    ) -> str:
        """Return the input for compatibility; expansion is explicitly unsupported."""
        return query

    @property
    def last_search_diagnostics(self) -> dict[str, Any]:
        return self._store.last_search_diagnostics

    def remove(self, document_id: str) -> None:
        chunk_ids = self.get_chunk_ids_for_document(document_id)
        if chunk_ids:
            self._store.delete_for_model(
                [self._storage_key(chunk_id) for chunk_id in chunk_ids],
                self._model_name,
                self._dim,
            )

    def remove_chunk(self, chunk_id: str) -> None:
        self._store.delete_for_model(
            [self._storage_key(chunk_id)], self._model_name, self._dim
        )

    def prune_document(self, doc_id: str) -> int:
        chunk_ids = self.get_chunk_ids_for_document(doc_id)
        if chunk_ids:
            self._store.delete_for_model(
                [self._storage_key(chunk_id) for chunk_id in chunk_ids],
                self._model_name,
                self._dim,
            )
        return len(chunk_ids)

    def update_chunk_path(
        self, old_chunk_id: str, new_chunk_id: str, new_metadata: dict
    ) -> bool:
        row = self._fetch_records([old_chunk_id]).get(old_chunk_id)
        if row is None:
            return False
        _source_id, body, _old_metadata = row
        workspace_id = self._workspace_id
        if workspace_id is None:
            raw_workspace_id = _old_metadata.get("workspace_id")
            if isinstance(raw_workspace_id, str):
                workspace_id = raw_workspace_id

        record = Record(
            source_kind=_SOURCE_KIND,
            source_id=new_chunk_id,
            title=new_metadata.get("header_path") or new_metadata.get("file_path", ""),
            body=body,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            workspace_id=workspace_id,
            metadata={**new_metadata, "workspace_id": workspace_id},
            uri=new_metadata.get("file_path"),
            status=RecordStatus.ACTIVE,
        )
        record.embedding = self._embedder.embed([body])[0]
        self._store.upsert([record], self._model_name, self._dim)
        self._store.delete_for_model(
            [canonical_storage_key(workspace_id, _SOURCE_KIND, old_chunk_id)],
            self._model_name,
            self._dim,
        )
        return True

    def is_ready(self) -> bool:
        return True

    def persist(self, path: Path) -> None:
        """No-op: pgvector data is already durable in Postgres."""

    def load(self, path: Path) -> None:
        """No-op: pgvector data is already durable in Postgres."""

    def clear(self) -> None:
        """Delete every record in *this* model's vector table.

        Scoped to `(model_name, dim)` rather than all `source_kind="chunk"`
        records, since multiple models can coexist in the same Postgres
        database (e.g. during a model migration) and clearing one index
        should not touch another's.
        """
        conn = self._conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            table_name = self._own_vector_table_name(cursor)
            if table_name is None:
                return
            cursor.execute(
                sql.SQL(
                    "SELECT v.record_id FROM {table} v "
                    "JOIN records r ON r.record_id = v.record_id "
                    "WHERE r.source_kind = %s "
                    "AND r.workspace_id IS NOT DISTINCT FROM %s;"
                ).format(table=sql.Identifier(table_name)),
                (_SOURCE_KIND, self._workspace_id),
            )
            record_ids = [r[0] for r in cursor.fetchall()]
        finally:
            if cursor is not None:
                cursor.close()
            self._conn_pool.put_connection(conn)
        if record_ids:
            self._store.delete_for_model(record_ids, self._model_name, self._dim)

    def _effective_workspace_id(self, chunk: Chunk) -> str | None:
        if self._workspace_id is not None:
            return self._workspace_id
        raw_workspace_id = chunk.metadata.get("workspace_id")
        return raw_workspace_id if isinstance(raw_workspace_id, str) else None

    def _storage_key(self, chunk_id: str) -> str:
        if chunk_id.startswith("record:"):
            return chunk_id
        if self._workspace_id is not None:
            return canonical_storage_key(self._workspace_id, _SOURCE_KIND, chunk_id)
        row = self._fetch_records([chunk_id]).get(chunk_id)
        if row is not None:
            workspace_id = row[2].get("workspace_id")
            if isinstance(workspace_id, str):
                return canonical_storage_key(workspace_id, _SOURCE_KIND, chunk_id)
        return canonical_storage_key(None, _SOURCE_KIND, chunk_id)

    def _fetch_records(
        self, record_ids: list[str]
    ) -> dict[str, tuple[str, str, dict[str, Any]]]:
        if not record_ids:
            return {}
        conn = self._conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            table_name = self._own_vector_table_name(cursor)
            if table_name is None:
                return {}
            canonical_ids = [
                record_id for record_id in record_ids if record_id.startswith("record:")
            ]
            bare_ids = [
                record_id
                for record_id in record_ids
                if not record_id.startswith("record:")
            ]
            if self._workspace_id is None and bare_ids:
                cursor.execute(
                    sql.SQL(
                        "SELECT r.record_id, r.source_id, r.workspace_id, "
                        "r.source_kind, r.body, r.metadata "
                        "FROM records r JOIN {table} v ON v.record_id = r.record_id "
                        "WHERE r.source_kind = %s "
                        "AND (r.record_id = ANY(%s) OR r.source_id = ANY(%s));"
                    ).format(table=sql.Identifier(table_name)),
                    (_SOURCE_KIND, canonical_ids, bare_ids),
                )
            else:
                storage_ids = canonical_ids + [
                    canonical_storage_key(self._workspace_id, _SOURCE_KIND, record_id)
                    for record_id in bare_ids
                ]
                cursor.execute(
                    sql.SQL(
                        "SELECT r.record_id, r.source_id, r.workspace_id, "
                        "r.source_kind, r.body, r.metadata "
                        "FROM records r JOIN {table} v ON v.record_id = r.record_id "
                        "WHERE r.record_id = ANY(%s);"
                    ).format(table=sql.Identifier(table_name)),
                    (storage_ids,),
                )
            rows = cursor.fetchall()
            result: dict[str, tuple[str, str, dict[str, Any]]] = {}
            for row in rows:
                metadata = {
                    **(row[5] or {}),
                    "workspace_id": row[2],
                    "source_kind": row[3],
                }
                for requested_id in record_ids:
                    if requested_id == row[0] or requested_id == row[1]:
                        if requested_id in result and result[requested_id][0] != row[1]:
                            raise ValueError(
                                f"ambiguous chunk identity {requested_id!r}; "
                                "provide workspace_id"
                            )
                        result[requested_id] = (row[1], row[4], metadata)
            return result
        finally:
            if cursor is not None:
                cursor.close()
            self._conn_pool.put_connection(conn)
