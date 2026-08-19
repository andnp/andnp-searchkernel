"""Shared PostgreSQL session and epoch storage for store adapters."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol


class _PostgresSession(Protocol):
    def cursor(self) -> Any: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class _PostgresConnectionLike(Protocol):
    def get_connection(self) -> _PostgresSession: ...

    def put_connection(self, conn: _PostgresSession) -> None: ...


class _PostgresEpochLane:
    """Own epoch SQL shared by the PostgreSQL storage lanes."""

    _LANE_COLUMNS: ClassVar[dict[str, str]] = {
        "keyword": "keyword_epoch",
        "vector": "vector_epoch",
        "graph": "graph_epoch",
    }

    @staticmethod
    def bump(
        cursor: Any,
        *,
        keyword: bool = False,
        vector: bool = False,
        graph: bool = False,
    ) -> None:
        if not any((keyword, vector, graph)):
            return
        cursor.execute(
            """
            UPDATE index_epoch
            SET epoch = epoch + 1,
                keyword_epoch = keyword_epoch + %s,
                vector_epoch = vector_epoch + %s,
                graph_epoch = graph_epoch + %s;
            """,
            (int(keyword), int(vector), int(graph)),
        )

    def read_all(self, conn_pool: _PostgresConnectionLike) -> dict[str, int]:
        conn = conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT keyword_epoch, vector_epoch, graph_epoch "
                "FROM index_epoch LIMIT 1;"
            )
            row = cursor.fetchone()
            if row is None:
                return {lane: 0 for lane in self._LANE_COLUMNS}
            return {
                lane: int(row[index])
                for index, lane in enumerate(self._LANE_COLUMNS)
            }
        finally:
            if cursor is not None:
                cursor.close()
            conn_pool.put_connection(conn)

    def read(self, conn_pool: _PostgresConnectionLike, lane: str) -> int:
        return self.read_all(conn_pool)[lane]

    @staticmethod
    def read_total(conn_pool: _PostgresConnectionLike) -> int:
        conn = conn_pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT epoch FROM index_epoch LIMIT 1;")
            row = cursor.fetchone()
            return int(row[0]) if row else 0
        finally:
            if cursor is not None:
                cursor.close()
            conn_pool.put_connection(conn)


_POSTGRES_EPOCH_LANE = _PostgresEpochLane()
