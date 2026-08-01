from searchkernel.adapters.stores.pgvector import PGVectorStore
from searchkernel.adapters.stores.pgvector_index import PGVectorIndex
from searchkernel.domain import RecordHit, Vector
from searchkernel.runtime import QueryEmbeddingCache


class _Embedder:
    model_name = "test-model"
    dim = 2

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[Vector]:
        return [[1.0, 0.0]]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0, 0.0]


class _Store(PGVectorStore):
    def __init__(self) -> None:
        self.calls = 0

    def search(
        self,
        query_vector: Vector,
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, object] | None = None,
    ) -> list[RecordHit]:
        self.calls += 1
        return []


def test_pgvector_index_reuses_model_safe_query_embedding_cache() -> None:
    embedder = _Embedder()
    store = _Store()
    index = object.__new__(PGVectorIndex)
    index._embedder = embedder
    index._store = store
    index._model_name = embedder.model_name
    index._dim = embedder.dim
    index._workspace_id = None
    index._encoder_namespace = "test-model|dim=2"
    index._query_embedding_cache = QueryEmbeddingCache()

    assert index.search("same query") == []
    assert index.search("same   query") == []

    assert embedder.calls == 1
    assert store.calls == 2
