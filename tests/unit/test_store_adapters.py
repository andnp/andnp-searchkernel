from searchkernel.adapters import stores


def test_store_adapters_expose_modern_pgvector_store() -> None:
    assert stores.PGVectorStore.__module__ == "searchkernel.adapters.stores.pgvector"
    assert not hasattr(stores, "PGVectorIndex")
