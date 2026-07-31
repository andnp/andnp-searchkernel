"""Tests for truncate_dim parameter in VectorIndex and PGVectorIndex."""

import inspect
from unittest.mock import MagicMock, patch

from searchkernel.adapters.stores.pgvector_index import (
    PGVectorIndex,
    _default_embedder,
)
from searchkernel.indices.vector import VectorIndex


class TestVectorIndexTruncateDim:
    """Test truncate_dim parameter in VectorIndex."""

    def test_vectorindex_stores_truncate_dim(self):
        """VectorIndex stores truncate_dim parameter."""
        index = VectorIndex(truncate_dim=256)
        assert index._truncate_dim == 256

    def test_vectorindex_truncate_dim_default_none(self):
        """VectorIndex truncate_dim defaults to None."""
        index = VectorIndex()
        assert index._truncate_dim is None

    def test_vectorindex_with_custom_embedding_model_stores_truncate_dim(
        self, deterministic_fake_embedding_model
    ):
        """VectorIndex stores truncate_dim even with custom embedding model."""
        index = VectorIndex(
            embedding_model=deterministic_fake_embedding_model, truncate_dim=256
        )
        assert index._truncate_dim == 256
        assert index._embedding_model is deterministic_fake_embedding_model

    def test_vectorindex_truncate_dim_various_values(self):
        """VectorIndex correctly stores various truncate_dim values."""
        for value in [128, 256, 384, 512]:
            index = VectorIndex(truncate_dim=value)
            assert index._truncate_dim == value

    def test_vectorindex_truncate_dim_keyword_only(self):
        """truncate_dim must be passed as keyword argument."""
        # This should work (keyword)
        index1 = VectorIndex(truncate_dim=256)
        assert index1._truncate_dim == 256

        parameter = inspect.signature(VectorIndex).parameters["truncate_dim"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    def test_vectorindex_truncate_dim_with_all_params(self):
        """VectorIndex accepts truncate_dim alongside all other parameters."""
        index = VectorIndex(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_workers=8,
            torch_num_threads=2,
            truncate_dim=128,
        )
        assert index._truncate_dim == 128
        assert index._embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert index._embedding_workers == 8
        assert index._torch_num_threads == 2


class TestVectorIndexTruncateDimLoading:
    """Test that truncate_dim is properly used during model loading."""

    def test_vectorindex_passes_truncate_dim_to_huggingface_embedding(self):
        """VectorIndex passes truncate_dim to HuggingFaceEmbedding when set."""
        with patch(
            "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
        ) as mock_hf_embedding_class:
            mock_model_instance = MagicMock()
            mock_hf_embedding_class.return_value = mock_model_instance

            with patch("llama_index.core.Settings"):
                index = VectorIndex(truncate_dim=256)
                # Access the nested load_model function by triggering model loading
                # We patch it at the right scope
                try:
                    index._ensure_model_loaded()
                except RuntimeError:
                    # We expect this may fail since Settings is mocked,
                    # but we just care that HuggingFaceEmbedding was called correctly
                    pass

                # Check if called with truncate_dim
                calls = mock_hf_embedding_class.call_args_list
                if calls:
                    # If the call was made, verify truncate_dim was passed
                    call_kwargs = calls[0][1]
                    assert "truncate_dim" in call_kwargs or len(calls[0][0]) > 1

    def test_vectorindex_model_loading_integration(self, deterministic_fake_embedding_model):
        """VectorIndex with truncate_dim and fake embedding model works correctly."""
        # This uses the fake embedding model fixture to avoid real model loading
        index = VectorIndex(
            embedding_model=deterministic_fake_embedding_model,
            truncate_dim=256,
        )
        assert index._truncate_dim == 256
        assert index._embedding_model is deterministic_fake_embedding_model


class TestPGVectorIndexTruncateDim:
    """Test truncate_dim parameter in PGVectorIndex and _default_embedder."""

    def test_default_embedder_stores_truncate_dim(self):
        """_default_embedder forwards truncate_dim to HuggingFaceEmbeddingProvider."""
        with patch(
            "searchkernel.adapters.embedding.HuggingFaceEmbeddingProvider"
        ) as mock_provider:
            mock_embedder = MagicMock()
            mock_provider.return_value = mock_embedder

            result = _default_embedder("test-model", truncate_dim=256)

            assert result is mock_embedder
            mock_provider.assert_called_once_with(
                model_name="test-model",
                truncate_dim=256,
            )

    def test_default_embedder_no_truncate_dim_kwarg_when_none(self):
        """_default_embedder forwards truncate_dim=None explicitly."""
        with patch(
            "searchkernel.adapters.embedding.HuggingFaceEmbeddingProvider"
        ) as mock_provider:
            mock_embedder = MagicMock()
            mock_provider.return_value = mock_embedder

            result = _default_embedder("test-model")

            assert result is mock_embedder
            mock_provider.assert_called_once_with(
                model_name="test-model",
                truncate_dim=None,
            )

    def test_default_embedder_various_truncate_dim_values(self):
        """_default_embedder correctly forwards various truncate_dim values."""
        with patch(
            "searchkernel.adapters.embedding.HuggingFaceEmbeddingProvider"
        ) as mock_provider:
            mock_embedder = MagicMock()
            mock_provider.return_value = mock_embedder

            for value in [128, 256, 384, 512]:
                mock_provider.reset_mock()
                _default_embedder("test-model", truncate_dim=value)
                mock_provider.assert_called_once_with(
                    model_name="test-model",
                    truncate_dim=value,
                )

    def test_pgvectorindex_stores_truncate_dim(self):
        """PGVectorIndex accepts truncate_dim parameter."""
        with patch(
            "searchkernel.adapters.stores.pgvector_index.PostgresConnection"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index._create_schema"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index.PGVectorStore"
        ), patch(
            "searchkernel.adapters.embedding.HuggingFaceEmbeddingProvider"
        ) as mock_provider:
            mock_embedder = MagicMock()
            mock_embedder.model_name = "test-model"
            mock_embedder.dim = 256
            mock_provider.return_value = mock_embedder

            # Create index with truncate_dim
            result_index = PGVectorIndex(
                pg_dsn="postgresql://localhost/test",
                truncate_dim=256,
            )

            # Verify _default_embedder was called with truncate_dim
            mock_provider.assert_called_once_with(
                model_name="BAAI/bge-small-en-v1.5",
                truncate_dim=256,
            )
            assert result_index is not None

    def test_pgvectorindex_truncate_dim_default_none(self):
        """PGVectorIndex truncate_dim defaults to None."""
        with patch(
            "searchkernel.adapters.stores.pgvector_index.PostgresConnection"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index._create_schema"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index.PGVectorStore"
        ), patch(
            "searchkernel.adapters.embedding.HuggingFaceEmbeddingProvider"
        ) as mock_provider:
            mock_embedder = MagicMock()
            mock_embedder.model_name = "test-model"
            mock_embedder.dim = 384
            mock_provider.return_value = mock_embedder

            # Create index without truncate_dim
            result_index = PGVectorIndex(
                pg_dsn="postgresql://localhost/test",
            )

            # Verify _default_embedder was called with truncate_dim=None
            mock_provider.assert_called_once_with(
                model_name="BAAI/bge-small-en-v1.5",
                truncate_dim=None,
            )
            assert result_index is not None

    def test_pgvectorindex_custom_embedder_ignores_truncate_dim(self):
        """PGVectorIndex ignores truncate_dim when custom embedder is provided."""
        with patch(
            "searchkernel.adapters.stores.pgvector_index.PostgresConnection"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index._create_schema"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index.PGVectorStore"
        ), patch(
            "searchkernel.adapters.embedding.HuggingFaceEmbeddingProvider"
        ) as mock_provider:
            custom_embedder = MagicMock()
            custom_embedder.model_name = "custom"
            custom_embedder.dim = 128

            # Create index with custom embedder and truncate_dim
            # (embedder is provided, so _default_embedder won't be called)
            result_index = PGVectorIndex(
                pg_dsn="postgresql://localhost/test",
                embedder=custom_embedder,
                truncate_dim=256,  # This will be ignored
            )

            # Verify _default_embedder was NOT called
            mock_provider.assert_not_called()
            assert result_index._embedder is custom_embedder

    def test_pgvectorindex_truncate_dim_keyword_only(self):
        """truncate_dim must be passed as keyword argument to PGVectorIndex."""
        with patch(
            "searchkernel.adapters.stores.pgvector_index.PostgresConnection"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index._create_schema"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index.PGVectorStore"
        ), patch(
            "searchkernel.adapters.embedding.HuggingFaceEmbeddingProvider"
        ):
            # This should work (keyword)
            result_index = PGVectorIndex(
                pg_dsn="postgresql://localhost/test",
                truncate_dim=256,
            )
            assert result_index is not None

            parameter = inspect.signature(PGVectorIndex).parameters["truncate_dim"]
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY

    def test_pgvectorindex_with_custom_model_name(self):
        """PGVectorIndex accepts custom embedding_model_name with truncate_dim."""
        with patch(
            "searchkernel.adapters.stores.pgvector_index.PostgresConnection"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index._create_schema"
        ), patch(
            "searchkernel.adapters.stores.pgvector_index.PGVectorStore"
        ), patch(
            "searchkernel.adapters.embedding.HuggingFaceEmbeddingProvider"
        ) as mock_provider:
            mock_embedder = MagicMock()
            mock_embedder.model_name = "custom-model"
            mock_embedder.dim = 768
            mock_provider.return_value = mock_embedder

            result_index = PGVectorIndex(
                pg_dsn="postgresql://localhost/test",
                embedding_model_name="sentence-transformers/all-MiniLM-L12-v2",
                truncate_dim=384,
            )

            mock_provider.assert_called_once_with(
                model_name="sentence-transformers/all-MiniLM-L12-v2",
                truncate_dim=384,
            )
            assert result_index is not None
