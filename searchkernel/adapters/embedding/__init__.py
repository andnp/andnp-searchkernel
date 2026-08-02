"""Embedding provider adapters implementing the EmbeddingProvider port."""

from searchkernel.adapters.embedding.huggingface import HuggingFaceEmbeddingProvider
from searchkernel.adapters.embedding.ollama import OllamaEmbeddingProvider

__all__ = ["HuggingFaceEmbeddingProvider", "OllamaEmbeddingProvider"]
