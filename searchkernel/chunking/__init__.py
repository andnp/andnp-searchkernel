from typing import TYPE_CHECKING, Any

from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.domain import Chunk

if TYPE_CHECKING:
    from searchkernel.chunking.factory import get_chunker
    from searchkernel.chunking.header_chunker import HeaderBasedChunker


def __getattr__(name: str) -> Any:
    if name == "HeaderBasedChunker":
        from searchkernel.chunking.header_chunker import HeaderBasedChunker

        return HeaderBasedChunker
    if name == "get_chunker":
        from searchkernel.chunking.factory import get_chunker

        return get_chunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Chunk", "ChunkingStrategy", "HeaderBasedChunker", "get_chunker"]
