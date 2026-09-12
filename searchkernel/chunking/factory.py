from typing import Literal

from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.chunking.fixed_window_chunker import FixedWindowChunker
from searchkernel.chunking.header_chunker import HeaderBasedChunker
from searchkernel.ports.chunking_config import ChunkTuningConfig

ChunkingStrategyName = Literal["markdown", "plain_text"]


def get_chunker(
    config: ChunkTuningConfig, *, strategy: ChunkingStrategyName = "markdown"
) -> ChunkingStrategy:
    """Build the chunker for one content shape.

    ``"markdown"`` (the default, and this function's original behavior)
    splits on document structure and needs headers to key off of.
    ``"plain_text"`` is for unstructured prose -- a ticket description, a
    support message -- that has no such structure to split on.
    """
    if strategy == "plain_text":
        return FixedWindowChunker(config)
    return HeaderBasedChunker(config)
