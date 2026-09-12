from searchkernel.chunking.base import ChunkingStrategy
from searchkernel.domain import Chunk, Record
from searchkernel.ports.chunking_config import ChunkTuningConfig


class FixedWindowChunker(ChunkingStrategy):
    """Split plain, unstructured text into fixed-size, overlapping windows.

    ``HeaderBasedChunker`` assumes markdown structure (headers) to split on.
    Prose sources with no such structure -- a ticket description, a support
    message, a plain-text field -- have no headers to key off of, so this
    strategy instead slides a fixed-size character window across the body,
    with configurable overlap so a match spanning a window boundary is not
    split away from its context entirely.
    """

    def __init__(self, config: ChunkTuningConfig) -> None:
        if config.max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be positive")
        if config.overlap_chars < 0:
            raise ValueError("overlap_chars must not be negative")
        if config.overlap_chars >= config.max_chunk_chars:
            raise ValueError("overlap_chars must be less than max_chunk_chars")
        self.config = config

    def chunk_record(self, record: Record) -> list[Chunk]:
        content = record.body
        max_chars = self.config.max_chunk_chars
        if len(content) <= max_chars:
            return [self._build_chunk(record, content, chunk_index=0, start_pos=0)]

        step = max_chars - self.config.overlap_chars
        windows: list[tuple[int, int]] = []
        start_pos = 0
        while start_pos < len(content):
            end_pos = min(start_pos + max_chars, len(content))
            windows.append((start_pos, end_pos))
            if end_pos == len(content):
                break
            start_pos += step

        windows = self._merge_trailing_small_window(windows)
        return [
            self._build_chunk(
                record,
                content[start_pos:end_pos],
                chunk_index=index,
                start_pos=start_pos,
            )
            for index, (start_pos, end_pos) in enumerate(windows)
        ]

    def _merge_trailing_small_window(
        self, windows: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Fold a too-small final window into its predecessor.

        A trailing remainder shorter than ``min_chunk_chars`` carries too
        little context to be useful as its own embedding, so it is merged
        into the previous window instead of shipped as a tiny chunk.
        """
        if len(windows) < 2:
            return windows
        last_start, last_end = windows[-1]
        if last_end - last_start >= self.config.min_chunk_chars:
            return windows
        previous_start, _ = windows[-2]
        return [*windows[:-2], (previous_start, last_end)]

    def _build_chunk(
        self,
        record: Record,
        content: str,
        *,
        chunk_index: int,
        start_pos: int,
    ) -> Chunk:
        metadata = dict(record.metadata)
        metadata["start_pos"] = start_pos
        metadata["end_pos"] = start_pos + len(content)
        chunk = Chunk(
            chunk_id=f"{record.source_id}_chunk_{chunk_index}",
            record_id=record.source_id,
            content=content,
            metadata=metadata,
            chunk_index=chunk_index,
        )
        chunk.content_hash = chunk.compute_content_hash()
        return chunk


__all__ = ["FixedWindowChunker"]
