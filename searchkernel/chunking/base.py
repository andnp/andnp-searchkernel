from abc import ABC, abstractmethod

from searchkernel.domain import Chunk, Record


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk_record(self, record: Record) -> list[Chunk]:
        pass
