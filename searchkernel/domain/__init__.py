"""Core domain types for the search kernel.

This module contains pure data types that form the contract between the kernel
and the outside world. Types here are source-agnostic and I/O-free.
"""

from searchkernel.domain.models import (
    ChangeSignal,
    Chunk,
    ChunkResult,
    CompressionStats,
    Cursor,
    GraphEdge,
    GraphNeighbor,
    Record,
    RecordHit,
    RecordIdentity,
    RecordStatus,
    SearchFilters,
    SearchResultProvenance,
    SearchStrategyStats,
    StrategyContribution,
    Tier,
    Vector,
    canonical_storage_key,
)
from searchkernel.domain.reindex import (
    ActiveModelMetadata,
    BackupMetadata,
    MigrationPhase,
    MigrationState,
    ModelDimensionMismatchError,
    ModelNamespace,
    RollbackMetadata,
    ValidationResult,
)

__all__ = [
    "ActiveModelMetadata",
    "BackupMetadata",
    "ChangeSignal",
    "Chunk",
    "ChunkResult",
    "CompressionStats",
    "Cursor",
    "GraphEdge",
    "GraphNeighbor",
    "MigrationPhase",
    "MigrationState",
    "ModelDimensionMismatchError",
    "ModelNamespace",
    "Record",
    "RecordHit",
    "RecordIdentity",
    "RecordStatus",
    "RollbackMetadata",
    "SearchFilters",
    "SearchResultProvenance",
    "SearchStrategyStats",
    "StrategyContribution",
    "Tier",
    "ValidationResult",
    "Vector",
    "canonical_storage_key",
]
