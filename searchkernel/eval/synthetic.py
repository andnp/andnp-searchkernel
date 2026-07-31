"""Deterministic synthetic corpora for evaluation and benchmark runs."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from searchkernel.domain import Record
from searchkernel.eval.golden import GoldenEntry, GoldenSet

_CORPUS_SIZES = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
}
_TOPICS = (
    "api",
    "cache",
    "config",
    "database",
    "embedding",
    "graph",
    "index",
    "latency",
    "metadata",
    "pipeline",
    "query",
    "ranking",
    "relevance",
    "storage",
    "trace",
    "vector",
)
_SOURCE_KINDS = ("docs", "commits", "issues", "notes")


@dataclass
class SyntheticCorpus:
    """Records and golden queries generated from one deterministic seed."""

    records: list[Record]
    golden_set: GoldenSet
    size: int
    seed: int
    version: str

    def __len__(self) -> int:
        """Return the number of records in the corpus."""
        return self.size


def _resolve_size(size: int | str) -> int:
    """Resolve a named benchmark size or validate an explicit count."""
    if isinstance(size, str):
        try:
            return _CORPUS_SIZES[size]
        except KeyError as error:
            raise ValueError(f"unknown synthetic corpus size: {size}") from error
    if size < 1:
        raise ValueError("synthetic corpus size must be positive")
    return size


def make_synthetic_corpus(size: int | str = "1k", *, seed: int = 0) -> SyntheticCorpus:
    """Create a deterministic synthetic corpus at the requested size."""
    record_count = _resolve_size(size)
    version = f"synthetic-v1-{record_count}-{seed}"
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    records: list[Record] = []
    topic_ids: dict[str, list[str]] = {topic: [] for topic in _TOPICS}

    for index in range(record_count):
        topic = _TOPICS[(index + seed) % len(_TOPICS)]
        source_kind = _SOURCE_KINDS[(index + seed) % len(_SOURCE_KINDS)]
        source_id = f"synthetic-{index:08d}"
        records.append(
            Record(
                source_kind=source_kind,
                source_id=source_id,
                title=f"{topic} synthetic record {index}",
                body=(
                    f"synthetic corpus {version} topic {topic} "
                    f"source {source_kind} ordinal {index}"
                ),
                created_at=base_time + timedelta(seconds=index),
                updated_at=base_time + timedelta(seconds=index),
                metadata={"synthetic": True, "ordinal": index, "topic": topic},
                uri=f"synthetic://{source_kind}/{source_id}",
                workspace_id=f"workspace-{(index + seed) % 4}",
            )
        )
        topic_ids[topic].append(source_id)

    entries = [
        GoldenEntry(
            query=f"topic:{topic}",
            relevant_ids=ids,
            relevance={
                result_id: 2.0 if position % 4 == 0 else 1.0
                for position, result_id in enumerate(ids)
            },
            query_type="synthetic_topic",
            source_kinds=list(_SOURCE_KINDS),
            tags=["synthetic", "benchmark"],
            corpus_version=version,
            split="test",
        )
        for topic, ids in topic_ids.items()
    ]
    return SyntheticCorpus(
        records=records,
        golden_set=GoldenSet(entries=entries),
        size=record_count,
        seed=seed,
        version=version,
    )


def make_1k_corpus(*, seed: int = 0) -> SyntheticCorpus:
    """Create the routine-test synthetic corpus."""
    return make_synthetic_corpus("1k", seed=seed)


def make_10k_corpus(*, seed: int = 0) -> SyntheticCorpus:
    """Create the medium benchmark synthetic corpus."""
    return make_synthetic_corpus("10k", seed=seed)


def make_100k_corpus(*, seed: int = 0) -> SyntheticCorpus:
    """Create the large benchmark synthetic corpus."""
    return make_synthetic_corpus("100k", seed=seed)
