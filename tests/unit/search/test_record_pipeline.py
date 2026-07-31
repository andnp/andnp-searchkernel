from datetime import UTC, datetime

import pytest

from searchkernel.domain import Record
from searchkernel.search.record_pipeline import (
    RecordSearchConfig,
    RecordSearchError,
    RecordSearchPipeline,
    RecordSearchPolicy,
)


def _record(record_id: str) -> Record:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    return Record(
        source_kind="fake",
        source_id=record_id,
        title=record_id,
        body=f"body for {record_id}",
        created_at=timestamp,
        updated_at=timestamp,
    )


class FakeKeywordStore:
    def __init__(self, results: list[tuple[str, float]]) -> None:
        self.results = results
        self.queries: list[tuple[str, int, dict[str, object] | None]] = []

    def index(self, records: list[Record]) -> None:
        pass

    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        self.queries.append((query, k, filters))
        return self.results


class FakeVectorStore:
    def __init__(self, results: list[tuple[str, float]]) -> None:
        self.results = results
        self.filters: list[dict[str, object] | None] = []

    def upsert(self, records: list[Record], model_name: str, dim: int) -> None:
        pass

    def search(
        self,
        query_vector: list[float],
        k: int,
        *,
        model_name: str,
        dim: int,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        assert query_vector == [1.0, 0.0]
        assert (model_name, dim) == ("fake-model", 2)
        self.filters.append(filters)
        return self.results

    def delete(self, record_ids: list[str]) -> None:
        pass

    def epoch(self) -> int:
        return 0


class FakeGraphStore:
    def __init__(self, neighbors: dict[str, list[tuple[str, str, float]]]) -> None:
        self._neighbors = neighbors

    def upsert_edges(
        self, edges: list[tuple[str, str, str, float]]
    ) -> None:
        pass

    def neighbors(
        self,
        record_id: str,
        edge_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[tuple[str, str, float]]:
        return self._neighbors.get(record_id, [])


class FakeEmbedder:
    model_name = "fake-model"
    dim = 2

    def embed_query(self, query: str) -> list[float]:
        assert query == "query"
        return [1.0, 0.0]


def _hydrator(records: dict[str, Record]):
    return lambda record_id: records.get(record_id)


def test_keyword_only_hydrates_records_with_deterministic_ties() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("b", 1.0), ("a", 1.0)]),
        hydrator=_hydrator(records),
    )

    outcome = pipeline.search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["a", "b"]
    assert all(result.record.source_kind == "fake" for result in outcome.results)
    assert outcome.results[0].provenance.strategies == ("keyword",)


def test_minimum_candidate_limit_applies_to_store_acquisition() -> None:
    records = {"a": _record("a")}
    keyword_store = FakeKeywordStore([("a", 1.0)])
    pipeline = RecordSearchPipeline(
        keyword_store=keyword_store,
        hydrator=_hydrator(records),
        config=RecordSearchConfig(minimum_candidate_limit=50),
    )

    pipeline.search("query", limit=1)

    assert keyword_store.queries[0][1] == 50


def test_hybrid_search_fuses_keyword_and_vector_rankings() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b", "c")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("b", 10.0), ("a", 1.0)]),
        vector_store=FakeVectorStore([("a", 0.9), ("c", 0.8)]),
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator(records),
    )

    outcome = pipeline.search("query", limit=3)

    assert [result.record_id for result in outcome.results] == ["a", "b", "c"]
    assert outcome.results[0].provenance.strategies == ("keyword", "vector")


def test_policy_can_bound_vector_acquisition_to_keyword_candidates() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    vector_store = FakeVectorStore([("a", 0.9)])
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("b", 1.0), ("a", 0.5)]),
        vector_store=vector_store,
        embedding_provider=FakeEmbedder(),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(
            vector_candidate_ids=lambda ranking, filters: [record_id for record_id, _ in ranking]
        ),
    )

    pipeline.search("query", limit=2, filters={"workspace_id": "workspace-1"})

    assert vector_store.filters == [
        {"workspace_id": "workspace-1", "candidate_ids": ["b", "a"]}
    ]


def test_candidate_filter_runs_before_graph_expansion() -> None:
    records = {record_id: _record(record_id) for record_id in ("seed", "blocked", "allowed")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 1.0)]),
        graph_store=FakeGraphStore(
            {"seed": [("blocked", "related", 1.0), ("allowed", "related", 0.5)]}
        ),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(
            candidate_filter=lambda candidate: candidate.record_id != "blocked"
        ),
    )

    outcome = pipeline.search("query", limit=3)

    assert [result.record_id for result in outcome.results] == ["seed", "allowed"]
    assert "blocked" not in {result.record_id for result in outcome.results}
    assert "graph" in outcome.results[-1].provenance.strategies


def test_policy_can_adjust_scores_reject_results_and_post_process() -> None:
    records = {record_id: _record(record_id) for record_id in ("a", "b")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("a", 1.0), ("b", 0.5)]),
        hydrator=_hydrator(records),
        policy=RecordSearchPolicy(
            score_adjuster=lambda candidate: candidate.score
            + (1.0 if candidate.record_id == "b" else 0.0),
            result_filter=lambda result: result.record_id != "a",
            post_process=lambda results: list(reversed(results)),
        ),
    )

    outcome = pipeline.search("query", limit=2)

    assert [result.record_id for result in outcome.results] == ["b"]
    assert outcome.results[0].score > 0


def test_graph_expansion_is_bounded_and_missing_records_are_reported() -> None:
    records = {"seed": _record("seed")}
    pipeline = RecordSearchPipeline(
        keyword_store=FakeKeywordStore([("seed", 1.0)]),
        graph_store=FakeGraphStore(
            {
                "seed": [
                    ("missing", "related", 1.0),
                    ("other", "related", 0.9),
                ]
            }
        ),
        hydrator=_hydrator(records),
        config=RecordSearchConfig(max_neighbors_per_seed=1),
        continue_on_error=True,
    )

    outcome = pipeline.search("query", limit=3)

    assert [result.record_id for result in outcome.results] == ["seed"]
    assert outcome.missing_record_ids == ("missing",)
    assert outcome.degraded


def test_callable_embedding_provider_accepts_explicit_vector_metadata() -> None:
    records = {"a": _record("a")}
    pipeline = RecordSearchPipeline(
        vector_store=FakeVectorStore([("a", 1.0)]),
        embedding_provider=lambda query: [1.0, 0.0],
        embedding_model_name="fake-model",
        embedding_dim=2,
        hydrator=_hydrator(records),
    )

    assert [result.record_id for result in pipeline.search("query").results] == ["a"]


def test_store_errors_raise_by_default() -> None:
    class BrokenKeywordStore(FakeKeywordStore):
        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[tuple[str, float]]:
            raise RuntimeError("backend unavailable")

    pipeline = RecordSearchPipeline(
        keyword_store=BrokenKeywordStore([]),
        hydrator=_hydrator({}),
    )

    with pytest.raises(RecordSearchError, match="keyword retrieval failed"):
        pipeline.search("query")


def test_store_errors_can_be_explicitly_returned_as_degraded() -> None:
    class BrokenKeywordStore(FakeKeywordStore):
        def search(
            self,
            query: str,
            k: int,
            filters: dict[str, object] | None = None,
        ) -> list[tuple[str, float]]:
            raise RuntimeError("backend unavailable")

    pipeline = RecordSearchPipeline(
        keyword_store=BrokenKeywordStore([]),
        hydrator=_hydrator({}),
        continue_on_error=True,
    )

    outcome = pipeline.search("query")

    assert not outcome.results
    assert outcome.degraded
    assert outcome.failures[0].stage == "keyword"
