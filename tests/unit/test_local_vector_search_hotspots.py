"""Contract tests for the local vector hotspot benchmark report."""

from benchmarks.local_vector_search_hotspots import run_benchmark


def test_typed_filter_report_contains_valid_stage_timings() -> None:
    """The typed-filter report exposes four nonnegative timing stages."""
    report = run_benchmark(32, 8, repetitions=2)

    for name in (
        "search_project_filtered",
        "search_path_filtered",
        "search_document_filtered",
        "search_source_scoped_filtered",
    ):
        timings = report["results"][name]["stage_timings_ms"]
        stages = {
            key: value
            for key, value in timings.items()
            if key != "total_p50_ms"
        }
        assert set(stages) == {
            "eligible_key_selection",
            "snapshot_metadata_materialization",
            "python_predicate_evaluation",
            "vector_scoring_top_k",
        }
        assert all(
            timing["latency_p50_ms"] >= 0
            and timing["latency_p95_ms"] >= timing["latency_p50_ms"]
            and timing["latency_p99_ms"] >= timing["latency_p95_ms"]
            for timing in stages.values()
        )
        assert timings["total_p50_ms"] == sum(
            timing["latency_p50_ms"] for timing in stages.values()
        )
