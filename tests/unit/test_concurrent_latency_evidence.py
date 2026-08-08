"""Tests for serial/concurrent benchmark evidence."""

from benchmarks.concurrent_latency_evidence import measure_concurrent_latency


def test_concurrent_evidence_preserves_quality_without_latency_gate() -> None:
    evidence = measure_concurrent_latency(repetitions=2, concurrent_workers=4)

    assert evidence["quality_equivalent"] is True
    assert evidence["concurrent"]["concurrency"] == 4
    assert evidence["serial"]["warmup_count"] == 2
    assert evidence["concurrent"]["measured_repetitions"] == 2
    assert evidence["concurrent"]["latency_p95_ms"] is not None
    assert evidence["concurrent"]["qps"] is not None
