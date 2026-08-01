"""Tests for the local scale gate configuration and evidence checks."""

import json
from pathlib import Path

from benchmarks.local_scale_gate import ScaleGateConfig, evaluate_gate, load_config


def test_local_scale_fixture_covers_all_reproducible_sizes() -> None:
    config = load_config(Path("benchmarks/local_scale_gate.json"))

    assert config.sizes == ("1k", "10k", "100k")
    assert config.seed == 0
    assert config.dimension >= 16
    assert config.ann.enabled is True
    assert config.ann.required is False


def test_local_scale_gate_accepts_unavailable_optional_ann() -> None:
    config = ScaleGateConfig(sizes=("1k",))
    result = {
        "size": "1k",
        "record_count": 1_000,
        "benchmark": {
            "metadata": {
                "rss_before_index_load_bytes": 1,
                "rss_after_index_load_bytes": 2,
                "rss_peak_query_bytes": 3,
                "index_size_bytes": 4,
            },
            "warm": {
                "latency_p50_ms": 1.0,
                "latency_p95_ms": 2.0,
                "latency_p99_ms": 3.0,
                "qps": 4.0,
            },
        },
        "ann": {"status": "unavailable", "reason": "faiss is not installed"},
    }

    gate = evaluate_gate([result], config)

    assert gate["passed"] is True
    assert gate["failures"] == []
    assert "optional FAISS ANN unavailable" in gate["warnings"][0]


def test_local_scale_gate_requires_resource_evidence() -> None:
    config = ScaleGateConfig(sizes=("1k",))
    result = {
        "size": "1k",
        "benchmark": {"metadata": {}, "warm": {}},
        "ann": {"status": "disabled"},
    }

    gate = evaluate_gate([result], config)

    assert gate["passed"] is False
    assert any("rss_before_index_load_bytes" in failure for failure in gate["failures"])
    json.dumps(gate)
