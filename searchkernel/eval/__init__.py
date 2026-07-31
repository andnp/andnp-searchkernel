"""Evaluation and observability harness for retrieval quality and latency measurement."""

from searchkernel.eval.gates import (
    EvalGatePolicy,
    EvalGateResult,
    EvaluationGateError,
    enforce_ab_gate,
    evaluate_ab_gate,
)
from searchkernel.eval.runner import (
    AbReport,
    BenchmarkConfig,
    BenchmarkHooks,
    BenchmarkReport,
    EvalReport,
    MetricSnapshot,
    SearchExecution,
    SliceReport,
    ab_eval,
    run_benchmark,
    run_eval,
)
from searchkernel.eval.synthetic import (
    SyntheticCorpus,
    make_1k_corpus,
    make_10k_corpus,
    make_100k_corpus,
    make_synthetic_corpus,
)

__all__ = [
    "AbReport",
    "BenchmarkConfig",
    "BenchmarkHooks",
    "BenchmarkReport",
    "EvalGatePolicy",
    "EvalGateResult",
    "EvalReport",
    "EvaluationGateError",
    "MetricSnapshot",
    "SearchExecution",
    "SliceReport",
    "SyntheticCorpus",
    "ab_eval",
    "enforce_ab_gate",
    "evaluate_ab_gate",
    "make_1k_corpus",
    "make_10k_corpus",
    "make_100k_corpus",
    "make_synthetic_corpus",
    "run_benchmark",
    "run_eval",
]
