"""Evaluate the checked-in labeled retrieval corpus with local FTS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from searchkernel.domain import Record
from searchkernel.eval import BenchmarkConfig, SearchExecution, run_eval
from searchkernel.eval.golden import GoldenSet
from searchkernel.indices import LocalRecordBackend

DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "labeled_retrieval_corpus.json"
)


def load_labeled_fixture(path: Path) -> tuple[list[Record], GoldenSet]:
    """Load records and labeled queries from one reproducible fixture."""
    data: dict[str, Any] = json.loads(path.read_text())
    records = [Record.from_dict(value) for value in data["records"]]
    return records, GoldenSet.from_dict(data)


def evaluate_fixture(
    path: Path = DEFAULT_FIXTURE,
    *,
    k: int = 3,
    warmup_count: int = 2,
    measured_repetitions: int = 5,
):
    """Run labeled quality metrics against the local record backend."""
    records, golden_set = load_labeled_fixture(path)
    with LocalRecordBackend() as backend:
        backend.index(records)
        entries = {entry.query: entry for entry in golden_set}

        def search(query: str) -> SearchExecution:
            entry = entries[query]
            filters = (
                {"workspace_id": entry.workspace_id}
                if entry.workspace_id
                else None
            )
            hits = backend.search_keyword(query, k, filters)
            return SearchExecution(
                ids=tuple(hit.source_id for hit in hits),
                source_kinds={hit.source_id: hit.source_kind for hit in hits},
            )

        corpus_versions = {
            entry.corpus_version for entry in golden_set if entry.corpus_version
        }
        corpus_version = next(iter(corpus_versions)) if len(corpus_versions) == 1 else None
        return run_eval(
            golden_set,
            search,
            k=k,
            config=BenchmarkConfig(
                warmup_count=warmup_count,
                measured_repetitions=measured_repetitions,
                corpus_version=corpus_version,
                backend="sqlite-fts5",
                metadata={"fixture": str(path)},
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    report = evaluate_fixture(
        args.fixture,
        k=args.k,
        warmup_count=args.warmup_count,
        measured_repetitions=args.repetitions,
    )
    json.dump(report.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
