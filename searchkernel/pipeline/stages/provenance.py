"""ProvenanceStage: builds per-chunk `SearchResultProvenance` from strategy results.

Lifted from `SearchOrchestrator._build_result_provenance`. Pure function of
`context.strategy_results` -- for each strategy's ranked
`(chunk_id, raw_score)` list, records the chunk's rank + raw score under
that strategy. Downstream stages/orchestrator glue (community-boost,
project-uplift, parent-expansion) mutate the resulting
`SearchResultProvenance` objects in place, exactly as before this
extraction.

Prefers `context.state["provenance_strategy_results"]` over
`context.strategy_results` when present: `StrategyResultsStage` writes the
*narrow* semantic/keyword/graph dict to `context.strategy_results` (what
`FusionStage` fuses) and a separately-scoped, richer dict -- one that also
carries a `tag_expansion` strategy when tag expansion contributed results
-- to that metadata key, since provenance and fusion need different views
of "strategy results" at the same point in the pipeline.
"""

from __future__ import annotations

from searchkernel.domain import SearchResultProvenance
from searchkernel.pipeline.stage import SearchContext, replace_state

_RESULT_PROVENANCE_KEY = "result_provenance"
_STRATEGY_RESULTS_OVERRIDE_KEY = "provenance_strategy_results"


class ProvenanceStage:
    """Build `context.state["result_provenance"]` from strategy results.

    Expects `context.strategy_results` (`dict[str, list[tuple[str, float]]]`,
    keyed by strategy name) or, when present,
    `context.state["provenance_strategy_results"]` (same shape,
    preferred over `context.strategy_results`). Writes
    `context.state["result_provenance"]` (`dict[str, SearchResultProvenance]`).
    """

    name = "provenance"

    def run(self, context: SearchContext) -> SearchContext:
        strategy_results = (
            context.state.provenance_strategy_results
            if _STRATEGY_RESULTS_OVERRIDE_KEY in context.state
            else context.strategy_results
        )
        result_provenance: dict[str, SearchResultProvenance] = {}

        for strategy, result_list in strategy_results.items():
            for rank, (chunk_id, raw_score) in enumerate(result_list, start=1):
                provenance = result_provenance.setdefault(
                    chunk_id,
                    SearchResultProvenance(),
                )
                provenance.add_strategy(strategy, rank, raw_score)

        metadata = dict(context.state)
        metadata[_RESULT_PROVENANCE_KEY] = result_provenance
        return replace_state(context, metadata)
