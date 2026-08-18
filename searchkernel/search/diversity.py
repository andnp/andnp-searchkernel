"""Maximal marginal relevance battery for ``RecordSearchPolicy.post_process``.

Nothing in the pipeline stops the top-k from being ten near-identical chunks
of one document once relevance sort has done its job. Maximal marginal
relevance (MMR) fixes that by trading relevance against similarity to what is
already selected, greedily:

    MMR(d) = lambda_ * Rel(d) - (1 - lambda_) * max(Sim(d, s) for s in selected)

``mmr_post_process`` builds a callable satisfying the existing
``RecordSearchPolicy.post_process`` hook. No new extension point is
introduced — this is a reference implementation for a socket that already
exists (see ``docs/retrieval-algorithm-design.md``).

Scale alignment (the one decision that matters here)
------------------------------------------------------
``RecordSearchResult`` carries two scores: ``score``, the raw fused value
(reciprocal-rank fusion increments are typically ~0.008-0.05), and
``normalized_score``, which the pipeline assigns last via
``normalize_scores`` and which is query-relative in ``[0, 1]``.

``Sim`` here is cosine similarity between embeddings, which — after the
clamp applied below — also lives in ``[0, 1]``. If ``Rel`` were the raw
fused ``score`` instead, its magnitude (~0.01-0.05) would be dwarfed by the
similarity penalty at every ``lambda_`` below roughly 0.98: the diversity
term would dominate the selection order regardless of the requested
``lambda_``, and the knob would look inert right up until it suddenly
wasn't — the same trap the project already hit once when mixing raw and
calibrated scores in fusion. Using ``normalized_score`` for ``Rel`` keeps
both terms on the same ``[0, 1]`` scale, so ``lambda_`` means what it says.

Because ``normalized_score`` is assigned only once, at the very end of
``_refine_results``, this callable must run there — which is exactly what
``post_process`` guarantees: it is the last hook to see the result list
before final truncation.

Missing embeddings
------------------
``embedding_of`` may return ``None`` for a result the caller cannot embed
(no cached vector, an unembeddable record kind, and so on). MMR needs an
embedding to compute both `Rel`'s scale-mate `Sim` and to be diversified
against everything already chosen, so a result without one cannot
participate in the greedy selection at all. These results are never
dropped — they are returned in their original relative order, appended
after the diversified prefix. Appending (rather than, say, interleaving them
by score) keeps the output easy to reason about: the front of the list is
always the greedy MMR order, and anything the algorithm structurally cannot
diversify trails behind it rather than silently overriding a diversified
pick.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from searchkernel.ports.search_results import RecordSearchResult


def mmr_post_process(
    embedding_of: Callable[[RecordSearchResult], Sequence[float] | None],
    *,
    lambda_: float = 0.7,
) -> Callable[[list[RecordSearchResult]], list[RecordSearchResult]]:
    """Build a ``post_process`` callable that greedily diversifies by MMR.

    ``embedding_of`` maps a result to its embedding, or ``None`` when one is
    unavailable. ``lambda_`` trades relevance (1.0) against diversity (0.0);
    must be in ``[0.0, 1.0]``.
    """
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError("lambda_ must be between 0.0 and 1.0")

    def _post_process(
        results: list[RecordSearchResult],
    ) -> list[RecordSearchResult]:
        if len(results) <= 1:
            return list(results)

        embeddings: list[np.ndarray | None] = []
        for result in results:
            vector = embedding_of(result)
            if vector is None:
                embeddings.append(None)
                continue
            embeddings.append(np.asarray(vector, dtype=np.float64))

        embeddable = [
            position
            for position, vector in enumerate(embeddings)
            if vector is not None
        ]
        skipped = [
            position
            for position, vector in enumerate(embeddings)
            if vector is None
        ]

        if not embeddable:
            return list(results)

        matrix = np.stack([embeddings[position] for position in embeddable])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        similarity = normalized @ normalized.T
        similarity = np.clip(similarity, -1.0, 1.0)

        relevance = np.array(
            [results[position].normalized_score for position in embeddable],
            dtype=np.float64,
        )

        remaining = set(range(len(embeddable)))
        selected: list[int] = []

        first = min(
            remaining,
            key=lambda idx: (-relevance[idx], embeddable[idx]),
        )
        selected.append(first)
        remaining.discard(first)

        while remaining:
            best_idx = None
            best_key: tuple[float, int] | None = None
            for idx in remaining:
                max_sim = max(similarity[idx, s] for s in selected)
                mmr_score = lambda_ * relevance[idx] - (1.0 - lambda_) * max_sim
                key = (-mmr_score, embeddable[idx])
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            assert best_idx is not None
            selected.append(best_idx)
            remaining.discard(best_idx)

        diversified = [results[embeddable[idx]] for idx in selected]
        trailing = [results[position] for position in skipped]
        return [*diversified, *trailing]

    return _post_process
