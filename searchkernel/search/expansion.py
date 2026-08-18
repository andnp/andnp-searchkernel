"""Query expansion batteries for ``RecordSearchPolicy.query_expander``.

``RecordSearchPolicy.query_expander`` exists and, before this module, ships no
reference implementation — a socket with no battery (see
``docs/retrieval-algorithm-design.md``). Two are provided here: a
caller-supplied synonym map needing no model, and HyDE (Hypothetical Document
Embeddings), which asks a model to write a short hypothetical answer and
expands the query with it on the theory that an answer sits closer in
embedding space to real answers than the bare question does.

Both factories return a plain ``Callable[[str], str]``, which is a subset of
what ``RecordSearchPolicy.query_expander`` accepts (it also allows an
async callable or a ``Sequence[str]`` return) — a sync, single-string return
is the simplest shape that satisfies the contract, so that is what these
batteries produce.

Contract reminder, read from ``RecordSearchPipeline._expand_query`` and
``_normalize_query_expansion`` in ``searchkernel/search/record_pipeline.py``:
the pipeline only calls the expander when ``RecordSearchConfig.
synonym_expansion_enabled`` is set, wraps the call in
``asyncio.wait_for(..., timeout=expansion_timeout_s)``, and feeds a
``str`` return through ``_normalize_query_expansion``, which:

- Takes only the first ``RecordSearchConfig.synonym_expansion_max_terms``
  *words* (default 3) of the returned string as a single glued-on term.
- Appends that term to the original query, unless it is identical to the
  original query (case-insensitively), in which case expansion is treated
  as a no-op and the pipeline falls back to its own (non-policy) expansion
  path.
- The result only ever feeds a *keyword* re-acquisition
  (``CandidateAcquirer.keyword``) — this hook does not re-embed the query
  for vector search. Consumers who want a HyDE passage to inform vector
  retrieval need an application-level integration beyond this hook; this
  battery satisfies the ``query_expander`` contract as specified, not a
  full re-embedding pipeline.

Both factories are pure with respect to their own inputs and do no I/O of
their own; neither is wired into any default.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence

_TOKEN_PATTERN = re.compile(r"\w+")

# A defensive bound on how many distinct synonym terms one call appends,
# independent of the pipeline's own word-count cap
# (``RecordSearchConfig.synonym_expansion_max_terms``). The pipeline's cap
# operates on the *final* glued string by word count, so a single very long
# synonym could consume the whole budget; capping the term *count* here
# keeps the output bounded even before that truncation applies, in case a
# caller's synonym map has unusually large value lists.
_MAX_APPENDED_TERMS = 20


def synonym_expander(
    synonyms: Mapping[str, Sequence[str]],
) -> Callable[[str], str]:
    """Build a ``query_expander`` that appends caller-supplied synonyms.

    Matching is case-insensitive: query tokens (``\\w+`` runs) and mapping
    keys are compared by ``casefold()``, but synonym values are appended
    verbatim, preserving whatever casing the caller supplied. Tokens are
    scanned left to right and every match's synonyms are appended (no
    "first match wins" shortcut), deduplicated against synonyms already
    added (including a synonym equal to the token that produced it, which
    would just echo the query back). Term order is therefore stable and
    deterministic: query order first, then the caller's per-term sequence
    order — never a set iteration.

    Output is bounded twice: once here, capping the number of distinct
    synonym terms appended at ``_MAX_APPENDED_TERMS`` (an unbounded synonym
    map should not produce an unbounded query), and again downstream by
    ``RecordSearchConfig.synonym_expansion_max_terms``, which truncates the
    final string by word count before it reaches retrieval. Consumers
    wanting longer synonym additions to survive should raise that config
    value.

    When no query token matches the map, the query is returned completely
    unchanged (identical string, not just case-insensitively equal) — this
    is deliberate, not an oversight: the pipeline's
    ``_normalize_query_expansion`` treats a returned value that is
    unchanged (case-insensitively) as "no expansion" and skips the extra
    keyword retrieval, which is exactly the right behavior for a query with
    nothing to expand.

    Raises ``ValueError`` if ``synonyms`` is empty.
    """
    if not synonyms:
        raise ValueError("synonyms must not be empty")
    normalized_map = {term.casefold(): list(values) for term, values in synonyms.items()}

    def expand(query: str) -> str:
        seen: set[str] = set()
        additions: list[str] = []
        for token in _TOKEN_PATTERN.findall(query):
            candidates = normalized_map.get(token.casefold())
            if not candidates:
                continue
            for candidate in candidates:
                key = candidate.casefold()
                if key in seen or key == token.casefold():
                    continue
                seen.add(key)
                additions.append(candidate)
                if len(additions) >= _MAX_APPENDED_TERMS:
                    break
            if len(additions) >= _MAX_APPENDED_TERMS:
                break
        if not additions:
            return query
        return f"{query} {' '.join(additions)}"

    return expand


_HYPOTHETICAL_ANSWER_PROMPT = (
    "Write a short, plausible passage that would directly answer the "
    "following question, as if it were an excerpt from a real document. "
    "Do not mention that this is hypothetical or refer to the question "
    "itself.\n\n"
    "Question: {query}"
)


def hypothetical_answer_expander(
    complete: Callable[[str], str],
    *,
    max_chars: int = 2000,
) -> Callable[[str], str]:
    """Expand a query with a model-written hypothetical answer.

    The idea comes from HyDE (Hypothetical Document Embeddings): a
    plausible answer sits closer to real answers than the question does.
    This is deliberately NOT named for it, because HyDE embeds the
    hypothetical passage and searches the vector lane with it, and this
    hook cannot: the pipeline keeps only the first few words of what is
    returned and sends them to the lexical lane alone. Calling this HyDE
    would promise retrieval behaviour it does not deliver. ``complete`` takes the same shape ``LLMJudgeReranker`` uses
    (``searchkernel/adapters/rerank/llm_judge.py``): a plain
    ``Callable[[str], str]``, so no LLM client library becomes a dependency
    of this module or its caller.

    Output is bounded at ``max_chars`` characters (default 2000): an
    unbounded generation is worse than none for this hook — it is both more
    expensive to carry through diagnostics/logging and, per the
    ``synonym_expansion_max_terms`` word-count truncation described in this
    module's docstring, only the leading words of it ever survive to
    retrieval anyway, so there is no benefit to an unbounded answer.

    Failure behavior: exceptions raised by ``complete`` are **not** caught
    here — they propagate to the caller. This mirrors ``LLMJudgeReranker``,
    which does not catch its own ``complete`` failures either, leaving
    fallback behavior to the single place that already owns it:
    ``RecordSearchPipeline._expand_query`` already wraps the whole
    expander call in ``asyncio.wait_for(timeout=expansion_timeout_s)`` and a
    broad ``except Exception``, recording a
    ``synonym_expansion:fallback:<ErrorType>`` diagnostic and continuing
    with the unexpanded query. Catching here too would just relabel the
    same failure one layer earlier, discard the real exception type from
    that diagnostic, and risk masking a caller bug (e.g. a misconfigured
    model client) as a silent no-op instead of a visible failure signal.

    Raises ``ValueError`` if ``max_chars`` is not positive.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")

    def expand(query: str) -> str:
        prompt = _HYPOTHETICAL_ANSWER_PROMPT.format(query=query)
        response = complete(prompt)
        return response[:max_chars]

    return expand
