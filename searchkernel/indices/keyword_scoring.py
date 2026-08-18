from __future__ import annotations

import re

_FTS_OPERATORS = frozenset({"AND", "NEAR", "NOT", "OR"})


def sanitize_fts_query(query: str) -> str:
    """Sanitize a query string for safe use in FTS5 MATCH."""
    terms: list[str] = []
    for match in re.finditer(r'"([^"]*)"|(\S+)', query):
        quoted = match.group(1) is not None
        raw = match.group(1) if quoted else match.group(2)
        if raw is None:
            continue
        raw = raw.rstrip(".,;!?")
        prefix = not quoted and raw.rstrip().endswith("*")
        # FTS5 treats several punctuation characters as query syntax even
        # though they are common in natural-language input. Keep only word
        # characters and whitespace in bare terms; phrase and prefix intent
        # is restored below from the original token shape.
        sanitized = re.sub(r"[^\w\s]", " ", raw)
        words = sanitized.split()
        if not words:
            continue
        if quoted and len(words) > 1:
            terms.append(f'"{" ".join(words)}"')
        else:
            terms.extend(words[:-1])
            word = words[-1]
            if word.upper() in _FTS_OPERATORS:
                terms.append(f'"{word}"')
            else:
                terms.append(word + "*" if prefix else word)
    if not terms:
        return '""'
    return " ".join(terms)
