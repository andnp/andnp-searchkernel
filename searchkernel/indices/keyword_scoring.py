from __future__ import annotations

import re
from pathlib import Path

_ARTIFACT_QUERY_RE = re.compile(r"[./\\_-]")
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


def normalize_artifact_value(value: str) -> str:
    return value.strip().lower().replace("\\", "/")


def normalize_field_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def has_phrase_boundary_match(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return (
        text == phrase
        or text.startswith(f"{phrase} ")
        or text.endswith(f" {phrase}")
        or f" {phrase} " in text
    )


def split_header_segments(headers: str) -> list[str]:
    return [
        normalized
        for segment in headers.split(">")
        if (normalized := normalize_field_text(segment))
    ]


def score_title_locality(normalized_query: str, normalized_title: str) -> float:
    if not normalized_title:
        return 0.0
    if normalized_title == normalized_query:
        return 80.0
    if has_phrase_boundary_match(normalized_title, normalized_query):
        return 24.0
    if normalized_query in normalized_title:
        return 14.0
    return 0.0


def score_header_locality(
    normalized_query: str,
    normalized_headers: str,
    header_segments: list[str],
) -> float:
    score = 0.0

    if normalized_headers == normalized_query:
        score = max(score, 34.0)
    elif has_phrase_boundary_match(normalized_headers, normalized_query):
        score = max(score, 14.0)
    elif normalized_query in normalized_headers:
        score = max(score, 10.0)

    for depth, segment in enumerate(header_segments):
        depth_decay = max(0.45, 1.0 - (depth * 0.25))
        if segment == normalized_query:
            score = max(score, 112.0 * depth_decay)
            continue
        if has_phrase_boundary_match(segment, normalized_query):
            score = max(score, 20.0 * depth_decay)
            continue
        if normalized_query in segment:
            score = max(score, 10.0 * depth_decay)

    return score


def looks_like_artifact_query(query: str) -> bool:
    normalized = query.strip()
    if not normalized:
        return False
    if len(normalized.split()) == 1:
        return _ARTIFACT_QUERY_RE.search(normalized) is not None
    return bool(_embedded_artifact_tokens(normalized))


def _embedded_artifact_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in query.split():
        token = normalize_artifact_value(raw_token.strip(".,;!?()[]{}<>\"'`"))
        if not token:
            continue
        if (
            "/" in token
            or "\\" in token
            or ":" in token
            or re.search(r"\.[a-z0-9]{1,12}$", token) is not None
        ):
            tokens.append(token)
    return tokens


def score_field_aware_match(
    query: str,
    *,
    content: str,
    title: str,
    headers: str,
    source_file: str,
) -> float:
    normalized_query = normalize_field_text(query)
    if not normalized_query:
        return 0.0

    normalized_title = normalize_field_text(title)
    normalized_headers = normalize_field_text(headers)
    normalized_content = normalize_field_text(content)
    normalized_source = normalize_artifact_value(source_file)
    normalized_query_artifact = normalize_artifact_value(query)
    header_segments = split_header_segments(headers)

    score = 0.0

    score += score_title_locality(normalized_query, normalized_title)
    score += score_header_locality(
        normalized_query,
        normalized_headers,
        header_segments,
    )

    artifact_queries = _embedded_artifact_tokens(query) or [normalized_query_artifact]
    for artifact_query in artifact_queries:
        score = max(score, score_title_locality(artifact_query, normalized_title))
        score = max(
            score,
            score_header_locality(
                artifact_query,
                normalized_headers,
                header_segments,
            ),
        )
        basename_query = Path(artifact_query).name
        source_basename = Path(normalized_source).name if normalized_source else ""
        if normalized_source == artifact_query:
            score += 60.0
        elif source_basename == artifact_query:
            score += 56.0
        elif basename_query and source_basename == basename_query:
            score += 52.0
        elif normalized_source.endswith(f"/{artifact_query}"):
            score += 48.0
        elif basename_query and normalized_source.endswith(f"/{basename_query}"):
            score += 44.0
        elif artifact_query in normalized_source:
            score += 16.0
        elif basename_query and basename_query in normalized_source:
            score += 12.0

    if normalized_query in normalized_content:
        score += 4.0

    return score


def score_artifact_match(
    normalized_query: str,
    basename_query: str,
    content: str,
    title: str,
    headers: str,
    source_file: str,
) -> float:
    queries = _embedded_artifact_tokens(normalized_query)
    if queries and queries != [normalized_query]:
        return max(
            score_artifact_match(
                query,
                Path(query).name,
                content,
                title,
                headers,
                source_file,
            )
            for query in queries
        )
    normalized_content = normalize_artifact_value(content)
    normalized_headers = normalize_artifact_value(headers)
    normalized_title = normalize_artifact_value(title)
    normalized_source = normalize_artifact_value(source_file)
    source_basename = Path(normalized_source).name if normalized_source else ""

    score = 0.0

    if normalized_source == normalized_query:
        score = max(score, 120.0)
    if source_basename == normalized_query:
        score = max(score, 115.0)
    if basename_query and source_basename == basename_query:
        score = max(score, 112.0)
    if normalized_source.endswith(f"/{normalized_query}"):
        score = max(score, 108.0)
    if basename_query and normalized_source.endswith(f"/{basename_query}"):
        score = max(score, 104.0)
    if normalized_title == normalized_query:
        score = max(score, 102.0)
    if basename_query and normalized_title == basename_query:
        score = max(score, 100.0)
    if normalized_query in normalized_source:
        score = max(score, 96.0)
    if basename_query and basename_query in normalized_source:
        score = max(score, 92.0)
    if normalized_query in normalized_title:
        score = max(score, 88.0)
    if basename_query and basename_query in normalized_title:
        score = max(score, 84.0)
    if normalized_query in normalized_headers:
        score = max(score, 82.0)
    if basename_query and basename_query in normalized_headers:
        score = max(score, 78.0)
    if normalized_query in normalized_content:
        score = max(score, 74.0)
    if basename_query and basename_query in normalized_content:
        score = max(score, 70.0)

    return score
