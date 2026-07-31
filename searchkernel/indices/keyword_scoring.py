from __future__ import annotations

import re
from pathlib import Path

_ARTIFACT_QUERY_RE = re.compile(r"[./\\_-]")


def sanitize_fts_query(query: str) -> str:
    """Sanitize a query string for safe use in FTS5 MATCH."""
    sanitized = re.sub(r"[\"\'*\^(){}[\]<>|~!:\-]", " ", query)
    sanitized = sanitized.strip()
    if not sanitized:
        return '""'
    return " ".join(sanitized.split())


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
            score = max(score, 44.0 * depth_decay)
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
    return _ARTIFACT_QUERY_RE.search(normalized) is not None


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
    basename_query = Path(normalized_query_artifact).name
    source_basename = Path(normalized_source).name if normalized_source else ""
    header_segments = split_header_segments(headers)

    score = 0.0

    score += score_title_locality(normalized_query, normalized_title)
    score += score_header_locality(
        normalized_query,
        normalized_headers,
        header_segments,
    )

    if normalized_source == normalized_query_artifact:
        score += 60.0
    elif source_basename == normalized_query_artifact:
        score += 56.0
    elif basename_query and source_basename == basename_query:
        score += 52.0
    elif normalized_source.endswith(f"/{normalized_query_artifact}"):
        score += 48.0
    elif basename_query and normalized_source.endswith(f"/{basename_query}"):
        score += 44.0
    elif normalized_query_artifact in normalized_source:
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
