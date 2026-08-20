from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_FTS_OPERATORS = frozenset({"AND", "NEAR", "NOT", "OR"})

_METADATA_KEYWORD_KEYS = (
    "tags",
    "keywords",
    "source_keywords",
    "aliases",
    "header_path",
    "headers",
    "file_path",
    "source_file",
    "path",
    "filename",
    "file_name",
    "paths",
    "filenames",
    "files_changed",
    "symbols",
    "symbol",
    "symbol_path",
    "symbol_paths",
    "file_tokens",
    "commit_file_tokens",
    "path_tokens",
    "symbol_tokens",
    "tokens",
    "exact_tokens",
)

_METADATA_URI_KEYS = (
    "uri",
    "source_file",
    "file_path",
    "path",
    "filename",
    "file_name",
)


def metadata_keyword_text(metadata: dict[str, Any]) -> str:
    """Derive artifact-scorer ``headers`` text from a record's metadata.

    Scrapes identifier-shaped fields (tags, paths, symbols, ...) out of a
    metadata dict so keyword artifact scorers can rerank against them without
    every store having to know the metadata schema itself.
    """
    values: list[str] = []
    for key in _METADATA_KEYWORD_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        values.extend(_metadata_keyword_values(value))
    return " ".join(" ".join(value.strip().lower().split()) for value in values if value)


def _metadata_keyword_values(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [
            item
            for key in sorted(value, key=str)
            for item in _metadata_keyword_values(value[key])
        ]
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [item for nested in items for item in _metadata_keyword_values(nested)]
    return [str(value)]


def metadata_uri(metadata: dict[str, Any]) -> str:
    """Derive a fallback URI from a record's metadata when the row has none."""
    for key in _METADATA_URI_KEYS:
        value = metadata.get(key)
        if value:
            return str(value)
    return ""


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
            for index, word in enumerate(words):
                if word.upper() in _FTS_OPERATORS:
                    terms.append(f'"{word}"')
                elif index == len(words) - 1 and prefix:
                    terms.append(word + "*")
                else:
                    terms.append(word)
    if not terms:
        return '""'
    return " ".join(terms)
