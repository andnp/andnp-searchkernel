import logging
import os
from functools import cache, lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_path(file_path: str, docs_root: Path):
    path = Path(file_path)

    if path.is_absolute():
        try:
            path = path.relative_to(docs_root)
        except ValueError:
            pass

    return str(path.with_suffix(""))


def matches_any_excluded(file_path: str, excluded_files: set[str], docs_root: Path):
    normalized = normalize_path(file_path, docs_root)

    if normalized in excluded_files:
        return True

    filename = Path(normalized).name
    return filename in excluded_files


def compute_doc_id(file_path: Path, docs_root: Path) -> str:
    """
    Compute document ID from file path relative to docs root.

    Args:
        file_path: Absolute path to the document file
        docs_root: Root directory for documentation

    Returns:
        Document ID (relative path with forward slashes, no extension)

    Example:
        >>> compute_doc_id(Path("/docs/guide/setup.md"), Path("/docs"))
        "guide/setup"
    """
    try:
        rel_path = file_path.relative_to(docs_root)
        # Remove extension and convert to forward slashes
        doc_id = str(rel_path.with_suffix("")).replace("\\", "/")
        return doc_id
    except ValueError:
        # file_path not under docs_root, use absolute path
        logger.warning(
            f"File {file_path} is outside docs root {docs_root}. "
            "Using absolute path as doc_id."
        )
        return str(file_path.with_suffix("")).replace("\\", "/")


def compute_doc_id_multi_root(file_path: Path, docs_roots: list[Path]) -> str:
    """Compute document ID relative to the common ancestor of multiple roots."""
    return _compute_doc_id_multi_root_cached(
        str(file_path), tuple(str(root) for root in docs_roots)
    )


@lru_cache(maxsize=200_000)
def _compute_doc_id_multi_root_cached(file_path: str, docs_roots: tuple[str, ...]) -> str:
    # Bounded because a long-lived daemon calls this for every link in every
    # rebuild, and the same (path, root-set) pairs recur heavily across a corpus.
    resolved_path = _canonical_path(file_path)
    common_root = _resolved_common_root(docs_roots)

    if common_root is None:
        return str(resolved_path.with_suffix("")).replace("\\", "/")

    resolved_str = str(resolved_path)
    for docs_root in _resolved_docs_roots(docs_roots):
        root_str = str(docs_root)
        prefix = root_str if root_str.endswith(os.sep) else root_str + os.sep
        if resolved_str == root_str or resolved_str.startswith(prefix):
            return compute_doc_id(resolved_path, common_root)

    _warn_outside_docs_roots(file_path, len(docs_roots))
    return compute_doc_id(resolved_path, common_root)


@lru_cache(maxsize=1024)
def _warn_outside_docs_roots(file_path: str, root_count: int) -> None:
    # Cached so a corpus-wide rebuild reports each stray path once instead of
    # once per link that mentions it.
    logger.warning(
        "File %s is outside the %d configured document roots. "
        "Falling back to common ancestor.",
        file_path,
        root_count,
    )


@lru_cache(maxsize=8192)
def _canonical_path(path: str) -> Path:
    # Bounded because a long-lived daemon canonicalizes every indexed file,
    # and the same paths recur across every link in a corpus.
    return Path(path).resolve()


@cache
def _resolved_docs_roots(docs_roots: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(Path(root).resolve() for root in docs_roots)


@cache
def _resolved_common_root(docs_roots: tuple[str, ...]) -> Path | None:
    resolved = _resolved_docs_roots(docs_roots)
    if not resolved:
        return None
    if len(resolved) == 1:
        return resolved[0]
    return Path(os.path.commonpath([str(root) for root in resolved])).resolve()


def _compute_common_docs_root(docs_roots: list[Path]) -> Path | None:
    # The document roots come from fixed configuration, so the canonicalized
    # ancestor is stable for the life of the process.
    return _resolved_common_root(tuple(str(root) for root in docs_roots))


def extract_doc_id_from_chunk_id(chunk_id: str) -> str:
    """
    Extract document ID from chunk ID.

    Handles the canonical hash-separated format:
    - "doc/path#chunk_0" → "doc/path"

    Args:
        chunk_id: Chunk identifier (with separator)

    Returns:
        Document ID (without chunk suffix)

    Example:
        >>> extract_doc_id_from_chunk_id("guide/setup#chunk_0")
        "guide/setup"
    """
    if "#" in chunk_id:
        return chunk_id.split("#")[0]

    logger.warning(f"Chunk ID '{chunk_id}' has unexpected format")
    return chunk_id


def resolve_doc_path(
    doc_id: str,
    docs_root: Path,
    extensions: list[str] | None = None,
) -> Path | None:
    """
    Resolve document ID back to absolute file path.

    Args:
        doc_id: Document identifier (relative path without extension)
        docs_root: Root directory for documentation
        extensions: File extensions to try (default: [".md", ".txt"])

    Returns:
        Absolute path if file exists, None otherwise

    Example:
        >>> resolve_doc_path("guide/setup", Path("/docs"))
        Path("/docs/guide/setup.md")  # if exists
    """
    if extensions is None:
        extensions = [".md", ".txt"]

    # Normalize doc_id (handle both forward and back slashes)
    normalized_id = doc_id.replace("\\", "/")

    for ext in extensions:
        candidate = docs_root / normalized_id
        candidate = candidate.with_suffix(ext)

        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def resolve_doc_path_multi_root(
    doc_id: str,
    docs_roots: list[Path],
    extensions: list[str] | None = None,
) -> Path | None:
    """Resolve a document ID across multiple roots.

    Tries the common-ancestor-relative document ID format first, then falls
    back to per-root-relative resolution for older manifest/doc_id formats.
    """
    common_root = _compute_common_docs_root(docs_roots)
    if common_root is not None:
        resolved = resolve_doc_path(doc_id, common_root, extensions)
        if resolved is not None:
            return resolved

    for docs_root in docs_roots:
        resolved = resolve_doc_path(doc_id, docs_root, extensions)
        if resolved is not None:
            return resolved
    return None
