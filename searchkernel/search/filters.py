from collections.abc import Callable, Hashable, Sequence


def filter_by_confidence(
    results: list[tuple[str, float]],
    threshold: float = 0.0,
) -> list[tuple[str, float]]:
    if threshold <= 0.0:
        return results
    return [(chunk_id, score) for chunk_id, score in results if score >= threshold]


def limit_per_group[ResultT, GroupKeyT: Hashable](
    results: Sequence[ResultT],
    group_key: Callable[[ResultT], GroupKeyT],
    max_per_group: int = 0,
) -> list[ResultT]:
    """Limit ranked results per caller-defined group, preserving order."""
    if max_per_group <= 0:
        return list(results)

    group_counts: dict[GroupKeyT, int] = {}
    limited: list[ResultT] = []
    for result in results:
        group = group_key(result)
        current_count = group_counts.get(group, 0)
        if current_count >= max_per_group:
            continue
        limited.append(result)
        group_counts[group] = current_count + 1
    return limited


def limit_per_document(
    results: list[tuple[str, float]],
    max_per_doc: int = 0,
) -> list[tuple[str, float]]:
    """Compatibility wrapper for the legacy chunk-id document grouping."""
    if max_per_doc <= 0:
        return results

    return limit_per_group(
        results,
        lambda result: (
            result[0].rsplit("_chunk_", 1)[0] if "_chunk_" in result[0] else result[0]
        ),
        max_per_doc,
    )


def normalize_project_filter(
    project_filter: list[str] | tuple[str, ...] | set[str] | None,
) -> set[str] | None:
    if not project_filter:
        return None

    normalized = {item.strip() for item in project_filter if item and item.strip()}
    return normalized or None


def matches_project_filter(
    project_id: str | None,
    project_filter: set[str] | None,
) -> bool:
    if project_filter is None:
        return True
    if project_id is None:
        return False
    return project_id in project_filter
