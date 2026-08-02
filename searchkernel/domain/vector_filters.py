"""Canonical vector eligibility filters shared by local and SQL stores."""

import json
from collections.abc import Mapping
from typing import Any

from searchkernel.domain.models import RecordIdentity, RecordStatus


def filter_values(value: Any) -> list[Any]:
    if isinstance(value, (str, RecordIdentity, RecordStatus)):
        return [value]
    if value is None:
        return []
    return list(value)


def metadata_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {}


def status_values(filters: Mapping[str, Any] | None) -> set[str]:
    filters = filters or {}
    if filters.get("include_inactive"):
        return {status.value for status in RecordStatus}
    if "status" in filters and filters["status"] is not None:
        values = [filters["status"]]
    elif "lifecycle_status" in filters and filters["lifecycle_status"] is not None:
        values = [filters["lifecycle_status"]]
    elif "statuses" in filters and filters["statuses"] is not None:
        values = filter_values(filters["statuses"])
    elif "lifecycle_statuses" in filters and filters["lifecycle_statuses"] is not None:
        values = filter_values(filters["lifecycle_statuses"])
    else:
        values = [RecordStatus.ACTIVE]
    return {
        value.value if isinstance(value, RecordStatus) else str(value)
        for value in values
    }


def candidate_storage_keys(value: Any) -> set[str]:
    """Return only canonical keys accepted by internal candidate filters."""
    storage_keys: set[str] = set()
    for item in filter_values(value):
        if isinstance(item, RecordIdentity):
            storage_keys.add(item.storage_key)
        elif isinstance(item, str) and item.startswith("record:"):
            try:
                storage_keys.add(RecordIdentity.from_storage_key(item).storage_key)
            except ValueError:
                continue
    return storage_keys


def _string_values(
    filters: Mapping[str, Any],
    *names: str,
) -> list[str] | None:
    for name in names:
        if name in filters and filters[name] is not None:
            return [
                value.value if isinstance(value, RecordStatus) else str(value)
                for value in filter_values(filters[name])
            ]
    return None


def _path_variants(value: Any) -> set[str]:
    normalized = str(value).replace("\\", "/")
    variants = {normalized}
    without_suffix = normalized.rsplit("/", 1)
    leaf = without_suffix[-1]
    if "." in leaf:
        variants.add(normalized[: -len(leaf.rsplit(".", 1)[-1]) - 1])
        variants.add("/".join(without_suffix[:-1] + [leaf.rsplit(".", 1)[0]]))
    variants.add(leaf)
    if "." in leaf:
        variants.add(leaf.rsplit(".", 1)[0])
    return variants


def _path_values(uri: str | None, metadata: Mapping[str, Any]) -> set[str]:
    for key in ("file_path", "path", "source_file"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return _path_variants(value)
    if uri:
        return _path_variants(uri)
    return set()


def record_matches_vector_filters(
    *,
    storage_key: str,
    source_id: str,
    workspace_id: str | None,
    source_kind: str,
    status: str | RecordStatus,
    metadata: Mapping[str, Any] | None = None,
    uri: str | None = None,
    filters: Mapping[str, Any] | None = None,
) -> bool:
    filters = filters or {}
    metadata = metadata_mapping(metadata)
    normalized_status = (
        status.value if isinstance(status, RecordStatus) else str(status)
    )
    if normalized_status not in status_values(filters):
        return False

    requested_workspace = filters.get("workspace_id")
    if requested_workspace is not None and workspace_id != requested_workspace:
        return False

    source_kinds = _string_values(
        filters, "source_kinds", "source_kind", "source_filter"
    )
    if source_kinds is not None and source_kind not in set(source_kinds):
        return False

    candidate_value = filters.get("candidate_ids")
    if candidate_value is None:
        candidate_value = filters.get("candidate_storage_keys")
    if candidate_value is not None:
        candidate_keys = candidate_storage_keys(candidate_value)
        if not candidate_keys:
            return False
        if storage_key not in candidate_keys:
            return False

    project_id = metadata.get("project_id")
    project_values = _string_values(
        filters, "project_ids", "project_id", "project_filter"
    )
    if project_values == [] and "project_filter" in filters:
        project_values = None
    if project_values is not None and str(project_id) not in set(project_values):
        return False
    excluded_projects = _string_values(
        filters, "excluded_projects", "excluded_project_ids"
    )
    if (
        excluded_projects is not None
        and project_id is not None
        and str(project_id) in set(excluded_projects)
    ):
        return False

    path_values = _path_values(uri, metadata)
    included_paths = _string_values(
        filters,
        "paths",
        "file_paths",
        "source_files",
        "path",
        "file_path",
        "source_file",
    )
    if included_paths is not None:
        expected = set().union(*(_path_variants(value) for value in included_paths))
        if not path_values.intersection(expected):
            return False

    excluded_paths = _string_values(
        filters,
        "excluded_files",
        "excluded_paths",
        "excluded_file_paths",
        "excluded_source_files",
    )
    if excluded_paths is not None:
        excluded = set().union(*(_path_variants(value) for value in excluded_paths))
        if path_values.intersection(excluded):
            return False

    document_id = metadata.get("doc_id", source_id)
    document_values = _string_values(
        filters, "document_ids", "document_id", "doc_ids", "doc_id"
    )
    if document_values is not None and str(document_id) not in set(document_values):
        return False
    excluded_documents = _string_values(
        filters,
        "excluded_documents",
        "excluded_document_ids",
        "excluded_doc_ids",
    )
    if excluded_documents is not None:
        excluded = set().union(
            *(_path_variants(value) for value in excluded_documents)
        )
        if str(document_id) in excluded or source_id in excluded:
            return False

    metadata_equals = filters.get("metadata_equals")
    if metadata_equals is not None:
        for field, value in metadata_equals.items():
            if value is not None and str(metadata.get(field)) != str(value):
                return False

    return True
