"""Canonical vector eligibility filters shared by local and SQL stores."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from searchkernel.domain.models import RecordIdentity, RecordStatus

_METADATA_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


@dataclass(frozen=True, slots=True)
class SourceScopedFilter:
    """Authorization constraints owned by one source kind."""

    source_kind: str
    workspace_ids: frozenset[str] | None
    metadata_non_empty: tuple[str, ...]
    metadata_contains_any: tuple[tuple[str, frozenset[str]], ...]

    def matches(self, workspace_id: str | None, metadata: Mapping[str, Any]) -> bool:
        if self.workspace_ids is not None and workspace_id not in self.workspace_ids:
            return False
        for field in self.metadata_non_empty:
            value = metadata.get(field)
            if not isinstance(value, list) or not value:
                return False
        for field, allowed_values in self.metadata_contains_any:
            value = metadata.get(field)
            if not isinstance(value, list) or not any(
                isinstance(item, str) and item in allowed_values for item in value
            ):
                return False
        return True


def compile_source_scoped_filters(
    filters: Mapping[str, Any] | None,
) -> tuple[SourceScopedFilter, ...]:
    """Validate and compile source-scoped authorization constraints."""
    if not filters:
        return ()
    compiled: list[SourceScopedFilter] = []
    if "source_scoped_filters" in filters:
        source_filters = filters["source_scoped_filters"]
        if not isinstance(source_filters, Mapping):
            raise TypeError("source_scoped_filters must be a mapping")
        for source_kind, constraints in source_filters.items():
            if not isinstance(source_kind, str) or not source_kind:
                raise TypeError("source_scoped_filters keys must be non-empty strings")
            if not isinstance(constraints, Mapping):
                raise TypeError(
                    f"source_scoped_filters[{source_kind!r}] must be a mapping"
                )
            allowed_keys = {
                "workspace_ids",
                "metadata_non_empty",
                "metadata_contains_any",
            }
            if not set(constraints).intersection(allowed_keys) or not set(
                constraints
            ).issubset(allowed_keys):
                raise ValueError(
                    f"source_scoped_filters[{source_kind!r}] has invalid constraints"
                )
            workspace_ids = (
                _compile_string_list(constraints["workspace_ids"], "workspace_ids")
                if "workspace_ids" in constraints
                else None
            )
            metadata_non_empty = (
                _compile_metadata_keys(
                    constraints["metadata_non_empty"], "metadata_non_empty"
                )
                if "metadata_non_empty" in constraints
                else ()
            )
            metadata_contains_any = (
                _compile_metadata_contains_any(constraints["metadata_contains_any"])
                if "metadata_contains_any" in constraints
                else ()
            )
            compiled.append(
                SourceScopedFilter(
                    source_kind,
                    workspace_ids,
                    metadata_non_empty,
                    metadata_contains_any,
                )
            )

    return tuple(compiled)


def _compile_string_list(
    value: Any,
    name: str,
) -> frozenset[str]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must contain only strings")
    return frozenset(value)


def _compile_metadata_keys(
    value: Any,
    name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    keys = tuple(value)
    if not all(isinstance(key, str) and _METADATA_FIELD_RE.fullmatch(key) for key in keys):
        raise ValueError(f"{name} must contain valid metadata keys")
    return keys


def _compile_metadata_contains_any(
    value: Any,
) -> tuple[tuple[str, frozenset[str]], ...]:
    if not isinstance(value, Mapping):
        raise TypeError("metadata_contains_any must be a mapping")
    compiled: list[tuple[str, frozenset[str]]] = []
    for key, values in value.items():
        _compile_metadata_keys([key], "metadata_contains_any field")
        compiled.append(
            (
                key,
                _compile_string_list(values, f"metadata_contains_any[{key!r}]")
                or frozenset(),
            )
        )
    return tuple(compiled)


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


@dataclass(frozen=True, slots=True)
class CompiledVectorFilter:
    statuses: frozenset[str]
    workspace_id: Any
    source_kinds: frozenset[str] | None
    candidate_keys: frozenset[str] | None
    excluded_storage_keys: frozenset[str] | None
    project_values: frozenset[str] | None
    excluded_projects: frozenset[str] | None
    included_paths: frozenset[str] | None
    excluded_paths: frozenset[str] | None
    document_values: frozenset[str] | None
    excluded_documents: frozenset[str] | None
    metadata_equals: tuple[tuple[str, str], ...] | None
    metadata_in: tuple[tuple[str, frozenset[str]], ...] | None
    source_scoped_filters: tuple[SourceScopedFilter, ...]

    def matches(
        self,
        *,
        storage_key: str,
        source_id: str,
        workspace_id: str | None,
        source_kind: str,
        status: str | RecordStatus,
        metadata: Mapping[str, Any] | None = None,
        uri: str | None = None,
    ) -> bool:
        normalized_status = (
            status.value if isinstance(status, RecordStatus) else str(status)
        )
        if normalized_status not in self.statuses:
            return False
        if self.workspace_id is not None and workspace_id != self.workspace_id:
            return False
        if self.source_kinds is not None and source_kind not in self.source_kinds:
            return False
        if self.candidate_keys is not None and storage_key not in self.candidate_keys:
            return False
        if (
            self.excluded_storage_keys is not None
            and storage_key in self.excluded_storage_keys
        ):
            return False

        metadata = metadata_mapping(metadata)
        scoped_filters = [
            source_filter
            for source_filter in self.source_scoped_filters
            if source_filter.source_kind == source_kind
        ]
        if scoped_filters and not all(
            source_filter.matches(workspace_id, metadata)
            for source_filter in scoped_filters
        ):
            return False
        project_id = metadata.get("project_id")
        if self.project_values is not None and str(project_id) not in self.project_values:
            return False
        if (
            self.excluded_projects is not None
            and project_id is not None
            and str(project_id) in self.excluded_projects
        ):
            return False

        path_values = _path_values(uri, metadata)
        if self.included_paths is not None and not path_values.intersection(
            self.included_paths
        ):
            return False
        if self.excluded_paths is not None and path_values.intersection(
            self.excluded_paths
        ):
            return False

        document_id = metadata.get("doc_id", source_id)
        if self.document_values is not None and str(document_id) not in self.document_values:
            return False
        if self.excluded_documents is not None and (
            str(document_id) in self.excluded_documents
            or source_id in self.excluded_documents
        ):
            return False
        if self.metadata_equals is not None:
            for field, value in self.metadata_equals:
                if str(metadata.get(field)) != value:
                    return False
        if self.metadata_in is not None:
            for field, allowed_values in self.metadata_in:
                if str(metadata.get(field)) not in allowed_values:
                    return False
        return True


def compile_vector_filters(
    filters: Mapping[str, Any] | None,
) -> CompiledVectorFilter:
    filters = filters or {}
    source_scoped_filters = compile_source_scoped_filters(filters)

    def string_set(*names: str) -> frozenset[str] | None:
        values = _string_values(filters, *names)
        return None if values is None else frozenset(values)

    included_paths = _string_values(
        filters,
        "paths",
        "file_paths",
        "source_files",
        "path",
        "file_path",
        "source_file",
    )
    excluded_paths = _string_values(
        filters,
        "excluded_files",
        "excluded_paths",
        "excluded_file_paths",
        "excluded_source_files",
    )

    def path_set(values: list[str] | None) -> frozenset[str] | None:
        if values is None:
            return None
        return frozenset().union(*(_path_variants(value) for value in values))

    document_values = string_set("document_ids", "document_id", "doc_ids", "doc_id")
    excluded_documents = _string_values(
        filters,
        "excluded_documents",
        "excluded_document_ids",
        "excluded_doc_ids",
    )
    project_values = string_set("project_ids", "project_id", "project_filter")
    if project_values == frozenset() and "project_filter" in filters:
        project_values = None
    metadata_equals = filters.get("metadata_equals")
    compiled_metadata = (
        tuple(
            (field, str(value))
            for field, value in metadata_equals.items()
            if value is not None
        )
        if metadata_equals is not None
        else None
    )
    metadata_in = filters.get("metadata_in")
    compiled_metadata_in = (
        tuple(
            (field, frozenset(str(value) for value in values))
            for field, values in metadata_in.items()
        )
        if metadata_in is not None
        else None
    )
    candidate_value = filters.get("candidate_ids")
    if candidate_value is None:
        candidate_value = filters.get("candidate_storage_keys")
    candidate_keys = (
        frozenset(candidate_storage_keys(candidate_value))
        if candidate_value is not None
        else None
    )
    exclude_value = filters.get("exclude_storage_keys")
    excluded_storage_keys = (
        frozenset(candidate_storage_keys(exclude_value))
        if exclude_value is not None
        else None
    )
    return CompiledVectorFilter(
        statuses=frozenset(status_values(filters)),
        workspace_id=filters.get("workspace_id"),
        source_kinds=string_set("source_kinds", "source_kind", "source_filter"),
        candidate_keys=candidate_keys,
        excluded_storage_keys=excluded_storage_keys,
        project_values=project_values,
        excluded_projects=string_set("excluded_projects", "excluded_project_ids"),
        included_paths=path_set(included_paths),
        excluded_paths=path_set(excluded_paths),
        document_values=document_values,
        excluded_documents=(
            frozenset(excluded_documents) if excluded_documents is not None else None
        ),
        metadata_equals=compiled_metadata,
        metadata_in=compiled_metadata_in,
        source_scoped_filters=source_scoped_filters,
    )


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
    return compile_vector_filters(filters).matches(
        storage_key=storage_key,
        source_id=source_id,
        workspace_id=workspace_id,
        source_kind=source_kind,
        status=status,
        metadata=metadata,
        uri=uri,
    )
