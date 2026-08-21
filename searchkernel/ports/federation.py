"""Versioned, transport-neutral contracts for federated search sources."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol, Self, runtime_checkable

from searchkernel.domain import RecordIdentity, RecordStatus

FEDERATION_CONTRACT_VERSION = "v1"
FederationEventKind = Literal["source", "provisional", "authoritative"]
MAX_QUERY_LENGTH = 4_096
MAX_RERANK_TEXT_LENGTH = 4_096
MAX_SNIPPET_LENGTH = 8_192
MAX_TOP_K = 1_000

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    unexpected = set(value) - expected
    if unexpected:
        raise ValueError(f"{name} contains unknown fields: {sorted(unexpected)}")


def _required(value: Mapping[str, object], key: str, name: str) -> object:
    if key not in value:
        raise ValueError(f"{name} is missing required field: {key}")
    return value[key]


def _string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _record_status(value: object, name: str) -> RecordStatus:
    if isinstance(value, RecordStatus):
        return value
    if isinstance(value, str):
        return RecordStatus(value)
    raise TypeError(f"{name} must be a record status")


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{name} must be an array of strings")
    result = tuple(_string(item, f"{name} item") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _json_value(value: object, name: str = "value") -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} object keys must be strings")
        return {
            key: _json_value(item, f"{name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, f"{name} item") for item in value]
    raise TypeError(f"{name} must contain only JSON-compatible values")


def _json_object(value: object, name: str) -> dict[str, JsonValue]:
    result = _json_value(value, name)
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a JSON object")
    return result


def _positive_int(value: object, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _timestamp(value: object, name: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime or ISO-8601 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include timezone information")
    return value.astimezone(UTC)


def _optional_timestamp(value: object, name: str) -> datetime | None:
    if value is None:
        return None
    return _timestamp(value, name)


def _timestamp_dict(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _loads(value: str, name: str) -> Mapping[str, object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    return _require_mapping(decoded, name)


class _JsonContract:
    """Shared JSON helpers for the public wire models."""

    def to_dict(self) -> dict[str, JsonValue]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Self:
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, value: str) -> Self:
        return cls.from_dict(_loads(value, cls.__name__))


@dataclass(frozen=True, slots=True)
class SourceIdentity(_JsonContract):
    """Stable identity of a federated search source."""

    source_kind: str
    source_id: str
    workspace_id: str | None = None

    def __post_init__(self) -> None:
        _string(self.source_kind, "source_kind")
        _string(self.source_id, "source_id")
        if self.workspace_id is not None:
            _string(self.workspace_id, "workspace_id")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "workspace_id": self.workspace_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceIdentity:
        value = _require_mapping(value, "source")
        _require_exact_keys(value, {"source_kind", "source_id", "workspace_id"}, "source")
        return cls(
            source_kind=_string(_required(value, "source_kind", "source"), "source_kind"),
            source_id=_string(_required(value, "source_id", "source"), "source_id"),
            workspace_id=_optional_string(value.get("workspace_id"), "workspace_id"),
        )


@dataclass(frozen=True, slots=True)
class CallerAuthorizationContext(_JsonContract):
    """Caller identity and source-owned authorization context."""

    caller_id: str
    tenant_id: str | None = None
    scopes: tuple[str, ...] = ()
    claims: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _string(self.caller_id, "caller_id")
        if self.tenant_id is not None:
            _string(self.tenant_id, "tenant_id")
        _string_tuple(self.scopes, "scopes")
        object.__setattr__(self, "claims", _json_object(self.claims, "claims"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "caller_id": self.caller_id,
            "tenant_id": self.tenant_id,
            "scopes": list(self.scopes),
            "claims": dict(self.claims),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CallerAuthorizationContext:
        value = _require_mapping(value, "caller")
        _require_exact_keys(value, {"caller_id", "tenant_id", "scopes", "claims"}, "caller")
        return cls(
            caller_id=_string(_required(value, "caller_id", "caller"), "caller_id"),
            tenant_id=_optional_string(value.get("tenant_id"), "tenant_id"),
            scopes=_string_tuple(value.get("scopes", ()), "scopes"),
            claims=_json_object(value.get("claims", {}), "claims"),
        )


@dataclass(frozen=True, slots=True)
class SearchHitProvenance(_JsonContract):
    """Source/query provenance retained with a normalized hit."""

    source: SourceIdentity | None = None
    request_id: str | None = None
    retrieval_method: str | None = None
    details: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.request_id is not None:
            _string(self.request_id, "provenance.request_id")
        if self.retrieval_method is not None:
            _string(self.retrieval_method, "provenance.retrieval_method")
        object.__setattr__(self, "details", _json_object(self.details, "provenance.details"))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "source": self.source.to_dict() if self.source is not None else None,
            "request_id": self.request_id,
            "retrieval_method": self.retrieval_method,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SearchHitProvenance:
        value = _require_mapping(value, "provenance")
        _require_exact_keys(
            value,
            {"source", "request_id", "retrieval_method", "details"},
            "provenance",
        )
        source_value = value.get("source")
        return cls(
            source=(
                SourceIdentity.from_dict(_require_mapping(source_value, "provenance.source"))
                if source_value is not None
                else None
            ),
            request_id=_optional_string(value.get("request_id"), "provenance.request_id"),
            retrieval_method=_optional_string(
                value.get("retrieval_method"),
                "provenance.retrieval_method",
            ),
            details=_json_object(value.get("details", {}), "provenance.details"),
        )


@dataclass(frozen=True, slots=True)
class SearchHit(_JsonContract):
    """Transport-neutral normalized result backed by a Record identity."""

    source_kind: str
    source_id: str
    title: str
    snippet: str
    source_rank: int
    workspace_id: str | None = None
    rerank_text: str | None = None
    uri: str | None = None
    native_score: float | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    lifecycle: RecordStatus = RecordStatus.ACTIVE
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    provenance: SearchHitProvenance = field(default_factory=SearchHitProvenance)

    def __post_init__(self) -> None:
        identity = RecordIdentity(self.workspace_id, self.source_kind, self.source_id)
        object.__setattr__(self, "source_kind", identity.source_kind)
        object.__setattr__(self, "source_id", identity.source_id)
        _string(self.title, "title", allow_empty=True)
        if len(self.snippet) > MAX_SNIPPET_LENGTH:
            raise ValueError(f"snippet must be at most {MAX_SNIPPET_LENGTH} characters")
        _string(self.snippet, "snippet", allow_empty=True)
        _positive_int(self.source_rank, "source_rank")
        if self.rerank_text is not None:
            _string(self.rerank_text, "rerank_text", allow_empty=True)
            if len(self.rerank_text) > MAX_RERANK_TEXT_LENGTH:
                raise ValueError(
                    f"rerank_text must be at most {MAX_RERANK_TEXT_LENGTH} characters"
                )
        if self.uri is not None:
            _string(self.uri, "uri")
        if self.native_score is not None:
            object.__setattr__(
                self,
                "native_score",
                _finite_float(self.native_score, "native_score"),
            )
        object.__setattr__(self, "created_at", _optional_timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", _optional_timestamp(self.updated_at, "updated_at"))
        if not isinstance(self.lifecycle, RecordStatus):
            try:
                object.__setattr__(self, "lifecycle", RecordStatus(self.lifecycle))
            except ValueError as exc:
                raise ValueError("lifecycle must be a valid RecordStatus") from exc
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))
        if not isinstance(self.provenance, SearchHitProvenance):
            raise TypeError("provenance must be SearchHitProvenance")

    @property
    def identity(self) -> RecordIdentity:
        return RecordIdentity(self.workspace_id, self.source_kind, self.source_id)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "workspace_id": self.workspace_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "title": self.title,
            "snippet": self.snippet,
            "rerank_text": self.rerank_text,
            "uri": self.uri,
            "source_rank": self.source_rank,
            "native_score": self.native_score,
            "created_at": _timestamp_dict(self.created_at),
            "updated_at": _timestamp_dict(self.updated_at),
            "lifecycle": self.lifecycle.value,
            "metadata": dict(self.metadata),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SearchHit:
        value = _require_mapping(value, "hit")
        _require_exact_keys(
            value,
            {
                "workspace_id",
                "source_kind",
                "source_id",
                "title",
                "snippet",
                "rerank_text",
                "uri",
                "source_rank",
                "native_score",
                "created_at",
                "updated_at",
                "lifecycle",
                "metadata",
                "provenance",
            },
            "hit",
        )
        lifecycle = _record_status(
            value.get("lifecycle", RecordStatus.ACTIVE.value),
            "lifecycle",
        )
        return cls(
            workspace_id=_optional_string(value.get("workspace_id"), "workspace_id"),
            source_kind=_string(_required(value, "source_kind", "hit"), "source_kind"),
            source_id=_string(_required(value, "source_id", "hit"), "source_id"),
            title=_string(_required(value, "title", "hit"), "title", allow_empty=True),
            snippet=_string(_required(value, "snippet", "hit"), "snippet", allow_empty=True),
            rerank_text=_optional_string(value.get("rerank_text"), "rerank_text"),
            uri=_optional_string(value.get("uri"), "uri"),
            source_rank=_positive_int(_required(value, "source_rank", "hit"), "source_rank"),
            native_score=(
                _finite_float(value["native_score"], "native_score")
                if value.get("native_score") is not None
                else None
            ),
            created_at=_optional_timestamp(value.get("created_at"), "created_at"),
            updated_at=_optional_timestamp(value.get("updated_at"), "updated_at"),
            lifecycle=lifecycle,
            metadata=_json_object(value.get("metadata", {}), "metadata"),
            provenance=SearchHitProvenance.from_dict(
                _require_mapping(value.get("provenance", {}), "provenance")
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchDiagnostics(_JsonContract):
    """Structured candidate, failure, and stage timing data."""

    candidate_count: int | None = None
    candidate_counts: Mapping[str, int] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    stage_timings_ms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_count is not None and (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 0
        ):
            raise TypeError("candidate_count must be a non-negative integer")
        normalized_counts: dict[str, int] = {}
        for key, value in self.candidate_counts.items():
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise TypeError(
                    "candidate_counts must map stage names to non-negative integers"
                )
            normalized_counts[key] = value
        object.__setattr__(self, "candidate_counts", normalized_counts)
        object.__setattr__(
            self,
            "failures",
            _string_tuple(self.failures, "failures"),
        )
        normalized_timings: dict[str, float] = {}
        for key, value in self.stage_timings_ms.items():
            if not isinstance(key, str):
                raise TypeError("stage_timings_ms keys must be strings")
            normalized_timings[key] = _non_negative_float(value, "stage timing")
        object.__setattr__(self, "stage_timings_ms", normalized_timings)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "candidate_count": self.candidate_count,
            "candidate_counts": dict(self.candidate_counts),
            "failures": list(self.failures),
            "stage_timings_ms": dict(self.stage_timings_ms),
        }

    @classmethod
    def from_outcome(cls, outcome: object) -> SearchDiagnostics:
        """Adapt a local search outcome without importing its implementation."""
        failures: list[str] = []
        for failure in getattr(outcome, "failures", ()):
            stage = getattr(failure, "stage", "search")
            message = getattr(failure, "message", str(failure))
            failures.append(f"{stage}: {message}")
        return cls(
            candidate_count=getattr(outcome, "candidate_count", None),
            candidate_counts=getattr(outcome, "candidate_counts", {}),
            failures=tuple(dict.fromkeys(failures)),
            stage_timings_ms=getattr(outcome, "stage_timings_ms", {}),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SearchDiagnostics:
        value = _require_mapping(value, "diagnostics")
        _require_exact_keys(
            value,
            {"candidate_count", "candidate_counts", "failures", "stage_timings_ms"},
            "diagnostics",
        )
        candidate_counts = value.get("candidate_counts", {})
        stage_timings = value.get("stage_timings_ms", {})
        if not isinstance(candidate_counts, Mapping):
            raise TypeError("candidate_counts must be an object")
        if not isinstance(stage_timings, Mapping):
            raise TypeError("stage_timings_ms must be an object")
        return cls(
            candidate_count=value.get("candidate_count"),
            candidate_counts=dict(candidate_counts),
            failures=_string_tuple(value.get("failures", ()), "failures"),
            stage_timings_ms=dict(stage_timings),
        )


@dataclass(frozen=True, slots=True)
class SourceCapabilities(_JsonContract):
    """Features and bounds advertised by a federated source."""

    contract_versions: tuple[str, ...] = (FEDERATION_CONTRACT_VERSION,)
    supports_filters: bool = True
    supports_source_selection: bool = False
    supports_rerank_text: bool = False
    supports_partial_results: bool = True
    supports_cancellation: bool = True
    max_top_k: int = MAX_TOP_K
    max_rerank_text_length: int = MAX_RERANK_TEXT_LENGTH

    def __post_init__(self) -> None:
        versions = _string_tuple(self.contract_versions, "contract_versions")
        if FEDERATION_CONTRACT_VERSION not in versions:
            raise ValueError(
                f"contract_versions must include {FEDERATION_CONTRACT_VERSION}"
            )
        object.__setattr__(self, "contract_versions", versions)
        for name in (
            "supports_filters",
            "supports_source_selection",
            "supports_rerank_text",
            "supports_partial_results",
            "supports_cancellation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        object.__setattr__(self, "max_top_k", _positive_int(self.max_top_k, "max_top_k"))
        object.__setattr__(
            self,
            "max_rerank_text_length",
            _positive_int(self.max_rerank_text_length, "max_rerank_text_length"),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "contract_versions": list(self.contract_versions),
            "supports_filters": self.supports_filters,
            "supports_source_selection": self.supports_source_selection,
            "supports_rerank_text": self.supports_rerank_text,
            "supports_partial_results": self.supports_partial_results,
            "supports_cancellation": self.supports_cancellation,
            "max_top_k": self.max_top_k,
            "max_rerank_text_length": self.max_rerank_text_length,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceCapabilities:
        value = _require_mapping(value, "capabilities")
        _require_exact_keys(
            value,
            {
                "contract_versions",
                "supports_filters",
                "supports_source_selection",
                "supports_rerank_text",
                "supports_partial_results",
                "supports_cancellation",
                "max_top_k",
                "max_rerank_text_length",
            },
            "capabilities",
        )
        return cls(
            contract_versions=_string_tuple(
                _required(value, "contract_versions", "capabilities"),
                "contract_versions",
            ),
            supports_filters=value.get("supports_filters", True),
            supports_source_selection=value.get("supports_source_selection", False),
            supports_rerank_text=value.get("supports_rerank_text", False),
            supports_partial_results=value.get("supports_partial_results", True),
            supports_cancellation=value.get("supports_cancellation", True),
            max_top_k=value.get("max_top_k", MAX_TOP_K),
            max_rerank_text_length=value.get(
                "max_rerank_text_length",
                MAX_RERANK_TEXT_LENGTH,
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchRequest(_JsonContract):
    """Versioned request accepted by local and remote SearchSource ports."""

    query: str
    top_k: int = 10
    filters: Mapping[str, JsonValue] = field(default_factory=dict)
    source_selection: tuple[str, ...] = ()
    caller: CallerAuthorizationContext | None = None
    deadline_at: datetime | None = None
    cancellation_id: str | None = None
    request_id: str = ""
    trace_id: str = ""
    contract_version: str = FEDERATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _string(self.query, "query")
        if len(self.query) > MAX_QUERY_LENGTH:
            raise ValueError(f"query must be at most {MAX_QUERY_LENGTH} characters")
        object.__setattr__(self, "top_k", _positive_int(self.top_k, "top_k", maximum=MAX_TOP_K))
        object.__setattr__(self, "filters", _json_object(self.filters, "filters"))
        object.__setattr__(
            self,
            "source_selection",
            _string_tuple(self.source_selection, "source_selection"),
        )
        if self.caller is not None and not isinstance(self.caller, CallerAuthorizationContext):
            raise TypeError("caller must be CallerAuthorizationContext or None")
        object.__setattr__(self, "deadline_at", _optional_timestamp(self.deadline_at, "deadline_at"))
        if self.cancellation_id is not None:
            _string(self.cancellation_id, "cancellation_id")
        _string(self.request_id, "request_id", allow_empty=True)
        _string(self.trace_id, "trace_id", allow_empty=True)
        _string(self.contract_version, "contract_version")
        if self.contract_version != FEDERATION_CONTRACT_VERSION:
            raise ValueError(f"unsupported contract_version: {self.contract_version}")

    @property
    def authorization(self) -> CallerAuthorizationContext | None:
        return self.caller

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "filters": dict(self.filters),
            "source_selection": list(self.source_selection),
            "caller": self.caller.to_dict() if self.caller is not None else None,
            "deadline_at": _timestamp_dict(self.deadline_at),
            "cancellation_id": self.cancellation_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SearchRequest:
        value = _require_mapping(value, "request")
        _require_exact_keys(
            value,
            {
                "query",
                "top_k",
                "filters",
                "source_selection",
                "caller",
                "deadline_at",
                "cancellation_id",
                "request_id",
                "trace_id",
                "contract_version",
            },
            "request",
        )
        caller_value = value.get("caller")
        return cls(
            query=_string(_required(value, "query", "request"), "query"),
            top_k=value.get("top_k", 10),
            filters=_json_object(value.get("filters", {}), "filters"),
            source_selection=_string_tuple(
                value.get("source_selection", ()),
                "source_selection",
            ),
            caller=(
                CallerAuthorizationContext.from_dict(
                    _require_mapping(caller_value, "caller")
                )
                if caller_value is not None
                else None
            ),
            deadline_at=_optional_timestamp(value.get("deadline_at"), "deadline_at"),
            cancellation_id=_optional_string(value.get("cancellation_id"), "cancellation_id"),
            request_id=_string(value.get("request_id", ""), "request_id", allow_empty=True),
            trace_id=_string(value.get("trace_id", ""), "trace_id", allow_empty=True),
            contract_version=_string(
                value.get("contract_version", FEDERATION_CONTRACT_VERSION),
                "contract_version",
            ),
        )


@dataclass(frozen=True, slots=True)
class SearchResponse(_JsonContract):
    """Versioned response returned by a federated search source."""

    source: SourceIdentity
    hits: tuple[SearchHit, ...] = ()
    index_epoch: str | None = None
    elapsed_ms: float = 0.0
    partial: bool = False
    warnings: tuple[str, ...] = ()
    diagnostics: SearchDiagnostics | None = None
    capabilities: SourceCapabilities = field(default_factory=SourceCapabilities)
    contract_version: str = FEDERATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        if not isinstance(self.hits, Sequence) or isinstance(self.hits, (str, bytes)):
            raise TypeError("hits must be an array of SearchHit")
        hits = tuple(self.hits)
        if any(not isinstance(hit, SearchHit) for hit in hits):
            raise TypeError("hits must be an array of SearchHit")
        object.__setattr__(self, "hits", hits)
        if self.index_epoch is not None:
            _string(self.index_epoch, "index_epoch")
        object.__setattr__(self, "elapsed_ms", _non_negative_float(self.elapsed_ms, "elapsed_ms"))
        if not isinstance(self.partial, bool):
            raise TypeError("partial must be a boolean")
        warnings = _string_tuple(self.warnings, "warnings")
        object.__setattr__(self, "warnings", warnings)
        if self.diagnostics is not None and not isinstance(
            self.diagnostics,
            SearchDiagnostics,
        ):
            raise TypeError("diagnostics must be SearchDiagnostics or None")
        if not isinstance(self.capabilities, SourceCapabilities):
            raise TypeError("capabilities must be SourceCapabilities")
        _string(self.contract_version, "contract_version")
        if self.contract_version != FEDERATION_CONTRACT_VERSION:
            raise ValueError(f"unsupported contract_version: {self.contract_version}")

    @property
    def source_kind(self) -> str:
        return self.source.source_kind

    @property
    def source_id(self) -> str:
        return self.source.source_id

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "source": self.source.to_dict(),
            "contract_version": self.contract_version,
            "hits": [hit.to_dict() for hit in self.hits],
            "index_epoch": self.index_epoch,
            "elapsed_ms": self.elapsed_ms,
            "partial": self.partial,
            "warnings": list(self.warnings),
            "diagnostics": (
                self.diagnostics.to_dict() if self.diagnostics is not None else None
            ),
            "capabilities": self.capabilities.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SearchResponse:
        value = _require_mapping(value, "response")
        _require_exact_keys(
            value,
            {
                "source",
                "contract_version",
                "hits",
                "index_epoch",
                "elapsed_ms",
                "partial",
                "warnings",
                "diagnostics",
                "capabilities",
            },
            "response",
        )
        hits_value = value.get("hits", ())
        if not isinstance(hits_value, Sequence) or isinstance(hits_value, (str, bytes)):
            raise TypeError("hits must be an array")
        return cls(
            source=SourceIdentity.from_dict(
                _require_mapping(_required(value, "source", "response"), "source")
            ),
            contract_version=_string(
                value.get("contract_version", FEDERATION_CONTRACT_VERSION),
                "contract_version",
            ),
            hits=tuple(
                SearchHit.from_dict(_require_mapping(item, "hit"))
                for item in hits_value
            ),
            index_epoch=_optional_string(value.get("index_epoch"), "index_epoch"),
            elapsed_ms=value.get("elapsed_ms", 0.0),
            partial=value.get("partial", False),
            warnings=_string_tuple(value.get("warnings", ()), "warnings"),
            diagnostics=(
                SearchDiagnostics.from_dict(
                    _require_mapping(value["diagnostics"], "diagnostics")
                )
                if value.get("diagnostics") is not None
                else None
            ),
            capabilities=SourceCapabilities.from_dict(
                _require_mapping(value.get("capabilities", {}), "capabilities")
            ),
        )


@runtime_checkable
class SearchSource(Protocol):
    """Local port implemented by both in-process and remote source adapters."""

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a versioned search request."""
        ...

    def capabilities(self) -> SourceCapabilities:
        """Return the source's supported contract features."""
        ...


__all__ = [
    "FEDERATION_CONTRACT_VERSION",
    "MAX_QUERY_LENGTH",
    "MAX_RERANK_TEXT_LENGTH",
    "MAX_SNIPPET_LENGTH",
    "MAX_TOP_K",
    "CallerAuthorizationContext",
    "FederationEventKind",
    "JsonValue",
    "SearchDiagnostics",
    "SearchHit",
    "SearchHitProvenance",
    "SearchRequest",
    "SearchResponse",
    "SearchSource",
    "SourceCapabilities",
    "SourceIdentity",
]
