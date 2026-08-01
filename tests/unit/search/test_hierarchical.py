from collections.abc import Iterable, Sequence
from typing import Any

import pytest

from searchkernel.domain import RecordIdentity, ScoredRef
from searchkernel.ports import SourceCapabilities
from searchkernel.runtime.federation import search_anything
from searchkernel.runtime.registry import SourceRegistry
from searchkernel.search.hierarchical import (
    HierarchicalRetrievalConfig,
    search_hierarchical,
)


class _HierarchicalSource:
    source_kind = "structured"
    capabilities = SourceCapabilities(supports_hierarchical_retrieval=True)

    async def search(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        return [ScoredRef("flat", 0.1, "structured")]

    async def search_parents(
        self, query: str, k: int, filters: dict[str, Any] | None = None
    ) -> Iterable[ScoredRef]:
        return [
            ScoredRef("parent-1", 1.0, "structured", {"text": "summary one"}),
            ScoredRef("parent-2", 0.9, "structured", {"text": "summary two"}),
        ]

    async def search_children(
        self,
        query: str,
        parent_ids: Sequence[str],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> Iterable[ScoredRef]:
        return [
            ScoredRef(
                "child-2",
                0.8,
                "structured",
                {"parent_id": "parent-2", "text": "fine two"},
            ),
            ScoredRef(
                "child-1",
                0.7,
                "structured",
                {"parent_id": "parent-1", "text": "fine one"},
            ),
        ]


@pytest.mark.asyncio
async def test_hierarchical_search_promotes_children_with_parent_provenance():
    source = _HierarchicalSource()

    results = await search_hierarchical(
        source,
        "query",
        k=2,
        config=HierarchicalRetrievalConfig(
            enabled=True,
            max_parents=2,
            children_per_parent=1,
        ),
    )

    assert [result.source_id for result in results] == ["child-1", "child-2"]
    assert results[0].storage_key != "record:[null,\"structured\",\"parent-1\"]"
    assert results[0].metadata["parent_id"] == "parent-1"
    assert results[0].metadata["hierarchical_provenance"] == {
        "parent_id": "parent-1",
        "parent_storage_key": "record:[null,\"structured\",\"parent-1\"]",
        "parent_score": 1.0,
        "kind": "fine_child",
    }


@pytest.mark.asyncio
async def test_hierarchical_search_groups_children_by_composite_parent_identity():
    parent_a = ScoredRef(
        "shared",
        1.0,
        "note",
        workspace_id="workspace-a",
    )
    parent_b = ScoredRef(
        "shared",
        0.9,
        "commit",
        workspace_id="workspace-b",
    )

    class _CollisionSource(_HierarchicalSource):
        def __init__(self) -> None:
            self.requested_parent_keys: list[str] = []

        async def search_parents(self, query, k, filters=None):
            return [parent_b, parent_a]

        async def search_children(self, query, parent_ids, k, filters=None):
            self.requested_parent_keys = list(parent_ids)
            return [
                ScoredRef(
                    "child-b",
                    0.8,
                    "commit",
                    {
                        "parent_storage_key": parent_b.storage_key,
                        "text": "commit child",
                    },
                    workspace_id="workspace-b",
                ),
                ScoredRef(
                    "child-a",
                    0.7,
                    "note",
                    {
                        "parent_identity": RecordIdentity(
                            "workspace-a", "note", "shared"
                        ),
                        "text": "note child",
                    },
                    workspace_id="workspace-a",
                ),
            ]

    source = _CollisionSource()
    results = await search_hierarchical(
        source,
        "query",
        k=2,
        config=HierarchicalRetrievalConfig(
            enabled=True,
            max_parents=2,
            children_per_parent=1,
        ),
    )

    assert source.requested_parent_keys == [
        parent_a.storage_key,
        parent_b.storage_key,
    ]
    assert [result.source_id for result in results] == ["child-a", "child-b"]
    assert [
        result.metadata["hierarchical_provenance"]["parent_storage_key"]
        for result in results
    ] == [parent_a.storage_key, parent_b.storage_key]


@pytest.mark.asyncio
async def test_hierarchical_search_adapts_legacy_parent_ids_when_unambiguous():
    class _LegacySource(_HierarchicalSource):
        async def search_children(self, query, parent_ids, k, filters=None):
            if parent_ids != ["parent-1", "parent-2"]:
                return []
            return [
                ScoredRef(
                    "child-1",
                    0.8,
                    "structured",
                    {"parent_id": "parent-1"},
                )
            ]

    results = await search_hierarchical(
        _LegacySource(),
        "query",
        k=1,
        config=HierarchicalRetrievalConfig(enabled=True),
    )

    assert [result.source_id for result in results] == ["child-1"]


@pytest.mark.asyncio
async def test_hierarchical_search_does_not_adapt_ambiguous_parent_ids():
    parent_a = ScoredRef("shared", 1.0, "note", workspace_id="workspace-a")
    parent_b = ScoredRef("shared", 0.9, "commit", workspace_id="workspace-b")

    class _LegacySource(_HierarchicalSource):
        async def search_parents(self, query, k, filters=None):
            return [parent_a, parent_b]

        async def search_children(self, query, parent_ids, k, filters=None):
            if parent_ids != ["shared", "shared"]:
                return []
            return [
                ScoredRef(
                    "child",
                    0.8,
                    "note",
                    {"parent_id": "shared"},
                )
            ]

    results = await search_hierarchical(
        _LegacySource(),
        "query",
        k=2,
        config=HierarchicalRetrievalConfig(enabled=True),
    )

    assert [result.source_id for result in results] == ["shared", "shared"]
    assert [
        result.metadata["hierarchical_provenance"]["parent_storage_key"]
        for result in results
    ] == [parent_a.storage_key, parent_b.storage_key]


@pytest.mark.asyncio
async def test_hierarchical_search_uses_parent_fallback_without_children():
    class _SummaryOnly(_HierarchicalSource):
        async def search_children(
            self, query, parent_ids: Sequence[str], k, filters=None
        ) -> Iterable[ScoredRef]:
            return []

    results = await search_hierarchical(
        _SummaryOnly(),
        "query",
        k=2,
        config=HierarchicalRetrievalConfig(enabled=True),
    )

    assert [result.source_id for result in results] == ["parent-1", "parent-2"]
    assert all(
        result.metadata["hierarchical_provenance"]["kind"] == "parent_fallback"
        for result in results
    )


@pytest.mark.asyncio
async def test_federation_hierarchical_mode_is_explicit_and_diagnosed():
    registry = SourceRegistry()
    registry.register(_HierarchicalSource())
    diagnostics = []

    results = await search_anything(
        "query",
        registry=registry,
        top_n=2,
        per_source_k=2,
        hierarchical_config=HierarchicalRetrievalConfig(enabled=True),
        diagnostics=diagnostics,
    )

    assert [result.source_id for result in results] == ["child-1", "child-2"]
    assert [item.message for item in diagnostics if item.stage == "hierarchical"] == [
        "hierarchical:applied"
    ]


@pytest.mark.asyncio
async def test_federation_falls_back_to_flat_search_without_capability():
    class _FlatOnly:
        source_kind = "flat"

        async def search(self, query, k, filters=None):
            return [ScoredRef("flat", 1.0, "flat")]

    registry = SourceRegistry()
    registry.register(_FlatOnly())
    diagnostics = []

    results = await search_anything(
        "query",
        registry=registry,
        hierarchical_config=HierarchicalRetrievalConfig(enabled=True),
        diagnostics=diagnostics,
    )

    assert [result.source_id for result in results] == ["flat"]
    assert diagnostics[0].message == "hierarchical:fallback:capability_unavailable:flat"


@pytest.mark.asyncio
async def test_disabled_hierarchical_mode_keeps_flat_retrieval():
    registry = SourceRegistry()
    registry.register(_HierarchicalSource())
    diagnostics = []

    results = await search_anything(
        "query",
        registry=registry,
        hierarchical_config=HierarchicalRetrievalConfig(enabled=False),
        diagnostics=diagnostics,
    )

    assert [result.source_id for result in results] == ["flat"]
    assert diagnostics[0].message == "hierarchical:disabled"
