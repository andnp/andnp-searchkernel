import pytest

from searchkernel.ports import SearchEpochProvider, SearchEpochs


def test_epoch_snapshot_requires_valid_complete_lane_values() -> None:
    snapshot = SearchEpochs.from_mapping(
        {"keyword": 1, "vector": 2, "graph": 3, "ignored": 4}
    )

    assert snapshot == SearchEpochs(keyword=1, vector=2, graph=3)
    assert snapshot.for_lane("vector") == 2

    with pytest.raises(ValueError, match="missing graph"):
        SearchEpochs.from_mapping({"keyword": 1, "vector": 2})
    with pytest.raises(TypeError, match="vector epoch"):
        SearchEpochs(keyword=0, vector=True, graph=0)
    with pytest.raises(ValueError, match="graph epoch"):
        SearchEpochs(keyword=0, vector=0, graph=-1)


def test_epoch_provider_protocol_accepts_legacy_mapping_adapters() -> None:
    class Adapter:
        def epochs(self) -> dict[str, int]:
            return {"keyword": 1, "vector": 2, "graph": 3}

    assert isinstance(Adapter(), SearchEpochProvider)
