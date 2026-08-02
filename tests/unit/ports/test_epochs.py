import pytest

from searchkernel.ports import SearchEpochs


def test_epoch_snapshot_requires_valid_complete_lane_values() -> None:
    snapshot = SearchEpochs(keyword=1, vector=2, graph=3)

    assert snapshot.for_lane("vector") == 2

    with pytest.raises(ValueError, match="unknown search epoch lane"):
        snapshot.for_lane("missing")
    with pytest.raises(TypeError, match="vector epoch"):
        SearchEpochs(keyword=0, vector=True, graph=0)
    with pytest.raises(ValueError, match="graph epoch"):
        SearchEpochs(keyword=0, vector=0, graph=-1)
