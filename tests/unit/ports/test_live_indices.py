from typing import TYPE_CHECKING

from searchkernel.indices.graph import GraphStore
from searchkernel.indices.keyword import KeywordIndex
from searchkernel.indices.vector import VectorIndex
from searchkernel.ports import GraphIndexPort, KeywordIndexPort, VectorIndexPort

if TYPE_CHECKING:
    def _accept_vector(index: VectorIndexPort) -> VectorIndexPort:
        return index

    def _accept_keyword(index: KeywordIndexPort) -> KeywordIndexPort:
        return index

    def _accept_graph(index: GraphIndexPort) -> GraphIndexPort:
        return index

    def _legacy_vector(index: VectorIndex) -> VectorIndexPort:
        return _accept_vector(index)

    def _legacy_keyword(index: KeywordIndex) -> KeywordIndexPort:
        return _accept_keyword(index)

    def _legacy_graph(index: GraphStore) -> GraphIndexPort:
        return _accept_graph(index)


def test_legacy_indices_expose_live_port_surfaces() -> None:
    assert issubclass(VectorIndex, VectorIndexPort)
    assert issubclass(KeywordIndex, KeywordIndexPort)
    assert issubclass(GraphStore, GraphIndexPort)
