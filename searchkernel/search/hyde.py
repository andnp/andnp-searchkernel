import logging
from pathlib import Path

from searchkernel.ports.live_indices import VectorIndexPort

logger = logging.getLogger(__name__)

type VectorSearchable = VectorIndexPort


def search_with_hypothesis(
    vector_index: VectorIndexPort,
    hypothesis: str,
    top_k: int = 10,
    excluded_files: set[str] | None = None,
    docs_root: Path | None = None,
):
    logger.info(f"HyDE search with hypothesis: {hypothesis[:100]}...")
    results = vector_index.search(hypothesis, top_k, excluded_files, docs_root)
    logger.info(f"HyDE search returned {len(results)} results")
    return results
