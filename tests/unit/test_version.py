from importlib.metadata import version as distribution_version

import searchkernel
from searchkernel import SearchAPI, SearchKernel


def test_public_version_matches_distribution_and_api_exports() -> None:
    assert searchkernel.__version__ == distribution_version("andnp-searchkernel")
    assert searchkernel.SearchAPI is SearchAPI
    assert searchkernel.SearchKernel is SearchKernel
