import tomllib
from importlib.metadata import version as distribution_version
from pathlib import Path

import searchkernel
from searchkernel import RecordHit, RecordIdentity, SearchKernel


def test_public_version_matches_distribution_and_api_exports() -> None:
    assert searchkernel.__version__ == distribution_version("andnp-searchkernel")
    assert searchkernel.RecordHit is RecordHit
    assert searchkernel.RecordIdentity is RecordIdentity
    assert searchkernel.SearchKernel is SearchKernel


def test_semantic_release_version_variable_targets_runtime_declaration() -> None:
    metadata = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert metadata["tool"]["semantic_release"]["version_variables"] == [
        "searchkernel/__init__.py:__version__"
    ]
