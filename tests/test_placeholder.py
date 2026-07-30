"""Placeholder test for package import."""

import pytest


@pytest.mark.unit
def test_package_imports() -> None:
    """Verify searchkernel package can be imported."""
    import searchkernel

    assert searchkernel.__version__ == "0.1.0"
