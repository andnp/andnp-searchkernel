"""Verify that the source and distribution versions agree."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _runtime_version() -> str:
    module = ast.parse(
        (ROOT / "searchkernel" / "__init__.py").read_text(encoding="utf-8")
    )
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id == "__version__":
            if isinstance(statement.value, ast.Constant) and isinstance(
                statement.value.value, str
            ):
                return statement.value.value
            break
    raise RuntimeError("searchkernel.__version__ is not a string assignment")


def _distribution_version() -> str:
    metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    try:
        version = metadata["project"]["version"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("pyproject.toml does not define project.version") from error
    if not isinstance(version, str):
        raise TypeError("pyproject.toml project.version must be a string")
    return version


def main() -> None:
    runtime_version = _runtime_version()
    distribution_version = _distribution_version()
    if runtime_version != distribution_version:
        raise SystemExit(
            "Version mismatch: "
            f"searchkernel.__version__={runtime_version!r}, "
            f"project.version={distribution_version!r}"
        )
    print(f"Version consistency check passed: {runtime_version}")


if __name__ == "__main__":
    main()
