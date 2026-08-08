"""Validate a labeled retrieval artifact and apply its stable baseline gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.evidence import compare_report, load_policy, validate_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    artifact: dict[str, Any] = json.loads(args.artifact.read_text())
    baseline: dict[str, Any] = json.loads(args.baseline.read_text())
    policy = load_policy(args.policy)
    failures = validate_report(artifact, policy)
    gate = compare_report(artifact, baseline, policy) if not failures else {"passed": False, "failures": [], "deltas": {}}
    gate["failures"] = failures + gate["failures"]
    gate["passed"] = not gate["failures"]
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
