#!/usr/bin/env python3
"""Require both SpotSound and Clotho reports to pass before authorizing a full run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova_safe import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
    names = [report["benchmark"] for report in reports]
    if len(reports) != 2 or len(set(names)) != 2:
        raise ValueError("Promotion requires exactly two distinct benchmark reports")
    passed = all(report["promotion_gate"]["passed"] for report in reports)
    decision = {
        "decision": "run_full" if passed else "rollback_and_audit",
        "passed": passed,
        "benchmarks": {
            report["benchmark"]: {
                "passed": report["promotion_gate"]["passed"],
                "delta_mIoU_points": report["delta_mIoU_points"],
                "new_catastrophic_regressions": report["new_catastrophic_regressions"],
            }
            for report in reports
        },
        "report_hashes": {str(path.resolve()): sha256_file(path) for path in args.report},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

