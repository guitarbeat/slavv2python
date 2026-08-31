"""Standalone Test Runner for MATLAB-to-Python Transpilation & Differential Audit E2E Test Suite.

Provides structured execution, per-tier breakdown (Tier 1..4), and test summary reporting.

Usage:
    python tests/e2e_audit/run_e2e_audit_tests.py
    python tests/e2e_audit/run_e2e_audit_tests.py --tier 1
    python tests/e2e_audit/run_e2e_audit_tests.py --json-output report.json
"""

import argparse
import json
import os
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests.e2e_audit.test_matlab2python_audit_e2e import (
    TestTier1FeatureCoverage,
    TestTier2BoundaryAndCornerCases,
    TestTier3CrossFeatureCombinations,
    TestTier4RealWorldScenarios,
)

TIER_CLASSES = {
    1: ("Tier 1: Feature Coverage", TestTier1FeatureCoverage),
    2: ("Tier 2: Boundary & Corner Cases", TestTier2BoundaryAndCornerCases),
    3: ("Tier 3: Cross-Feature Combinations", TestTier3CrossFeatureCombinations),
    4: ("Tier 4: Real-World Scenarios", TestTier4RealWorldScenarios),
}


class TierTestResult:
    """Stores test execution metrics for a specific tier."""

    def __init__(self, tier_num: int, name: str) -> None:
        self.tier_num = tier_num
        self.name = name
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.duration_seconds = 0.0
        self.test_details: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier_num,
            "name": self.name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 4),
            "status": "PASSED" if (self.failed == 0 and self.errors == 0) else "FAILED",
            "tests": self.test_details,
        }


def run_tier_tests(tier_num: int, tier_name: str, test_class: type) -> TierTestResult:
    """Run all test methods on a given tier test class."""
    result = TierTestResult(tier_num, tier_name)
    suite = unittest.TestSuite()

    # Discover test methods
    test_methods = [m for m in dir(test_class) if m.startswith("test_")]
    test_methods.sort()

    start_time = time.time()
    instance = test_class()

    for method_name in test_methods:
        result.total += 1
        test_start = time.time()
        test_func = getattr(instance, method_name)
        status = "PASSED"
        error_msg = None

        try:
            test_func()
            result.passed += 1
        except AssertionError as e:
            result.failed += 1
            status = "FAILED"
            error_msg = str(e)
        except Exception as e:
            result.errors += 1
            status = "ERROR"
            error_msg = "{}: {}".format(type(e).__name__, str(e))

        test_duration = time.time() - test_start
        result.test_details.append(
            {
                "method": method_name,
                "status": status,
                "duration": round(test_duration, 4),
                "error": error_msg,
            }
        )

    result.duration_seconds = time.time() - start_time
    return result


def print_summary_table(results: List[TierTestResult], total_duration: float) -> None:
    """Print clean formatted ASCII summary table."""
    print("\n" + "=" * 78)
    print("  E2E MATLAB-to-Python Differential Audit Test Suite Results")
    print("=" * 78)
    print(
        "{:<8} {:<38} {:>7} {:>7} {:>7} {:>8}".format(
            "Tier", "Description", "Total", "Pass", "Fail", "Status"
        )
    )
    print("-" * 78)

    grand_total = 0
    grand_passed = 0
    grand_failed = 0

    for r in results:
        status_str = "[PASS]" if (r.failed == 0 and r.errors == 0) else "[FAIL]"
        print(
            "{:<8} {:<38} {:>7} {:>7} {:>7} {:>8}".format(
                "Tier {}".format(r.tier_num),
                r.name.split(": ")[-1],
                r.total,
                r.passed,
                r.failed + r.errors,
                status_str,
            )
        )
        grand_total += r.total
        grand_passed += r.passed
        grand_failed += r.failed + r.errors

    print("-" * 78)
    overall_status = "ALL PASSED" if grand_failed == 0 else "FAILURES DETECTED"
    print(
        "{:<47} {:>7} {:>7} {:>7} {:>8}".format(
            "TOTAL (Duration: {:.2f}s)".format(total_duration),
            grand_total,
            grand_passed,
            grand_failed,
            "[{}]".format(overall_status),
        )
    )
    print("=" * 78 + "\n")

    # Print failure details if any
    for r in results:
        for t in r.test_details:
            if t["status"] != "PASSED":
                print("[!] {}::{} -> {}".format(r.name, t["method"], t["status"]))
                if t["error"]:
                    print("    Details: {}".format(t["error"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E2E differential audit tests.")
    parser.add_argument(
        "--tier", type=int, choices=[1, 2, 3, 4], help="Run tests for a specific tier only"
    )
    parser.add_argument("--json-output", type=str, help="Path to write JSON summary results")
    args = parser.parse_args()

    selected_tiers = [args.tier] if args.tier else [1, 2, 3, 4]
    results: List[TierTestResult] = []

    overall_start = time.time()
    for tier_num in selected_tiers:
        name, test_class = TIER_CLASSES[tier_num]
        print("[*] Running {}...".format(name))
        res = run_tier_tests(tier_num, name, test_class)
        results.append(res)

    total_duration = time.time() - overall_start
    print_summary_table(results, total_duration)

    if args.json_output:
        out_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_duration_seconds": round(total_duration, 4),
            "tiers": [r.to_dict() for r in results],
            "all_passed": all(r.failed == 0 and r.errors == 0 for r in results),
        }
        with open(args.json_output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print("[+] JSON summary written to {}".format(args.json_output))

    all_passed = all(r.failed == 0 and r.errors == 0 for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
