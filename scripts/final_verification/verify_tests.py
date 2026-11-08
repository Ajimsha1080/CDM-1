#!/usr/bin/env python3
"""CDM-1 Final Verification: Complete Automated Pytest Suite.

Executes the full automated test suite in a clean subprocess and asserts:
- Zero test failures
- Zero test errors
- Zero test skips
- Collects test execution metrics and generates test_report.json
"""

import os
import sys
import re
import json
import time
import subprocess
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def verify_tests(report_path: str = "test_report.json") -> Dict[str, Any]:
    print("=" * 70)
    print("CDM-1 VERIFICATION: AUTOMATED TEST SUITE EXECUTION")
    print("=" * 70)

    report: Dict[str, Any] = {
        "timestamp": time.time(),
        "command": f"{sys.executable} -m pytest tests/ -v",
        "passed": False,
        "total_collected": 0,
        "total_passed": 0,
        "total_failed": 0,
        "total_errors": 0,
        "total_skipped": 0,
        "duration_seconds": 0.0,
        "test_files": [],
        "output_summary": "",
    }

    print(f"[*] Running full test suite: {report['command']}")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
    )
    duration = round(time.time() - t0, 2)
    output = proc.stdout
    report["duration_seconds"] = duration
    print(f"[*] Pytest run finished in {duration}s with exit code {proc.returncode}")

    # Parse collected items
    collected_match = re.search(r"collected (\d+) items", output)
    if collected_match:
        report["total_collected"] = int(collected_match.group(1))

    # Parse short test summary info or result line
    # Examples:
    # "=== 116 passed in 180.20s ==="
    # "=== 1 failed, 115 passed in 200.43s ==="
    passed_match = re.search(r"(\d+) passed", output)
    if passed_match:
        report["total_passed"] = int(passed_match.group(1))

    failed_match = re.search(r"(\d+) failed", output)
    if failed_match:
        report["total_failed"] = int(failed_match.group(1))

    errors_match = re.search(r"(\d+) error", output)
    if errors_match:
        report["total_errors"] = int(errors_match.group(1))

    skipped_match = re.search(r"(\d+) skipped", output)
    if skipped_match:
        report["total_skipped"] = int(skipped_match.group(1))

    # Parse individual test files passed
    test_lines = [line.strip() for line in output.splitlines() if "PASSED" in line or "FAILED" in line or "ERROR" in line]
    report["test_lines_count"] = len(test_lines)

    summary_lines = [line.strip() for line in output.splitlines() if line.startswith("=") and ("passed" in line or "failed" in line)]
    report["output_summary"] = summary_lines[-1] if summary_lines else output.splitlines()[-1]

    # Acceptance criteria: exit code 0, 0 failed, 0 errors, 0 skipped, passed >= 115
    passed = (
        proc.returncode == 0
        and report["total_failed"] == 0
        and report["total_errors"] == 0
        and report["total_skipped"] == 0
        and report["total_passed"] >= 115
    )
    report["passed"] = passed

    print(f"[*] Summary: {report['output_summary']}")
    print(f"[*] Total Collected: {report['total_collected']}")
    print(f"[*] Total Passed:    {report['total_passed']}")
    print(f"[*] Total Failed:    {report['total_failed']}")
    print(f"[*] Total Errors:    {report['total_errors']}")
    print(f"[*] Total Skipped:   {report['total_skipped']}")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Pytest verification complete: {'ALL PASS' if passed else 'FAILURES DETECTED'}")
    print(f"[+] Report written to {report_path}")
    return report


if __name__ == "__main__":
    rep = verify_tests()
    sys.exit(0 if rep.get("passed") else 1)
