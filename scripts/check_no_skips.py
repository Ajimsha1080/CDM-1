#!/usr/bin/env python3
"""Zero-skips verification gate: Parses pytest JUnit XML output and enforces
that zero tests were skipped. Exits 1 if any test was skipped, 0 otherwise.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def check_no_skips(xml_path: str) -> int:
    path = Path(xml_path)
    if not path.exists():
        print(f"[-] FATAL: JUnit XML test report not found at {xml_path}", file=sys.stderr)
        return 1

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        print(f"[-] FATAL: Failed to parse JUnit XML {xml_path}: {e}", file=sys.stderr)
        return 1

    skipped_tests = []
    total_tests = 0

    # Handle both <testsuites> root or single <testsuite> root
    suites = root.findall(".//testsuite") if root.tag == "testsuites" else [root]
    for suite in suites:
        total_tests += int(suite.attrib.get("tests", 0))

    for testcase in root.findall(".//testcase"):
        skipped_elem = testcase.find("skipped")
        if skipped_elem is not None:
            classname = testcase.attrib.get("classname", "")
            name = testcase.attrib.get("name", "")
            message = skipped_elem.attrib.get("message", "") or skipped_elem.text or "No message"
            skipped_tests.append((f"{classname}::{name}", message.strip()))

    if skipped_tests:
        print(f"\n[-] ZERO-SKIPS GATE FAILED: Found {len(skipped_tests)} skipped test(s):", file=sys.stderr)
        for name, msg in skipped_tests:
            print(f"    - {name}: {msg}", file=sys.stderr)
        print("\nAll tests in CI must either run and pass or be cleanly filtered via markers.", file=sys.stderr)
        print("Bare pytest.skip calls are prohibited in CI runs.\n", file=sys.stderr)
        return 1

    print(f"[+] ZERO-SKIPS GATE PASSED: 0 skipped tests out of {total_tests} executed tests in {path.name}.")
    return 0


def main():
    xml_file = sys.argv[1] if len(sys.argv) > 1 else "test-results.xml"
    sys.exit(check_no_skips(xml_file))


if __name__ == "__main__":
    main()
