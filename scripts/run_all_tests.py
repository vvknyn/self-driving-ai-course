#!/usr/bin/env python3
"""
Test Runner: executes each module's tests in a fresh subprocess.

Usage:
    python scripts/run_all_tests.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time


def run_tests() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modules_dir = os.path.join(repo_root, "modules")

    print("\n" + "=" * 75)
    print("Zero2FSD test suite — one subprocess per module")
    print("=" * 75)

    module_dirs: list[str] = []
    for item in sorted(os.listdir(modules_dir)):
        item_path = os.path.join(modules_dir, item)
        tests_path = os.path.join(item_path, "tests")
        if os.path.isdir(item_path) and os.path.isdir(tests_path):
            module_dirs.append(item_path)

    start_time = time.time()
    total_failures = 0
    total_errors = 0
    total_run = 0

    for module_path in module_dirs:
        module_name = os.path.basename(module_path)
        print(f"\n--- {module_name} ---")
        cmd = [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            os.path.join(module_path, "tests"),
            "-v",
        ]
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        combined = result.stdout + result.stderr
        print(combined, end="" if combined.endswith("\n") or not combined else "\n")

        if result.returncode != 0:
            total_failures += 1
        else:
            for line in combined.splitlines():
                # unittest writes "Ran N tests in ..." to stderr.
                if line.startswith("Ran ") and " test" in line:
                    try:
                        total_run += int(line.split()[1])
                    except (IndexError, ValueError):
                        pass

    elapsed = time.time() - start_time
    print("\n" + "-" * 75)
    print(f"Modules tested: {len(module_dirs)}")
    print(f"Approx tests (passed modules only): {total_run}")
    print(f"Module failures: {total_failures}")
    print(f"Elapsed: {elapsed:.3f}s")
    print("=" * 75)

    if total_failures > 0:
        print("Some module test suites failed.")
        return 1
    print("All module test suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
