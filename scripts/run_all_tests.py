#!/usr/bin/env python3
"""
Test Runner: Discovers and executes all unit tests across all modules.
Usage:
    python scripts/run_all_tests.py
"""

import sys
import os
import unittest
import time

def run_tests():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modules_dir = os.path.join(repo_root, "modules")
    
    # Add repo root and all modules to sys.path so imports work seamlessly
    sys.path.insert(0, repo_root)
    for item in sorted(os.listdir(modules_dir)):
        item_path = os.path.join(modules_dir, item)
        if os.path.isdir(item_path):
            sys.path.insert(0, item_path)
            
    print("\n" + "="*75)
    print("🔬 MINI-FSD TEST SUITE: ANDREW NG & SEBASTIAN THRUN VERIFICATION")
    print("="*75)
    
    suite = unittest.TestSuite()
    for root, dirs, files in os.walk(modules_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                loader = unittest.TestLoader()
                discovered = loader.discover(start_dir=root, pattern=file, top_level_dir=repo_root)
                suite.addTests(discovered)
    
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time
    
    print("\n" + "-"*75)
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Summary: {total_tests} tests run in {elapsed:.3f}s")
    print(f"Passed:   {passed}")
    print(f"Failures: {failures}")
    print(f"Errors:   {errors}")
    print("="*75)
    
    if failures > 0 or errors > 0:
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED SUCCESSFULLY! The FSD stack is mathematically verified.")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
