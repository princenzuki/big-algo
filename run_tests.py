#!/usr/bin/env python3
"""
Test Runner Script

Runs all tests including parity tests and unit tests.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error Output:")
            print(e.stderr)
        return False

def main():
    """Main test runner"""
    print("🧪 Lorentzian Trading Bot - Test Runner")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Test results
    results = []
    
    # Run parity tests
    print("\n🔍 Running Parity Tests...")
    success = run_command(
        "python -m pytest tests/parity_test.py -v",
        "Pine Script Parity Tests"
    )
    results.append(("Parity Tests", success))
    
    # Run unit tests
    print("\n🧪 Running Unit Tests...")
    success = run_command(
        "python -m pytest tests/unit_tests/ -v",
        "Unit Tests"
    )
    results.append(("Unit Tests", success))
    
    # Run linting
    print("\n🔍 Running Code Linting...")
    success = run_command(
        "python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics",
        "Code Linting"
    )
    results.append(("Linting", success))
    
    # Run type checking
    print("\n🔍 Running Type Checking...")
    success = run_command(
        "python -m mypy . --ignore-missing-imports",
        "Type Checking"
    )
    results.append(("Type Checking", success))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
