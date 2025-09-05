"""
Test runner for MT5 Trading Bot

Provides a convenient way to run all tests with different configurations
and generate comprehensive test reports.
"""

import pytest
import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_unit_tests():
    """Run unit tests only"""
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60)
    
    pytest_args = [
        "tests/test_smart_tp.py",
        "tests/test_risk_manager.py", 
        "tests/test_trade_execution.py",
        "tests/test_main_trading_logic.py",
        "-v",
        "-s",
        "--tb=short",
        "-m", "unit"
    ]
    
    return pytest.main(pytest_args)

def run_integration_tests():
    """Run integration tests only"""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS")
    print("="*60)
    
    pytest_args = [
        "tests/test_integration.py",
        "-v",
        "-s",
        "--tb=short",
        "-m", "integration"
    ]
    
    return pytest.main(pytest_args)

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    pytest_args = [
        "tests/",
        "-v",
        "-s",
        "--tb=short",
        "--durations=10"
    ]
    
    return pytest.main(pytest_args)

def run_tests_with_coverage():
    """Run tests with coverage report"""
    print("\n" + "="*60)
    print("RUNNING TESTS WITH COVERAGE")
    print("="*60)
    
    pytest_args = [
        "tests/",
        "-v",
        "-s",
        "--tb=short",
        "--cov=core",
        "--cov=adapters",
        "--cov-report=html",
        "--cov-report=term-missing"
    ]
    
    return pytest.main(pytest_args)

def run_specific_test(test_file):
    """Run a specific test file"""
    print(f"\n" + "="*60)
    print(f"RUNNING SPECIFIC TEST: {test_file}")
    print("="*60)
    
    pytest_args = [
        f"tests/{test_file}",
        "-v",
        "-s",
        "--tb=short"
    ]
    
    return pytest.main(pytest_args)

def run_fast_tests():
    """Run fast tests only (exclude slow tests)"""
    print("\n" + "="*60)
    print("RUNNING FAST TESTS")
    print("="*60)
    
    pytest_args = [
        "tests/",
        "-v",
        "-s",
        "--tb=short",
        "-m", "not slow"
    ]
    
    return pytest.main(pytest_args)

def main():
    """Main test runner"""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [unit|integration|all|coverage|fast|specific] [test_file]")
        print("\nOptions:")
        print("  unit       - Run unit tests only")
        print("  integration - Run integration tests only")
        print("  all        - Run all tests")
        print("  coverage   - Run tests with coverage report")
        print("  fast       - Run fast tests only (exclude slow tests)")
        print("  specific   - Run specific test file")
        print("\nExamples:")
        print("  python run_tests.py unit")
        print("  python run_tests.py integration")
        print("  python run_tests.py all")
        print("  python run_tests.py coverage")
        print("  python run_tests.py specific test_smart_tp.py")
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "unit":
        return run_unit_tests()
    elif command == "integration":
        return run_integration_tests()
    elif command == "all":
        return run_all_tests()
    elif command == "coverage":
        return run_tests_with_coverage()
    elif command == "fast":
        return run_fast_tests()
    elif command == "specific":
        if len(sys.argv) < 3:
            print("Error: Please specify test file for specific test run")
            return 1
        test_file = sys.argv[2]
        return run_specific_test(test_file)
    else:
        print(f"Error: Unknown command '{command}'")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
