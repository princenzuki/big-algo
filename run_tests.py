#!/usr/bin/env python3
"""
Quick test runner for MT5 Trading Bot

This script provides easy access to run tests from the project root.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the test runner
from tests.run_tests import main

if __name__ == "__main__":
    sys.exit(main())