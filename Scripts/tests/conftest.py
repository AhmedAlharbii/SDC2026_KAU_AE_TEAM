"""
pytest configuration for SDC2026 KAU AE Team tests.

This file ensures pytest can find and import Scripts modules correctly.
"""

import sys
import os

# Add Scripts directory to path so all test files can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
