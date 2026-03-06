"""
Root pytest configuration.

This file ensures the Python binding test package is importable no matter
where pytest is launched from.
"""

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent

# Ensure `tests.*` fixture plugins are importable when pytest is launched
# from the repository root instead of `bindings/python/`.
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
