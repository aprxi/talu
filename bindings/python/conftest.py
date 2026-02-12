"""
Root pytest configuration.

This file registers shared fixture plugins at the package root level,
as required by pytest for pytest_plugins declarations.
"""

# Register fixture plugins that need to be shared across test directories.
# This is required to be at the pytest rootdir level.
pytest_plugins = ["tests.tokenizer.conftest"]
