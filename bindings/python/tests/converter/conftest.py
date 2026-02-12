"""
Converter API test fixtures.

Provides fixtures for testing converter API without external dependencies.
"""

import shutil
import tempfile

import pytest


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for conversion output."""
    temp_dir = tempfile.mkdtemp(prefix="talu_convert_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def convert_func(talu):
    """Return the convert function for API testing."""
    return talu.convert


@pytest.fixture
def ConvertError():
    """Return the ConvertError exception class."""
    from talu.exceptions import ConvertError as CE

    return CE
