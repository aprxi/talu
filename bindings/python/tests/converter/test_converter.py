"""Tests for talu.converter module.

Tests for the convert() function and module-level API.
Note: Full validation tests requiring model weights are in tests/reference/converter/.
"""

import pytest

from talu.converter import IMPLEMENTED_SCHEMES, SCHEME_INFO, convert, schemes
from talu.exceptions import ConvertError


class TestConvertFunction:
    """Tests for the convert() function."""

    def test_convert_invalid_model_raises(self):
        """convert() raises for invalid model paths."""
        with pytest.raises((ConvertError, RuntimeError)):
            convert("/definitely/not/a/real/path/to/model")

    def test_convert_invalid_scheme_raises(self):
        """convert() raises for invalid scheme."""
        with pytest.raises(ValueError):
            convert("some/model", scheme="invalid_scheme")

    def test_schemes_matches_implemented(self):
        """schemes() returns implemented schemes."""
        assert schemes() == sorted(IMPLEMENTED_SCHEMES)


class TestConvertError:
    """Tests for ConvertError exception."""

    def test_convert_error_is_exception(self):
        """ConvertError is an Exception."""
        assert issubclass(ConvertError, Exception)

    def test_convert_error_message(self):
        """ConvertError contains error message."""
        try:
            raise ConvertError("Test error message")
        except ConvertError as e:
            assert "Test error message" in str(e)


class TestUnimplementedSchemes:
    """Tests for unimplemented schemes that should raise ConvertError."""

    def test_fp8_e4m3_not_implemented(self):
        """fp8_e4m3 scheme raises ConvertError (not implemented)."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert("some/model", scheme="fp8_e4m3")

    def test_fp8_e5m2_not_implemented(self):
        """fp8_e5m2 scheme raises ConvertError (not implemented)."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert("some/model", scheme="fp8_e5m2")

    def test_mxfp4_not_implemented(self):
        """mxfp4 scheme raises ConvertError (not implemented)."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert("some/model", scheme="mxfp4")

    def test_nvfp4_not_implemented(self):
        """nvfp4 scheme raises ConvertError (not implemented)."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert("some/model", scheme="nvfp4")


class TestSchemes:
    """Tests for schemes() module function."""

    def test_schemes_returns_list(self):
        """schemes() returns a list."""
        result = schemes()
        assert isinstance(result, list)

    def test_schemes_contains_gaf_schemes(self):
        """schemes() contains grouped affine (GAF) schemes."""
        result = schemes()
        assert "gaf4_32" in result
        assert "gaf4_64" in result
        assert "gaf4_128" in result
        assert "gaf8_32" in result
        assert "gaf8_64" in result
        assert "gaf8_128" in result

    def test_schemes_is_sorted(self):
        """schemes() returns sorted list."""
        result = schemes()
        assert result == sorted(result)
