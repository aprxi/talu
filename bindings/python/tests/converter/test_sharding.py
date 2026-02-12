"""
Tests for converter sharding functionality.

Tests the max_shard_size parameter and _parse_size helper function.
"""

import pytest


class TestParseSizeHelper:
    """Tests for _parse_size helper function."""

    def test_parse_size_int_passthrough(self):
        """Integer values are returned unchanged."""
        from talu.converter import _parse_size

        assert _parse_size(1000) == 1000
        assert _parse_size(0) == 0
        assert _parse_size(5_000_000_000) == 5_000_000_000

    def test_parse_size_bytes(self):
        """Parse 'B' suffix correctly."""
        from talu.converter import _parse_size

        assert _parse_size("100B") == 100
        assert _parse_size("100 B") == 100
        assert _parse_size("100b") == 100  # case insensitive

    def test_parse_size_kilobytes(self):
        """Parse 'KB' suffix correctly."""
        from talu.converter import _parse_size

        assert _parse_size("1KB") == 1024
        assert _parse_size("1 KB") == 1024
        assert _parse_size("1kb") == 1024
        assert _parse_size("10KB") == 10 * 1024

    def test_parse_size_megabytes(self):
        """Parse 'MB' suffix correctly."""
        from talu.converter import _parse_size

        assert _parse_size("1MB") == 1024**2
        assert _parse_size("1 MB") == 1024**2
        assert _parse_size("500MB") == 500 * 1024**2
        assert _parse_size("500mb") == 500 * 1024**2

    def test_parse_size_gigabytes(self):
        """Parse 'GB' suffix correctly."""
        from talu.converter import _parse_size

        assert _parse_size("1GB") == 1024**3
        assert _parse_size("1 GB") == 1024**3
        assert _parse_size("5GB") == 5 * 1024**3
        assert _parse_size("5gb") == 5 * 1024**3

    def test_parse_size_terabytes(self):
        """Parse 'TB' suffix correctly."""
        from talu.converter import _parse_size

        assert _parse_size("1TB") == 1024**4
        assert _parse_size("1 TB") == 1024**4
        assert _parse_size("2TB") == 2 * 1024**4

    def test_parse_size_fractional(self):
        """Parse fractional sizes correctly."""
        from talu.converter import _parse_size

        assert _parse_size("0.5GB") == int(0.5 * 1024**3)
        assert _parse_size("1.5GB") == int(1.5 * 1024**3)
        assert _parse_size("2.5MB") == int(2.5 * 1024**2)

    def test_parse_size_plain_number_string(self):
        """Parse plain number strings (no unit) as bytes."""
        from talu.converter import _parse_size

        assert _parse_size("1000") == 1000
        assert _parse_size("5000000000") == 5_000_000_000

    def test_parse_size_whitespace_handling(self):
        """Handle whitespace correctly."""
        from talu.converter import _parse_size

        assert _parse_size("  5GB  ") == 5 * 1024**3
        assert _parse_size("5 GB") == 5 * 1024**3
        assert _parse_size("  1000  ") == 1000

    def test_parse_size_invalid_format_raises(self):
        """Invalid formats raise ValueError."""
        from talu.converter import _parse_size

        with pytest.raises(ValueError, match="Invalid size format"):
            _parse_size("invalid")

        with pytest.raises(ValueError, match="Invalid size format"):
            _parse_size("GB5")

        with pytest.raises(ValueError, match="Invalid size format"):
            _parse_size("fiveGB")

    def test_parse_size_empty_string_raises(self):
        """Empty string raises ValueError."""
        from talu.converter import _parse_size

        with pytest.raises(ValueError):
            _parse_size("")

    def test_parse_size_gb_not_matched_as_b(self):
        """'GB' suffix is not partially matched as 'B'.

        This was a regression where '5GB' would match 'B' first
        due to incorrect unit ordering.
        """
        from talu.converter import _parse_size

        # 5GB should be ~5 billion bytes, not 5
        result = _parse_size("5GB")
        assert result == 5 * 1024**3
        assert result > 1_000_000_000  # Must be > 1 billion

    def test_parse_size_mb_not_matched_as_b(self):
        """'MB' suffix is not partially matched as 'B'."""
        from talu.converter import _parse_size

        result = _parse_size("500MB")
        assert result == 500 * 1024**2
        assert result > 500_000_000  # Must be > 500 million

    def test_parse_size_kb_not_matched_as_b(self):
        """'KB' suffix is not partially matched as 'B'."""
        from talu.converter import _parse_size

        result = _parse_size("100KB")
        assert result == 100 * 1024
        assert result > 100_000  # Must be > 100,000


class TestMaxShardSizeParameter:
    """Tests for max_shard_size parameter in convert API."""

    def test_max_shard_size_parameter_exists_in_convert(self, talu):
        """convert() accepts max_shard_size parameter."""
        import inspect

        sig = inspect.signature(talu.convert)
        assert "max_shard_size" in sig.parameters
        # Default should be None
        assert sig.parameters["max_shard_size"].default is None

    def test_max_shard_size_parameter_exists_in_convert(self, talu):
        """convert() accepts max_shard_size parameter (duplicate check)."""
        import inspect

        sig = inspect.signature(talu.convert)
        assert "max_shard_size" in sig.parameters
        assert sig.parameters["max_shard_size"].default is None

    def test_max_shard_size_accepts_int(self, convert_func, ConvertError):
        """max_shard_size accepts integer (bytes)."""
        # Should raise ConvertError (model not found), not TypeError
        with pytest.raises(ConvertError):
            convert_func("nonexistent/model", max_shard_size=5_000_000_000)

    def test_max_shard_size_accepts_string(self, convert_func, ConvertError):
        """max_shard_size accepts string (human-readable)."""
        # Should raise ConvertError (model not found), not TypeError/ValueError
        with pytest.raises(ConvertError):
            convert_func("nonexistent/model", max_shard_size="5GB")

    def test_max_shard_size_accepts_none(self, convert_func, ConvertError):
        """max_shard_size accepts None (no sharding)."""
        # Should raise ConvertError (model not found), not TypeError
        with pytest.raises(ConvertError):
            convert_func("nonexistent/model", max_shard_size=None)

    def test_max_shard_size_invalid_string_raises_value_error(self, convert_func):
        """max_shard_size with invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid size format"):
            convert_func("nonexistent/model", max_shard_size="invalid")

    def test_max_shard_size_combines_with_scheme(self, convert_func, ConvertError):
        """max_shard_size can be combined with scheme parameter."""
        # Should raise ConvertError (model not found), not parameter errors
        with pytest.raises(ConvertError):
            convert_func(
                "nonexistent/model",
                scheme="gaf4_64",
                max_shard_size="5GB",
            )

    def test_max_shard_size_combines_with_output_dir(
        self, convert_func, ConvertError, temp_output_dir
    ):
        """max_shard_size can be combined with output_dir parameter."""
        # Should raise ConvertError (model not found), not parameter errors
        with pytest.raises(ConvertError):
            convert_func(
                "nonexistent/model",
                output_dir=temp_output_dir,
                max_shard_size="5GB",
            )


class TestMaxShardSizeCtypesStruct:
    """Tests that max_shard_size field is correctly defined in ctypes struct."""

    def test_max_shard_size_field_exists(self):
        """ConvertOptions has max_shard_size field."""
        from talu.converter import ConvertOptions

        field_names = [name for name, _ in ConvertOptions._fields_]
        assert "max_shard_size" in field_names

    def test_max_shard_size_field_type(self):
        """max_shard_size field is c_uint64."""
        import ctypes

        from talu.converter import ConvertOptions

        for name, field_type in ConvertOptions._fields_:
            if name == "max_shard_size":
                assert field_type == ctypes.c_uint64
                break
        else:
            pytest.fail("max_shard_size field not found")

    def test_max_shard_size_field_position(self):
        """max_shard_size is after num_overrides in ConvertOptions."""
        from talu.converter import ConvertOptions

        field_names = [name for name, _ in ConvertOptions._fields_]
        # max_shard_size comes right after num_overrides (second to last)
        assert "max_shard_size" in field_names
        max_shard_idx = field_names.index("max_shard_size")
        num_overrides_idx = field_names.index("num_overrides")
        assert max_shard_idx == num_overrides_idx + 1

    def test_max_shard_size_default_zero(self):
        """max_shard_size defaults to 0 (no sharding)."""
        from talu.converter import ConvertOptions

        opts = ConvertOptions()
        assert opts.max_shard_size == 0
