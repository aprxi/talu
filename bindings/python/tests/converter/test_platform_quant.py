"""
Tests for platform/quant automatic scheme resolution.

Tests the Platform, QuantLevel enums and scheme resolution logic.
"""

import pytest


class TestPlatformEnum:
    """Tests for Platform enum and platform_to_enum helper."""

    def test_platform_enum_values(self):
        """Platform enum has correct values."""
        from talu.converter import Platform

        assert Platform.CPU == 0
        assert Platform.METAL == 1
        assert Platform.CUDA == 2

    def test_platform_to_enum_cpu(self):
        """platform_to_enum maps CPU correctly."""
        from talu.converter import Platform, platform_to_enum

        assert platform_to_enum("cpu") == Platform.CPU
        assert platform_to_enum("CPU") == Platform.CPU

    def test_platform_to_enum_metal(self):
        """platform_to_enum maps Metal/MPS/Apple correctly."""
        from talu.converter import Platform, platform_to_enum

        assert platform_to_enum("metal") == Platform.METAL
        assert platform_to_enum("mps") == Platform.METAL
        assert platform_to_enum("apple") == Platform.METAL
        assert platform_to_enum("METAL") == Platform.METAL

    def test_platform_to_enum_cuda(self):
        """platform_to_enum maps CUDA/GPU/NVIDIA correctly."""
        from talu.converter import Platform, platform_to_enum

        assert platform_to_enum("cuda") == Platform.CUDA
        assert platform_to_enum("gpu") == Platform.CUDA
        assert platform_to_enum("nvidia") == Platform.CUDA

    def test_platform_to_enum_invalid_raises(self):
        """platform_to_enum raises for invalid platform."""
        from talu.converter import platform_to_enum
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Unknown platform"):
            platform_to_enum("invalid")


class TestQuantLevelEnum:
    """Tests for QuantLevel enum and quant_to_enum helper."""

    def test_quant_level_enum_values(self):
        """QuantLevel enum has correct values."""
        from talu.converter import QuantLevel

        assert QuantLevel.Q4 == 0
        assert QuantLevel.Q8 == 1

    def test_quant_to_enum_4bit(self):
        """quant_to_enum maps 4-bit variants correctly."""
        from talu.converter import QuantLevel, quant_to_enum

        assert quant_to_enum("4bit") == QuantLevel.Q4
        assert quant_to_enum("q4") == QuantLevel.Q4
        assert quant_to_enum("int4") == QuantLevel.Q4
        assert quant_to_enum("4BIT") == QuantLevel.Q4

    def test_quant_to_enum_8bit(self):
        """quant_to_enum maps 8-bit variants correctly."""
        from talu.converter import QuantLevel, quant_to_enum

        assert quant_to_enum("8bit") == QuantLevel.Q8
        assert quant_to_enum("q8") == QuantLevel.Q8
        assert quant_to_enum("int8") == QuantLevel.Q8

    def test_quant_to_enum_invalid_raises(self):
        """quant_to_enum raises for invalid quant level."""
        from talu.converter import quant_to_enum
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Unknown quant level"):
            quant_to_enum("invalid")


class TestConvertOptionsStruct:
    """Tests for ConvertOptions ctypes struct."""

    def test_convert_options_has_platform_field(self):
        """ConvertOptions has platform field."""
        from talu.converter import ConvertOptions

        opts = ConvertOptions()
        # Default is 0 (cpu)
        assert opts.platform == 0

    def test_convert_options_has_quant_field(self):
        """ConvertOptions has quant field."""
        from talu.converter import ConvertOptions

        opts = ConvertOptions()
        # Default is 0 (q4)
        assert opts.quant == 0

    def test_convert_options_has_use_platform_quant_field(self):
        """ConvertOptions has use_platform_quant field."""
        from talu.converter import ConvertOptions

        opts = ConvertOptions()
        # Default is False
        assert opts.use_platform_quant is False

    def test_convert_options_platform_quant_settable(self):
        """ConvertOptions platform/quant fields are settable."""
        from talu.converter import ConvertOptions, Platform, QuantLevel

        opts = ConvertOptions()
        opts.platform = Platform.METAL
        opts.quant = QuantLevel.Q8
        opts.use_platform_quant = True

        assert opts.platform == Platform.METAL
        assert opts.quant == QuantLevel.Q8
        assert opts.use_platform_quant is True


class TestConverterPlatformQuant:
    """Tests for Converter platform/quant parameters."""

    def test_convert_accepts_platform_parameter(self):
        """convert() accepts platform parameter."""
        import inspect

        from talu.converter import convert

        sig = inspect.signature(convert)
        assert "platform" in sig.parameters
        assert sig.parameters["platform"].default is None

    def test_convert_accepts_quant_parameter(self):
        """convert() accepts quant parameter."""
        import inspect

        from talu.converter import convert

        sig = inspect.signature(convert)
        assert "quant" in sig.parameters
        assert sig.parameters["quant"].default is None

    def test_convert_func_accepts_platform_parameter(self):
        """convert() function accepts platform parameter."""
        import inspect

        from talu.converter import convert

        sig = inspect.signature(convert)
        assert "platform" in sig.parameters
        assert sig.parameters["platform"].default is None

    def test_convert_func_accepts_quant_parameter(self):
        """convert() function accepts quant parameter."""
        import inspect

        from talu.converter import convert

        sig = inspect.signature(convert)
        assert "quant" in sig.parameters
        assert sig.parameters["quant"].default is None


class TestPlatformQuantValidation:
    """Tests for platform/quant parameter validation."""

    def test_platform_without_scheme_uses_resolution(self, convert_func, ConvertError):
        """Using platform without scheme triggers platform/quant resolution."""
        # Should raise ConvertError (model not found), not TypeError
        with pytest.raises(ConvertError):
            convert_func("nonexistent/model", platform="cpu")

    def test_platform_metal_without_scheme(self, convert_func, ConvertError):
        """Using platform=metal works."""
        with pytest.raises(ConvertError):
            convert_func("nonexistent/model", platform="metal")

    def test_platform_with_quant(self, convert_func, ConvertError):
        """Using platform with quant works."""
        with pytest.raises(ConvertError):
            convert_func("nonexistent/model", platform="cpu", quant="8bit")

    def test_invalid_platform_raises_validation_error(self, convert_func):
        """Invalid platform raises ValidationError."""
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Unknown platform"):
            convert_func("nonexistent/model", platform="invalid")

    def test_invalid_quant_raises_validation_error(self, convert_func):
        """Invalid quant raises ValidationError."""
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Unknown quant level"):
            convert_func("nonexistent/model", platform="cpu", quant="invalid")

    def test_scheme_overrides_platform(self, convert_func, ConvertError):
        """Explicit scheme takes precedence over platform/quant."""
        # When scheme is set, platform should be ignored
        with pytest.raises(ConvertError):
            convert_func("nonexistent/model", scheme="gaf4_32", platform="metal")
