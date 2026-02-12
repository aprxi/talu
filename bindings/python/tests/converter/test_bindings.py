"""Tests for talu.converter._bindings module.

Tests for the FFI bindings helper functions without requiring actual conversion.
Focuses on parameter validation and error handling.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from talu.converter import (
    ALL_SCHEMES,
    GAF_SCHEMES,
    HARDWARE_SCHEMES,
    IMPLEMENTED_SCHEMES,
    MAX_OVERRIDES,
    SCHEME_NAME_TO_ENUM,
    ConvertOptions,
    ConvertResult,
    OverrideRule,
    Platform,
    ProgressAction,
    QuantLevel,
    Scheme,
    get_schemes_json,
    is_gaf_scheme,
    is_implemented,
    platform_to_enum,
    quant_to_enum,
    scheme_to_enum,
)
from talu.exceptions import ValidationError


class TestSchemeConstants:
    """Tests for Scheme class constants."""

    def test_gaf4_schemes_have_correct_values(self):
        """GAF4 schemes have expected enum values."""
        assert Scheme.GAF4_32 == 10
        assert Scheme.GAF4_64 == 11
        assert Scheme.GAF4_128 == 12

    def test_gaf8_schemes_have_correct_values(self):
        """GAF8 schemes have expected enum values."""
        assert Scheme.GAF8_32 == 13
        assert Scheme.GAF8_64 == 14
        assert Scheme.GAF8_128 == 15

    def test_hardware_schemes_have_correct_values(self):
        """Hardware float schemes have expected enum values."""
        assert Scheme.FP8_E4M3 == 20
        assert Scheme.FP8_E5M2 == 21
        assert Scheme.MXFP4 == 22
        assert Scheme.NVFP4 == 23


class TestSchemeNameMappings:
    """Tests for scheme name to enum mappings."""

    def test_all_schemes_contains_all_scheme_names(self):
        """ALL_SCHEMES contains all 10 scheme names."""
        assert len(ALL_SCHEMES) == 10
        assert "gaf4_64" in ALL_SCHEMES
        assert "fp8_e4m3" in ALL_SCHEMES

    def test_gaf_schemes_contains_only_gaf(self):
        """GAF_SCHEMES contains only GAF schemes."""
        assert len(GAF_SCHEMES) == 6
        for scheme in GAF_SCHEMES:
            assert scheme.startswith("gaf")

    def test_hardware_schemes_contains_only_hardware(self):
        """HARDWARE_SCHEMES contains only hardware schemes."""
        assert len(HARDWARE_SCHEMES) == 4
        expected = {"fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4"}
        assert HARDWARE_SCHEMES == expected

    def test_implemented_schemes_equals_gaf(self):
        """IMPLEMENTED_SCHEMES currently equals GAF_SCHEMES."""
        assert IMPLEMENTED_SCHEMES == GAF_SCHEMES

    def test_scheme_name_to_enum_maps_correctly(self):
        """SCHEME_NAME_TO_ENUM maps scheme names to enum values."""
        assert SCHEME_NAME_TO_ENUM["gaf4_64"] == Scheme.GAF4_64
        assert SCHEME_NAME_TO_ENUM["fp8_e4m3"] == Scheme.FP8_E4M3


class TestIsGafScheme:
    """Tests for is_gaf_scheme function."""

    def test_gaf_schemes_return_true(self):
        """GAF scheme names return True."""
        assert is_gaf_scheme("gaf4_32") is True
        assert is_gaf_scheme("gaf4_64") is True
        assert is_gaf_scheme("gaf8_128") is True

    def test_hardware_schemes_return_false(self):
        """Hardware scheme names return False."""
        assert is_gaf_scheme("fp8_e4m3") is False
        assert is_gaf_scheme("mxfp4") is False

    def test_case_insensitive(self):
        """is_gaf_scheme is case insensitive."""
        assert is_gaf_scheme("GAF4_64") is True
        assert is_gaf_scheme("Gaf8_32") is True

    def test_invalid_scheme_returns_false(self):
        """Invalid scheme names return False."""
        assert is_gaf_scheme("invalid") is False
        assert is_gaf_scheme("") is False


class TestIsImplemented:
    """Tests for is_implemented function."""

    def test_gaf_schemes_are_implemented(self):
        """GAF schemes return True (implemented)."""
        assert is_implemented("gaf4_64") is True
        assert is_implemented("gaf8_32") is True

    def test_hardware_schemes_not_implemented(self):
        """Hardware schemes return False (not implemented)."""
        assert is_implemented("fp8_e4m3") is False
        assert is_implemented("mxfp4") is False

    def test_case_insensitive(self):
        """is_implemented is case insensitive."""
        assert is_implemented("GAF4_64") is True
        assert is_implemented("FP8_E4M3") is False


class TestPlatformConstants:
    """Tests for Platform class constants."""

    def test_platform_values(self):
        """Platform enum values are correct."""
        assert Platform.CPU == 0
        assert Platform.METAL == 1
        assert Platform.CUDA == 2


class TestPlatformToEnum:
    """Tests for platform_to_enum function."""

    def test_valid_platform_names(self):
        """Valid platform names return correct enum values."""
        assert platform_to_enum("cpu") == Platform.CPU
        assert platform_to_enum("metal") == Platform.METAL
        assert platform_to_enum("cuda") == Platform.CUDA

    def test_platform_aliases(self):
        """Platform aliases work correctly."""
        assert platform_to_enum("mps") == Platform.METAL
        assert platform_to_enum("apple") == Platform.METAL
        assert platform_to_enum("gpu") == Platform.CUDA
        assert platform_to_enum("nvidia") == Platform.CUDA

    def test_case_insensitive(self):
        """platform_to_enum is case insensitive."""
        assert platform_to_enum("CPU") == Platform.CPU
        assert platform_to_enum("Metal") == Platform.METAL
        assert platform_to_enum("CUDA") == Platform.CUDA

    def test_invalid_platform_raises(self):
        """Invalid platform raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            platform_to_enum("invalid")

        assert "Unknown platform" in str(exc_info.value)
        assert exc_info.value.code == "INVALID_ARGUMENT"
        assert exc_info.value.details["param"] == "platform"

    def test_error_lists_valid_options(self):
        """Error message lists valid platform options."""
        with pytest.raises(ValidationError) as exc_info:
            platform_to_enum("xbox")

        error_msg = str(exc_info.value)
        assert "cpu" in error_msg or "metal" in error_msg


class TestQuantLevelConstants:
    """Tests for QuantLevel class constants."""

    def test_quant_level_values(self):
        """QuantLevel enum values are correct."""
        assert QuantLevel.Q4 == 0
        assert QuantLevel.Q8 == 1


class TestQuantToEnum:
    """Tests for quant_to_enum function."""

    def test_valid_quant_names(self):
        """Valid quant names return correct enum values."""
        assert quant_to_enum("4bit") == QuantLevel.Q4
        assert quant_to_enum("8bit") == QuantLevel.Q8

    def test_quant_aliases(self):
        """Quant aliases work correctly."""
        assert quant_to_enum("q4") == QuantLevel.Q4
        assert quant_to_enum("int4") == QuantLevel.Q4
        assert quant_to_enum("q8") == QuantLevel.Q8
        assert quant_to_enum("int8") == QuantLevel.Q8

    def test_case_insensitive(self):
        """quant_to_enum is case insensitive."""
        assert quant_to_enum("4BIT") == QuantLevel.Q4
        assert quant_to_enum("Q8") == QuantLevel.Q8

    def test_invalid_quant_raises(self):
        """Invalid quant level raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            quant_to_enum("16bit")

        assert "Unknown quant level" in str(exc_info.value)
        assert exc_info.value.code == "INVALID_ARGUMENT"
        assert exc_info.value.details["param"] == "quant"

    def test_error_lists_valid_options(self):
        """Error message lists valid quant options."""
        with pytest.raises(ValidationError) as exc_info:
            quant_to_enum("32bit")

        error_msg = str(exc_info.value)
        assert "4bit" in error_msg or "8bit" in error_msg


class TestProgressAction:
    """Tests for ProgressAction constants."""

    def test_progress_action_values(self):
        """ProgressAction enum values are correct."""
        assert ProgressAction.ADD == 0
        assert ProgressAction.UPDATE == 1
        assert ProgressAction.COMPLETE == 2


class TestMaxOverrides:
    """Tests for MAX_OVERRIDES constant."""

    def test_max_overrides_value(self):
        """MAX_OVERRIDES has expected value."""
        assert MAX_OVERRIDES == 32


class TestCtypesStructs:
    """Tests for ctypes struct definitions."""

    def test_override_rule_fields(self):
        """OverrideRule has expected fields."""
        import ctypes

        rule = OverrideRule()
        # Check fields exist
        assert hasattr(rule, "pattern")
        assert hasattr(rule, "scheme")
        # Check types
        fields = dict(OverrideRule._fields_)
        assert fields["pattern"] == ctypes.c_char_p
        assert fields["scheme"] == ctypes.c_uint32

    def test_convert_options_fields(self):
        """ConvertOptions has expected fields."""

        opts = ConvertOptions()
        # Check key fields exist
        assert hasattr(opts, "scheme")
        assert hasattr(opts, "force")
        assert hasattr(opts, "offline")
        assert hasattr(opts, "destination")
        assert hasattr(opts, "overrides")
        assert hasattr(opts, "num_overrides")
        assert hasattr(opts, "dry_run")
        assert hasattr(opts, "platform")
        assert hasattr(opts, "quant")
        assert hasattr(opts, "progress_callback")

    def test_convert_result_fields(self):
        """ConvertResult has expected fields."""
        import ctypes

        result = ConvertResult()
        # Check fields exist
        assert hasattr(result, "output_path")
        assert hasattr(result, "error_msg")
        assert hasattr(result, "success")
        # output_path and error_msg are c_void_p (not c_char_p)
        fields = dict(ConvertResult._fields_)
        assert fields["output_path"] == ctypes.c_void_p
        assert fields["error_msg"] == ctypes.c_void_p
        assert fields["success"] == ctypes.c_bool


class TestSchemeToEnumWithMocking:
    """Tests for scheme_to_enum function with mocked C library."""

    def test_scheme_to_enum_returns_code(self):
        """scheme_to_enum returns code from C library."""
        mock_lib = MagicMock()
        mock_lib.talu_convert_parse_scheme.return_value = 11  # GAF4_64

        with patch("talu.converter._get_convert_lib", return_value=mock_lib):
            result = scheme_to_enum("gaf4_64")

            assert result == 11
            mock_lib.talu_convert_parse_scheme.assert_called_once_with(b"gaf4_64")

    def test_scheme_to_enum_invalid_raises_validation_error(self):
        """scheme_to_enum raises ValidationError for invalid scheme."""
        mock_lib = MagicMock()
        mock_lib.talu_convert_parse_scheme.return_value = -1  # Invalid

        with patch("talu.converter._get_convert_lib", return_value=mock_lib):
            with patch(
                "talu.converter.get_schemes_json", return_value='{"gaf4_64": ["mlx"]}'
            ):
                with pytest.raises(ValidationError) as exc_info:
                    scheme_to_enum("invalid_scheme")

                assert "Unknown scheme" in str(exc_info.value)
                assert exc_info.value.code == "INVALID_ARGUMENT"

    def test_scheme_to_enum_error_fallback_on_json_failure(self):
        """scheme_to_enum falls back to ALL_SCHEMES on JSON decode error."""
        mock_lib = MagicMock()
        mock_lib.talu_convert_parse_scheme.return_value = -1

        with patch("talu.converter._get_convert_lib", return_value=mock_lib):
            with patch("talu.converter.get_schemes_json", return_value="invalid json"):
                with pytest.raises(ValidationError) as exc_info:
                    scheme_to_enum("invalid")

                # Should still have error message with schemes
                assert "Unknown scheme" in str(exc_info.value)


class TestGetSchemesJson:
    """Tests for get_schemes_json function."""

    def test_get_schemes_json_returns_string(self):
        """get_schemes_json returns a JSON string."""
        result = get_schemes_json()
        assert isinstance(result, str)
        # Should be valid JSON (might be empty {} on error)
        json.loads(result)

    def test_get_schemes_json_error_returns_empty(self):
        """get_schemes_json returns {} on C API error."""
        mock_lib = MagicMock()
        mock_lib.talu_convert_schemes.return_value = 1  # Error code

        with patch("talu.converter._get_convert_lib", return_value=mock_lib):
            result = get_schemes_json()

            assert result == "{}"
