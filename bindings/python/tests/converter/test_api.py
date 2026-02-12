"""
Tests for convert API behavior.

Tests the Python API contract, parameter validation, and error handling.
These tests focus on the API surface, not the actual conversion logic.
"""

import pytest

from talu.exceptions import ConvertError


class TestConvertApiSignature:
    """Tests for convert() function signature and defaults."""

    def test_convert_exists(self, talu):
        """convert function is exported from talu."""
        assert hasattr(talu, "convert")
        assert callable(talu.convert)

    def test_convert_error_exists(self):
        """ConvertError exception is accessible from talu.converter."""
        assert issubclass(ConvertError, Exception)

    def test_default_scheme_is_gaf4_64(self, talu):
        """Default scheme should be 'gaf4_64' (best for local use)."""
        import inspect

        # Check that scheme parameter exists with None default
        # (defaults to gaf4_64 at runtime when neither scheme nor platform is set)
        sig = inspect.signature(talu.convert)
        scheme_param = sig.parameters.get("scheme")
        assert scheme_param is not None
        assert scheme_param.default is None  # None allows platform/quant mode

        # convert() is the primary API (no Converter class)


class TestConvertParameterValidation:
    """Tests for parameter validation."""

    def test_invalid_scheme_raises_value_error(self, convert_func):
        """Invalid scheme raises ValueError with valid options listed."""
        with pytest.raises(ValueError, match="Unknown scheme") as exc_info:
            convert_func("dummy/model", scheme="invalid")

        # Error message should list valid schemes or aliases
        error_msg = str(exc_info.value)
        assert "gaf4_64" in error_msg or "Available" in error_msg or "Valid options" in error_msg, (
            f"Error should list valid schemes, got: {error_msg}"
        )

    def test_unimplemented_scheme_raises_convert_error(self, convert_func, ConvertError):
        """Unimplemented scheme (fp8, mxfp4, nvfp4) raises ConvertError."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert_func("dummy/model", scheme="fp8_e4m3")

    def test_valid_gaf_schemes(self, talu):
        """All valid GAF schemes are accepted."""
        valid_schemes = ["gaf4_32", "gaf4_64", "gaf4_128", "gaf8_32", "gaf8_64", "gaf8_128"]
        available = talu.converter.list_schemes(include_unimplemented=True)
        for scheme in valid_schemes:
            assert scheme in available, f"{scheme} not in available types"


class TestAllSchemeStringValidation:
    """Tests that all 10 scheme strings are properly recognized.

    We have:
    - 6 GAF schemes: gaf4_32, gaf4_64, gaf4_128, gaf8_32, gaf8_64, gaf8_128
    - 4 hardware schemes (not implemented): fp8_e4m3, fp8_e5m2, mxfp4, nvfp4
    """

    # All 6 implemented schemes
    IMPLEMENTED_GAF_SCHEMES = ["gaf4_32", "gaf4_64", "gaf4_128", "gaf8_32", "gaf8_64", "gaf8_128"]

    # All 4 unimplemented schemes
    UNIMPLEMENTED_SCHEMES = ["fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4"]

    @pytest.mark.parametrize("scheme", IMPLEMENTED_GAF_SCHEMES)
    def test_gaf_scheme_is_recognized(self, talu, scheme):
        """Each GAF scheme string is recognized as valid."""
        schemes = talu.converter.list_schemes(include_unimplemented=True)
        assert scheme in schemes, f"GAF scheme '{scheme}' not recognized"
        assert schemes[scheme]["category"] == "gaf"
        assert schemes[scheme]["status"] == "stable"

    @pytest.mark.parametrize("scheme", UNIMPLEMENTED_SCHEMES)
    def test_unimplemented_scheme_is_recognized(self, talu, scheme):
        """Each unimplemented scheme string is recognized (but marked unimplemented)."""
        schemes = talu.converter.list_schemes(include_unimplemented=True)
        assert scheme in schemes, f"Unimplemented scheme '{scheme}' not recognized"
        assert schemes[scheme]["status"] == "not implemented"

    @pytest.mark.parametrize("scheme", IMPLEMENTED_GAF_SCHEMES)
    def test_gaf_scheme_does_not_raise_value_error(self, convert_func, ConvertError, scheme):
        """GAF schemes don't raise ValueError (may raise ConvertError for missing model)."""
        # Should raise ConvertError (model not found), NOT ValueError (invalid scheme)
        with pytest.raises(ConvertError):
            convert_func("nonexistent/model/path", scheme=scheme)

    @pytest.mark.parametrize("scheme", UNIMPLEMENTED_SCHEMES)
    def test_unimplemented_scheme_raises_convert_error(self, convert_func, ConvertError, scheme):
        """Unimplemented schemes raise ConvertError with 'not yet implemented' message."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert_func("dummy/model", scheme=scheme)

    def test_total_scheme_count(self, talu):
        """There are exactly 10 schemes total (6 implemented + 4 unimplemented)."""
        all_schemes = talu.converter.list_schemes(include_unimplemented=True)
        assert len(all_schemes) == 10, f"Expected 10 schemes, got {len(all_schemes)}"

    def test_implemented_scheme_count(self, talu):
        """There are exactly 6 implemented schemes."""
        implemented = talu.converter.list_schemes(include_unimplemented=False)
        assert len(implemented) == 6, f"Expected 6 implemented schemes, got {len(implemented)}"

    def test_gaf_scheme_count(self, talu):
        """There are exactly 6 GAF schemes."""
        gaf = talu.converter.list_schemes(category="gaf", include_unimplemented=True)
        assert len(gaf) == 6, f"Expected 6 GAF schemes, got {len(gaf)}"

    def test_hardware_scheme_count(self, talu):
        """There are exactly 4 hardware schemes."""
        hardware = talu.converter.list_schemes(category="hardware", include_unimplemented=True)
        assert len(hardware) == 4, f"Expected 4 hardware schemes, got {len(hardware)}"


class TestSchemeMetadata:
    """Tests for scheme metadata correctness."""

    def test_gaf_schemes_have_correct_bits(self, talu):
        """GAF schemes have correct bit values."""
        schemes = talu.converter.list_schemes(category="gaf")
        expected_bits = {
            "gaf4_32": 4,
            "gaf4_64": 4,
            "gaf4_128": 4,
            "gaf8_32": 8,
            "gaf8_64": 8,
            "gaf8_128": 8,
        }
        for scheme, bits in expected_bits.items():
            assert schemes[scheme]["bits"] == bits, f"{scheme} should have bits={bits}"

    def test_gaf_schemes_have_correct_group_size(self, talu):
        """GAF schemes have correct group_size values."""
        schemes = talu.converter.list_schemes(category="gaf")
        expected_group_size = {
            "gaf4_32": 32,
            "gaf4_64": 64,
            "gaf4_128": 128,
            "gaf8_32": 32,
            "gaf8_64": 64,
            "gaf8_128": 128,
        }
        for scheme, group_size in expected_group_size.items():
            assert schemes[scheme]["group_size"] == group_size, (
                f"{scheme} should have group_size={group_size}"
            )

    def test_gaf_schemes_are_mlx_compatible(self, talu):
        """All GAF schemes are marked as MLX compatible."""
        schemes = talu.converter.list_schemes(category="gaf")
        for scheme_name, scheme_info in schemes.items():
            assert scheme_info.get("mlx_compatible") is True, (
                f"{scheme_name} should be mlx_compatible"
            )


class TestConvertErrorHandling:
    """Tests for error handling."""

    def test_nonexistent_model_raises_convert_error(
        self, convert_func, temp_output_dir, ConvertError
    ):
        """Nonexistent model raises ConvertError."""
        with pytest.raises(ConvertError):
            convert_func(
                "nonexistent/model/path/that/does/not/exist",
                output_dir=temp_output_dir,
            )

    def test_invalid_hf_model_raises_convert_error(
        self, convert_func, temp_output_dir, ConvertError
    ):
        """Invalid HuggingFace model ID raises ConvertError."""
        with pytest.raises(ConvertError):
            convert_func(
                "this-org-does-not-exist/this-model-does-not-exist-12345",
                output_dir=temp_output_dir,
            )

class TestConvertErrorHierarchy:
    """Tests for ConvertError exception."""

    def test_convert_error_is_exception(self, ConvertError):
        """ConvertError is an Exception."""
        assert issubclass(ConvertError, Exception)

    def test_convert_error_can_be_raised(self, ConvertError):
        """ConvertError can be raised and caught."""
        with pytest.raises(ConvertError):
            raise ConvertError("Test error")

    def test_convert_error_message(self, ConvertError):
        """ConvertError preserves message."""
        err = ConvertError("Custom message")
        assert str(err) == "Custom message"


class TestUnimplementedSchemes:
    """Tests for unimplemented schemes that should raise ConvertError."""

    def test_fp8_e4m3_not_implemented(self, convert_func, ConvertError):
        """fp8_e4m3 scheme raises ConvertError (not implemented)."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert_func("some/model", scheme="fp8_e4m3")

    def test_fp8_e5m2_not_implemented(self, convert_func, ConvertError):
        """fp8_e5m2 scheme raises ConvertError (not implemented)."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert_func("some/model", scheme="fp8_e5m2")

    def test_mxfp4_not_implemented(self, convert_func, ConvertError):
        """mxfp4 scheme raises ConvertError (not implemented)."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert_func("some/model", scheme="mxfp4")

    def test_nvfp4_not_implemented(self, convert_func, ConvertError):
        """nvfp4 scheme raises ConvertError (not implemented)."""
        with pytest.raises(ConvertError, match="not yet implemented"):
            convert_func("some/model", scheme="nvfp4")


class TestApiParameters:
    """Tests for API parameters."""

    def test_scheme_parameter_exists(self, talu):
        """convert() accepts scheme parameter."""
        import inspect

        sig = inspect.signature(talu.convert)
        assert "scheme" in sig.parameters
        # Default is None to allow platform/quant mode; gaf4_64 is used at runtime
        assert sig.parameters["scheme"].default is None

    def test_destination_parameter_exists(self, talu):
        """convert() accepts destination parameter."""
        import inspect

        sig = inspect.signature(talu.convert)
        assert "destination" in sig.parameters
        # Default should be None
        assert sig.parameters["destination"].default is None

    def test_destination_parameter_in_convert(self, talu):
        """convert() accepts destination parameter (duplicate check)."""
        import inspect

        sig = inspect.signature(talu.convert)
        assert "destination" in sig.parameters
        assert sig.parameters["destination"].default is None


class TestListSchemes:
    """Tests for list_schemes() discovery API."""

    def test_list_schemes_exists(self, talu):
        """list_schemes function is accessible via talu.converter."""
        assert callable(talu.converter.list_schemes)

    def test_list_schemes_returns_dict(self, talu):
        """list_schemes returns a dictionary."""
        schemes = talu.converter.list_schemes()
        assert isinstance(schemes, dict)

    def test_list_schemes_contains_implemented_schemes(self, talu):
        """list_schemes includes gaf schemes by default."""
        schemes = talu.converter.list_schemes()
        # GAF schemes
        assert "gaf4_64" in schemes
        assert "gaf4_32" in schemes
        assert "gaf8_64" in schemes

    def test_list_schemes_excludes_unimplemented_by_default(self, talu):
        """list_schemes excludes unimplemented schemes by default."""
        schemes = talu.converter.list_schemes()
        assert "fp8_e4m3" not in schemes
        assert "mxfp4" not in schemes
        assert "nvfp4" not in schemes

    def test_list_schemes_includes_unimplemented_when_requested(self, talu):
        """list_schemes includes unimplemented schemes when requested."""
        schemes = talu.converter.list_schemes(include_unimplemented=True)
        assert "fp8_e4m3" in schemes
        assert "mxfp4" in schemes
        assert "nvfp4" in schemes

    def test_list_schemes_filter_by_category(self, talu):
        """list_schemes can filter by category."""
        gaf = talu.converter.list_schemes(category="gaf")
        assert "gaf4_64" in gaf
        assert "gaf4_32" in gaf

    def test_list_schemes_has_required_fields(self, talu):
        """Each scheme has category, bits, description, quality, and status."""
        schemes = talu.converter.list_schemes()

        for scheme_name, scheme_info in schemes.items():
            assert "category" in scheme_info, f"{scheme_name} missing category"
            assert "bits" in scheme_info, f"{scheme_name} missing bits"
            assert "description" in scheme_info, f"{scheme_name} missing description"
            assert "quality" in scheme_info, f"{scheme_name} missing quality"
            assert "status" in scheme_info, f"{scheme_name} missing status"

    def test_list_schemes_gaf_has_group_size(self, talu):
        """GAF schemes include group_size."""
        schemes = talu.converter.list_schemes(category="gaf")

        for scheme_name, scheme_info in schemes.items():
            assert "group_size" in scheme_info, f"{scheme_name} missing group_size"


class TestOverridesParameter:
    """Tests for per-tensor quantization overrides (not supported for GAF schemes)."""

    def test_overrides_parameter_exists_in_convert(self, talu):
        """convert() accepts overrides parameter."""
        import inspect

        sig = inspect.signature(talu.convert)
        assert "overrides" in sig.parameters
        # Default should be None
        assert sig.parameters["overrides"].default is None

    def test_overrides_parameter_exists_in_convert(self, talu):
        """convert() accepts overrides parameter (duplicate check)."""
        import inspect

        sig = inspect.signature(talu.convert)
        assert "overrides" in sig.parameters
        assert sig.parameters["overrides"].default is None

    def test_overrides_not_supported_for_gaf_schemes(self, convert_func):
        """Overrides raise ValueError for GAF schemes."""
        with pytest.raises(ValueError, match="not supported"):
            convert_func(
                "dummy/model",
                scheme="gaf4_64",
                overrides={"model.layers.*": "gaf4_64"},
            )


class TestVerifyModel:
    """Tests for verify() function and verify parameter."""

    def test_verify_function_exists(self, talu):
        """verify function is accessible via talu.converter."""
        assert callable(talu.converter.verify)

    def test_verify_parameter_exists_in_convert(self, talu):
        """convert() accepts verify parameter."""
        import inspect

        sig = inspect.signature(talu.convert)
        assert "verify" in sig.parameters
        # Default should be False
        assert sig.parameters["verify"].default is False

    def test_verify_model_returns_verification_result(self, talu):
        """verify() returns a VerificationResult."""
        from talu.converter import VerificationResult

        # Use a nonexistent path to get a failed result
        result = talu.converter.verify("/nonexistent/model/path")

        assert isinstance(result, VerificationResult)
        assert hasattr(result, "success")
        assert hasattr(result, "model_path")
        assert hasattr(result, "output")
        assert hasattr(result, "tokens_generated")
        assert hasattr(result, "error")

    def test_verify_model_fails_for_nonexistent_path(self, talu):
        """verify() returns failure for nonexistent model."""
        result = talu.converter.verify("/nonexistent/model/path")

        assert not result.success
        assert result.error is not None
        assert result.model_path == "/nonexistent/model/path"

    def test_verification_result_bool(self, talu):
        """VerificationResult is truthy when successful, falsy when failed."""
        from talu.converter import VerificationResult

        success = VerificationResult(
            success=True,
            model_path="/test",
            output="test output",
            tokens_generated=5,
        )
        assert bool(success) is True

        failure = VerificationResult(
            success=False,
            model_path="/test",
            error="test error",
        )
        assert bool(failure) is False

    def test_verification_result_repr(self, talu):
        """VerificationResult has useful repr."""
        from talu.converter import VerificationResult

        success = VerificationResult(
            success=True,
            model_path="/test",
            output="test",
            tokens_generated=5,
        )
        assert "success=True" in repr(success)
        assert "tokens=5" in repr(success)

        failure = VerificationResult(
            success=False,
            model_path="/test",
            error="some error",
        )
        assert "success=False" in repr(failure)
        assert "some error" in repr(failure)


class TestConverterStructLayouts:
    """Tests that Python ctypes structs match Zig extern structs.

    These tests prevent ABI mismatches that cause segfaults or Bus errors.
    If any test fails, the Python struct definition is out of sync
    with the corresponding Zig struct in converter/scheme.zig.
    """

    def test_override_rule_fields(self):
        """OverrideRule has correct fields in order."""
        import ctypes

        from talu.converter import OverrideRule

        expected_fields = [
            ("pattern", ctypes.c_char_p),
            ("scheme", ctypes.c_uint32),
        ]

        actual_fields = OverrideRule._fields_

        assert len(actual_fields) == len(expected_fields), (
            f"Field count mismatch: expected {len(expected_fields)}, got {len(actual_fields)}"
        )

        for (exp_name, exp_type), (act_name, act_type) in zip(
            expected_fields, actual_fields, strict=True
        ):
            assert exp_name == act_name, (
                f"Field name mismatch: expected '{exp_name}', got '{act_name}'"
            )
            assert exp_type == act_type, f"Field '{exp_name}' type mismatch"

    def test_convert_options_fields(self):
        """ConvertOptions has correct fields in order."""
        import ctypes

        from talu.converter import MAX_OVERRIDES, ConvertOptions, OverrideRule

        expected_fields = [
            ("scheme", ctypes.c_uint32),
            ("force", ctypes.c_bool),
            ("offline", ctypes.c_bool),
            ("destination", ctypes.c_char_p),
            ("overrides", OverrideRule * MAX_OVERRIDES),
            ("num_overrides", ctypes.c_uint32),
            ("max_shard_size", ctypes.c_size_t),
            ("dry_run", ctypes.c_bool),
            ("platform", ctypes.c_uint32),
            ("quant", ctypes.c_uint32),
            ("use_platform_quant", ctypes.c_bool),
            # Unified progress callback (CProgressCallback)
            ("progress_callback", ctypes.c_void_p),
            ("progress_user_data", ctypes.c_void_p),
        ]

        actual_fields = ConvertOptions._fields_

        assert len(actual_fields) == len(expected_fields), (
            f"Field count mismatch: expected {len(expected_fields)}, got {len(actual_fields)}"
        )

        for i, ((exp_name, exp_type), (act_name, act_type)) in enumerate(
            zip(expected_fields, actual_fields, strict=True)
        ):
            assert exp_name == act_name, (
                f"Field {i} name mismatch: expected '{exp_name}', got '{act_name}'"
            )
            # For array types, compare the base type and length
            if hasattr(exp_type, "_length_") and hasattr(act_type, "_length_"):
                assert exp_type._type_ == act_type._type_, (
                    f"Field '{exp_name}' array element type mismatch"
                )
                assert exp_type._length_ == act_type._length_, (
                    f"Field '{exp_name}' array length mismatch"
                )
            else:
                assert exp_type == act_type, (
                    f"Field '{exp_name}' type mismatch: expected {exp_type}, got {act_type}"
                )

    def test_convert_result_fields(self):
        """ConvertResult has correct fields in order."""
        import ctypes

        from talu.converter import ConvertResult

        expected_fields = [
            ("output_path", ctypes.c_void_p),
            ("error_msg", ctypes.c_void_p),
            ("success", ctypes.c_bool),
        ]

        actual_fields = ConvertResult._fields_

        assert len(actual_fields) == len(expected_fields), (
            f"Field count mismatch: expected {len(expected_fields)}, got {len(actual_fields)}"
        )

        for (exp_name, exp_type), (act_name, act_type) in zip(
            expected_fields, actual_fields, strict=True
        ):
            assert exp_name == act_name, (
                f"Field name mismatch: expected '{exp_name}', got '{act_name}'"
            )
            assert exp_type == act_type, f"Field '{exp_name}' type mismatch"

    def test_max_overrides_matches_zig(self):
        """MAX_OVERRIDES constant matches Zig."""
        from talu.converter import MAX_OVERRIDES

        # Zig defines MAX_OVERRIDES = 32
        assert MAX_OVERRIDES == 32, f"MAX_OVERRIDES should be 32, got {MAX_OVERRIDES}"

    def test_convert_options_size_reasonable(self):
        """ConvertOptions size is reasonable for the field count."""
        import ctypes

        from talu.converter import ConvertOptions

        size = ctypes.sizeof(ConvertOptions)
        # ConvertOptions contains:
        # - scheme: u32 (4 bytes)
        # - force: bool (1 byte, padded)
        # - offline: bool (1 byte, padded)
        # - destination: pointer (8 bytes on 64-bit)
        # - overrides: 32 * OverrideRule (32 * 16 = 512 bytes on 64-bit)
        # - num_overrides: u32 (4 bytes)
        # Total: ~530+ bytes with padding
        assert 500 <= size <= 600, f"Unexpected ConvertOptions size: {size} bytes"
