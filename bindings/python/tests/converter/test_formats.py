"""
Tests for talu.converter.formats module.

Tests for scheme discovery and metadata: SCHEME_INFO, list_schemes(), schemes().
"""

from talu.converter import SCHEME_INFO, list_schemes, schemes


class TestSchemeInfo:
    """Tests for SCHEME_INFO constant."""

    def test_scheme_info_is_dict(self):
        """SCHEME_INFO is a dictionary."""
        assert isinstance(SCHEME_INFO, dict)

    def test_scheme_info_has_tq_schemes(self):
        """SCHEME_INFO contains TQ (grouped affine) schemes."""
        assert "tq4" in SCHEME_INFO
        assert "tq4_64" in SCHEME_INFO
        assert "tq4_128" in SCHEME_INFO
        assert "tq8_32" in SCHEME_INFO
        assert "tq8" in SCHEME_INFO
        assert "tq8_128" in SCHEME_INFO

    def test_scheme_info_has_hardware_schemes(self):
        """SCHEME_INFO contains hardware float schemes."""
        assert "fp8_e4m3" in SCHEME_INFO
        assert "fp8_e5m2" in SCHEME_INFO
        assert "mxfp4" in SCHEME_INFO
        assert "nvfp4" in SCHEME_INFO

    def test_schemes_have_required_metadata(self):
        """All schemes have required metadata fields."""
        for scheme_name, scheme_info in SCHEME_INFO.items():
            assert "category" in scheme_info, f"{scheme_name} missing 'category'"
            assert "bits" in scheme_info, f"{scheme_name} missing 'bits'"
            assert "description" in scheme_info, f"{scheme_name} missing 'description'"
            assert "quality" in scheme_info, f"{scheme_name} missing 'quality'"
            assert "status" in scheme_info, f"{scheme_name} missing 'status'"

    def test_tq_schemes_have_tq_category(self):
        """TQ schemes have category='tq'."""
        tq_schemes = ["tq4", "tq4_64", "tq4_128", "tq8_32", "tq8", "tq8_128"]
        for scheme_name in tq_schemes:
            assert SCHEME_INFO[scheme_name]["category"] == "tq"

    def test_hardware_schemes_have_hardware_category(self):
        """Hardware schemes have category='hardware'."""
        hardware_schemes = ["fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4"]
        for scheme_name in hardware_schemes:
            assert SCHEME_INFO[scheme_name]["category"] == "hardware"

    def test_tq_schemes_have_group_size(self):
        """TQ schemes have group_size field."""
        tq_schemes = ["tq4", "tq4_64", "tq4_128", "tq8_32", "tq8", "tq8_128"]
        for scheme_name in tq_schemes:
            assert "group_size" in SCHEME_INFO[scheme_name], f"{scheme_name} missing 'group_size'"

    def test_tq_schemes_have_mlx_compatible_flag(self):
        """TQ schemes expose mlx_compatible when that metadata is present."""
        tq_schemes = ["tq4", "tq4_64", "tq4_128", "tq8_32", "tq8", "tq8_128"]
        for scheme_name in tq_schemes:
            if "mlx_compatible" in SCHEME_INFO[scheme_name]:
                assert SCHEME_INFO[scheme_name]["mlx_compatible"] is True


class TestListSchemes:
    """Tests for list_schemes() function."""

    def test_list_schemes_returns_dict(self):
        """list_schemes() returns a dictionary."""
        result = list_schemes()
        assert isinstance(result, dict)

    def test_list_schemes_excludes_unimplemented_by_default(self):
        """list_schemes() excludes unimplemented schemes by default."""
        result = list_schemes()

        for name, info in result.items():
            assert info.get("status") != "not implemented", f"{name} should not be included"

    def test_list_schemes_includes_implemented(self):
        """list_schemes() includes implemented schemes."""
        result = list_schemes()

        # TQ schemes
        assert "tq4_64" in result
        assert "tq4" in result
        assert "tq8" in result

    def test_list_schemes_include_unimplemented(self):
        """list_schemes(include_unimplemented=True) includes all schemes."""
        result = list_schemes(include_unimplemented=True)

        # Should include unimplemented schemes
        assert "fp8_e4m3" in result
        assert "mxfp4" in result
        assert "nvfp4" in result

    def test_list_schemes_filter_by_category_tq(self):
        """list_schemes(category='tq') returns only TQ schemes."""
        result = list_schemes(category="tq")

        for name, info in result.items():
            assert info["category"] == "tq", f"{name} should be tq category"

        # Should have TQ schemes
        assert "tq4_64" in result
        assert "tq4" in result

    def test_list_schemes_returns_copy(self):
        """list_schemes() returns a copy, not the original."""
        result1 = list_schemes()
        result2 = list_schemes()

        # Modifying one should not affect the other
        result1["test"] = {"name": "Test"}
        assert "test" not in result2


class TestSchemes:
    """Tests for schemes() function."""

    def test_schemes_returns_list(self):
        """schemes() returns a list."""
        result = schemes()
        assert isinstance(result, list)

    def test_schemes_contains_tq_schemes(self):
        """schemes() contains TQ schemes."""
        result = schemes()

        assert "tq4" in result
        assert "tq4_64" in result
        assert "tq4_128" in result

    def test_schemes_excludes_unimplemented(self):
        """schemes() excludes unimplemented schemes."""
        result = schemes()

        # Hardware schemes are not implemented
        assert "fp8_e4m3" not in result
        assert "mxfp4" not in result
        assert "nvfp4" not in result

    def test_schemes_is_sorted(self):
        """schemes() returns sorted list."""
        result = schemes()
        assert result == sorted(result)

    def test_schemes_all_strings(self):
        """schemes() returns list of strings."""
        result = schemes()

        for item in result:
            assert isinstance(item, str)
