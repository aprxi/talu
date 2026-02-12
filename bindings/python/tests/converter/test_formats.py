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

    def test_scheme_info_has_gaf_schemes(self):
        """SCHEME_INFO contains GAF (grouped affine) schemes."""
        assert "gaf4_32" in SCHEME_INFO
        assert "gaf4_64" in SCHEME_INFO
        assert "gaf4_128" in SCHEME_INFO
        assert "gaf8_32" in SCHEME_INFO
        assert "gaf8_64" in SCHEME_INFO
        assert "gaf8_128" in SCHEME_INFO

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

    def test_gaf_schemes_have_gaf_category(self):
        """GAF schemes have category='gaf'."""
        gaf_schemes = ["gaf4_32", "gaf4_64", "gaf4_128", "gaf8_32", "gaf8_64", "gaf8_128"]
        for scheme_name in gaf_schemes:
            assert SCHEME_INFO[scheme_name]["category"] == "gaf"

    def test_hardware_schemes_have_hardware_category(self):
        """Hardware schemes have category='hardware'."""
        hardware_schemes = ["fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4"]
        for scheme_name in hardware_schemes:
            assert SCHEME_INFO[scheme_name]["category"] == "hardware"

    def test_gaf_schemes_have_group_size(self):
        """GAF schemes have group_size field."""
        gaf_schemes = ["gaf4_32", "gaf4_64", "gaf4_128", "gaf8_32", "gaf8_64", "gaf8_128"]
        for scheme_name in gaf_schemes:
            assert "group_size" in SCHEME_INFO[scheme_name], f"{scheme_name} missing 'group_size'"

    def test_gaf_schemes_have_mlx_compatible_flag(self):
        """GAF schemes have mlx_compatible flag."""
        gaf_schemes = ["gaf4_32", "gaf4_64", "gaf4_128", "gaf8_32", "gaf8_64", "gaf8_128"]
        for scheme_name in gaf_schemes:
            assert SCHEME_INFO[scheme_name].get("mlx_compatible") is True


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

        # GAF schemes
        assert "gaf4_64" in result
        assert "gaf4_32" in result
        assert "gaf8_64" in result

    def test_list_schemes_include_unimplemented(self):
        """list_schemes(include_unimplemented=True) includes all schemes."""
        result = list_schemes(include_unimplemented=True)

        # Should include unimplemented schemes
        assert "fp8_e4m3" in result
        assert "mxfp4" in result
        assert "nvfp4" in result

    def test_list_schemes_filter_by_category_gaf(self):
        """list_schemes(category='gaf') returns only GAF schemes."""
        result = list_schemes(category="gaf")

        for name, info in result.items():
            assert info["category"] == "gaf", f"{name} should be gaf category"

        # Should have GAF schemes
        assert "gaf4_64" in result
        assert "gaf4_32" in result

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

    def test_schemes_contains_gaf_schemes(self):
        """schemes() contains GAF schemes."""
        result = schemes()

        assert "gaf4_32" in result
        assert "gaf4_64" in result
        assert "gaf4_128" in result

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
