"""
Tests for converter dry_run functionality.

Tests the dry_run parameter that estimates conversion without writing files.
"""

from talu.converter import convert


class TestDryRunEstimation:
    """Tests for dry_run estimation mode."""

    def test_dry_run_returns_dict(self, synthetic_model, temp_output_dir):
        """dry_run=True returns a dict instead of a path."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            dry_run=True,
        )
        assert isinstance(result, dict)

    def test_dry_run_has_required_keys(self, synthetic_model, temp_output_dir):
        """dry_run result contains all required keys."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            dry_run=True,
        )
        assert "total_params" in result
        assert "estimated_size_bytes" in result
        assert "shard_count" in result
        assert "scheme" in result
        assert "bits_per_param" in result

    def test_dry_run_total_params_positive(self, synthetic_model, temp_output_dir):
        """dry_run reports positive total parameters."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            dry_run=True,
        )
        assert result["total_params"] > 0

    def test_dry_run_estimated_size_positive(self, synthetic_model, temp_output_dir):
        """dry_run reports positive estimated size."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            dry_run=True,
        )
        assert result["estimated_size_bytes"] > 0

    def test_dry_run_scheme_name_matches(self, synthetic_model, temp_output_dir):
        """dry_run reports correct scheme name."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            dry_run=True,
        )
        assert result["scheme"] == "gaf4_64"

    def test_dry_run_default_shard_count_is_one(self, synthetic_model, temp_output_dir):
        """dry_run with no max_shard_size reports shard_count=1."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            dry_run=True,
        )
        assert result["shard_count"] == 1

    def test_dry_run_with_shard_size_increases_shard_count(
        self, synthetic_model, temp_output_dir
    ):
        """dry_run with small max_shard_size increases shard_count."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            max_shard_size="1KB",  # Very small to force multiple shards
            dry_run=True,
        )
        # If estimated size > 1KB, should have multiple shards
        if result["estimated_size_bytes"] > 1024:
            assert result["shard_count"] > 1

    def test_dry_run_bits_per_param_reasonable(self, synthetic_model, temp_output_dir):
        """dry_run reports reasonable bits_per_param for scheme."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            dry_run=True,
        )
        # GAF4_64 is ~4.5 bits per param
        assert 4.0 < result["bits_per_param"] < 5.0


class TestDryRunSchemes:
    """Tests for dry_run with different schemes."""

    def test_dry_run_gaf4_64(self, synthetic_model, temp_output_dir):
        """dry_run works with gaf4_64 scheme."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            dry_run=True,
        )
        assert result["scheme"] == "gaf4_64"
        # GAF4_64 is ~4.5 bits per param
        assert 4.0 < result["bits_per_param"] < 5.0

    def test_dry_run_gaf8_64(self, synthetic_model, temp_output_dir):
        """dry_run works with gaf8_64 scheme."""
        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf8_64",
            dry_run=True,
        )
        assert result["scheme"] == "gaf8_64"
        # GAF8_64 is ~8.5 bits per param
        assert 8.0 < result["bits_per_param"] < 9.0


class TestDryRunConvenienceFunction:
    """Tests for dry_run via convert() function."""

    def test_convert_function_dry_run(self, synthetic_model, convert_func, temp_output_dir):
        """convert() function supports dry_run parameter."""
        result = convert_func(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            scheme="gaf4_64",
            dry_run=True,
        )
        assert isinstance(result, dict)
        assert "total_params" in result
        assert "estimated_size_bytes" in result


class TestDryRunNoFileOutput:
    """Tests verifying dry_run doesn't write files."""

    def test_dry_run_does_not_create_output_dir(self, synthetic_model, temp_output_dir):
        """dry_run mode does not create output directory."""
        import os

        # Define a non-existent output path
        output_subdir = os.path.join(temp_output_dir, "dry_run_test_output")
        assert not os.path.exists(output_subdir)

        result = convert(
            str(synthetic_model.path),
            output_dir=temp_output_dir,
            destination=output_subdir,
            scheme="gaf4_64",
            dry_run=True,
        )

        # Should return dict, not create files
        assert isinstance(result, dict)
        # Output directory should NOT exist (no files written)
        assert not os.path.exists(output_subdir)
