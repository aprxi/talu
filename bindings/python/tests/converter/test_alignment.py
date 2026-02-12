"""Tests for memory alignment handling in converter.

These tests verify that the converter handles unaligned memory correctly,
preventing Bus errors that can occur when SafeTensors files have tensor
data at arbitrary offsets.

The intermittent Bus errors were caused by:
1. SafeTensors files storing tensor data at offsets determined by JSON header length
2. F32 tensor data could be at addresses not aligned to 4 bytes
3. Using std.mem.bytesAsSlice which checks alignment at runtime

These tests ensure the fixes remain in place.
"""

import json
import struct

import pytest


class TestUnalignedSafeTensorsData:
    """Tests that converter handles SafeTensors files with unaligned tensor data."""

    def _create_safetensors_with_header_length(
        self, header_length: int, tensor_data: bytes, tensor_name: str = "weight"
    ) -> bytes:
        """Create a SafeTensors file with a specific header length.

        By controlling header length, we can control the alignment of tensor data.
        header_length should include any padding needed.
        """
        # Calculate data offset (header comes after 8-byte length prefix)
        data_start = 0
        data_end = len(tensor_data)

        # Create header JSON
        header = {
            tensor_name: {
                "dtype": "F32",
                "shape": [len(tensor_data) // 4],
                "data_offsets": [data_start, data_end],
            }
        }
        header_json = json.dumps(header)

        # Pad header to desired length
        if len(header_json) > header_length:
            raise ValueError(
                f"Header JSON ({len(header_json)}) exceeds target length ({header_length})"
            )
        header_json = header_json + " " * (header_length - len(header_json))

        # Build file: [8-byte length][header][tensor data]
        result = struct.pack("<Q", header_length) + header_json.encode() + tensor_data
        return result

    def test_tensor_data_at_odd_offset(self, tmp_path):
        """Test handling tensor data at an odd byte offset (worst case for alignment)."""
        # Create F32 tensor data (4 floats = 16 bytes)
        tensor_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)

        # Header length of 69 means tensor data starts at offset 8 + 69 = 77 (odd)
        # This is unaligned for f32 (requires 4-byte alignment)
        # Note: minimum header JSON is ~67 bytes, so we need at least that
        safetensors_data = self._create_safetensors_with_header_length(
            header_length=69, tensor_data=tensor_data
        )

        # Verify the tensor data offset is indeed odd
        data_offset = 8 + 69
        assert data_offset % 4 != 0, "Tensor data should be at unaligned offset"

        # Write to file
        model_dir = tmp_path / "unaligned_model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(safetensors_data)

        # Create minimal config.json
        config = {
            "architectures": ["TestModel"],
            "model_type": "test",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "vocab_size": 100,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        # This should NOT cause a Bus error
        # The converter should handle unaligned F32 data correctly
        from talu.converter import describe

        # describe() reads tensor metadata which exercises the SafeTensors loader
        try:
            info = describe(str(model_dir))
            # If we get here without Bus error, the test passes
            assert info is not None
        except Exception as e:
            # Any exception other than Bus error is acceptable for this minimal model
            # (it may fail validation, but shouldn't crash)
            assert "Bus error" not in str(e)

    def test_tensor_data_at_offset_mod_2(self, tmp_path):
        """Test handling tensor data at offset divisible by 2 but not 4."""
        tensor_data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)

        # Header length of 70 means tensor data starts at offset 8 + 70 = 78
        # 78 % 2 == 0 but 78 % 4 == 2 (aligned for u16 but not f32)
        # Note: minimum header JSON is ~67 bytes
        safetensors_data = self._create_safetensors_with_header_length(
            header_length=70, tensor_data=tensor_data
        )

        data_offset = 8 + 70
        assert data_offset % 2 == 0, "Should be 2-byte aligned"
        assert data_offset % 4 != 0, "Should NOT be 4-byte aligned"

        model_dir = tmp_path / "half_aligned_model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_bytes(safetensors_data)

        config = {
            "architectures": ["TestModel"],
            "model_type": "test",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "vocab_size": 100,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        from talu.converter import describe

        try:
            info = describe(str(model_dir))
            assert info is not None
        except Exception as e:
            assert "Bus error" not in str(e)


class TestQuantizationAlignmentRegression:
    """Regression tests for quantization alignment issues.

    These tests verify that quantization functions allocate memory with
    proper alignment.
    """

    @pytest.fixture
    def reference_model(self, test_model_path):
        """Get path to reference model for conversion tests."""
        return test_model_path

    def test_gaf4_64_conversion_no_bus_error(self, reference_model, tmp_path):
        """GAF4_64 conversion should not cause Bus errors from alignment issues."""
        from talu.converter import convert

        output_dir = tmp_path / "gaf4_64_output"

        try:
            result = convert(reference_model, scheme="gaf4_64", output_dir=str(output_dir))
            assert result is not None
        except Exception as e:
            # Check it's not a Bus error (other errors may be acceptable)
            error_str = str(e).lower()
            if "already quantized" in error_str:
                pytest.skip("Test requires FP16 source model, got quantized model")
            assert "bus error" not in error_str
            assert "sigbus" not in error_str

    def test_gaf8_64_conversion_no_bus_error(self, reference_model, tmp_path):
        """GAF8_64 conversion should not cause Bus errors from alignment issues."""
        from talu.converter import convert

        output_dir = tmp_path / "gaf8_64_output"

        try:
            result = convert(reference_model, scheme="gaf8_64", output_dir=str(output_dir))
            assert result is not None
        except Exception as e:
            error_str = str(e).lower()
            if "already quantized" in error_str:
                pytest.skip("Test requires FP16 source model, got quantized model")
            assert "bus error" not in error_str
            assert "sigbus" not in error_str


class TestRepeatedConversionStability:
    """Tests that conversions are stable across multiple runs.

    Intermittent Bus errors often manifest as failures that only occur
    sometimes, depending on memory allocation patterns.
    """

    @pytest.fixture
    def reference_model(self, test_model_path):
        """Get path to reference model."""
        return test_model_path

    def test_repeated_gaf4_64_conversions(self, reference_model, tmp_path):
        """Multiple GAF4_64 conversions should all succeed without crashes.

        This test runs multiple conversions to catch intermittent alignment
        issues that only manifest with certain memory allocation patterns.
        """
        from talu.converter import convert

        num_iterations = 3
        for i in range(num_iterations):
            output_dir = tmp_path / f"gaf4_64_run_{i}"
            try:
                result = convert(reference_model, scheme="gaf4_64", output_dir=str(output_dir))
                assert result is not None, f"Iteration {i} returned None"
            except Exception as e:
                error_str = str(e).lower()
                # Skip if model is already quantized (GAF4 test model)
                if "already quantized" in error_str:
                    pytest.skip("Test requires FP16 source model, got quantized model")
                assert "bus error" not in error_str, f"Bus error on iteration {i}"
                assert "sigbus" not in error_str, f"SIGBUS on iteration {i}"
                # Re-raise non-alignment errors
                if "bus" not in error_str and "alignment" not in error_str:
                    raise

    def test_alternating_scheme_conversions(self, reference_model, tmp_path):
        """Alternating between schemes should not cause alignment issues.

        Different schemes use different group sizes with different alignment
        requirements. Switching between them exercises more code paths.
        """
        from talu.converter import convert

        schemes = ["gaf4_32", "gaf4_64", "gaf8_64"]

        for scheme in schemes:
            output_dir = tmp_path / f"{scheme}_run"
            try:
                result = convert(reference_model, scheme=scheme, output_dir=str(output_dir))
                assert result is not None, f"Scheme {scheme} returned None"
            except Exception as e:
                error_str = str(e).lower()
                # Skip if model is already quantized (GAF4 test model)
                if "already quantized" in error_str:
                    pytest.skip("Test requires FP16 source model, got quantized model")
                assert "bus error" not in error_str, f"Bus error with scheme {scheme}"
                assert "sigbus" not in error_str, f"SIGBUS with scheme {scheme}"
