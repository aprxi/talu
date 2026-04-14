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

import contextlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

if os.name != "nt":
    import fcntl


_CONVERTER_LOCK_PATH = Path(tempfile.gettempdir()) / "talu_converter_alignment.lock"
_SUBPROCESS_RESULT_PREFIX = "TALU_CONVERTER_SUBPROCESS_RESULT="
_DEFAULT_CONVERT_TIMEOUT_SECONDS = 10
_DEFAULT_LOCK_TIMEOUT_SECONDS = 10
_STABILITY_TESTS_ENV = "TALU_RUN_CONVERTER_STABILITY"
_MODEL_UNAVAILABLE_MESSAGES = ("model not found", "api returned 401 without token")


@contextlib.contextmanager
def _converter_alignment_lock():
    """Serialize heavy converter runs across xdist workers."""
    lock_file = _CONVERTER_LOCK_PATH.open("w")
    try:
        if "fcntl" in globals():
            timeout_seconds = _DEFAULT_LOCK_TIMEOUT_SECONDS
            raw_timeout = os.environ.get("TALU_TEST_LOCK_TIMEOUT_SECONDS")
            if raw_timeout:
                try:
                    timeout_seconds = max(1, int(raw_timeout))
                except ValueError:
                    timeout_seconds = _DEFAULT_LOCK_TIMEOUT_SECONDS

            start = __import__("time").monotonic()
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if (__import__("time").monotonic() - start) >= timeout_seconds:
                        raise AssertionError(
                            "Converter alignment lock acquisition timed out "
                            f"after {timeout_seconds}s"
                        )
                    __import__("time").sleep(0.05)
        yield
    finally:
        if "fcntl" in globals():
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _run_convert_subprocess(model: str, scheme: str, output_dir: str) -> tuple[bool, str | None]:
    """Run talu.convert in an isolated subprocess.

    Returns:
        (success, error_message). success=True means conversion completed.
        success=False returns the conversion error string.

    Raises:
        AssertionError: if subprocess crashes before reporting a structured result.
    """
    script = f"""
import json
import talu
import sys

model = sys.argv[1]
scheme = sys.argv[2]
output_dir = sys.argv[3]

try:
    result = talu.convert(model, scheme=scheme, output_dir=output_dir)
    print("{_SUBPROCESS_RESULT_PREFIX}" + json.dumps({{"ok": True, "result": result}}))
except Exception as exc:
    print("{_SUBPROCESS_RESULT_PREFIX}" + json.dumps({{"ok": False, "error": str(exc), "type": type(exc).__name__}}))
    raise SystemExit(1)
"""
    timeout_seconds = _DEFAULT_CONVERT_TIMEOUT_SECONDS
    raw_timeout = os.environ.get("TALU_TEST_CONVERTER_TIMEOUT_SECONDS")
    if raw_timeout:
        try:
            timeout_seconds = max(1, int(raw_timeout))
        except ValueError:
            timeout_seconds = _DEFAULT_CONVERT_TIMEOUT_SECONDS

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, model, scheme, output_dir],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"conversion timed out after {timeout_seconds}s") from exc

    payload = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_SUBPROCESS_RESULT_PREFIX):
            payload = json.loads(line[len(_SUBPROCESS_RESULT_PREFIX) :])
            break

    if payload is None:
        raise AssertionError(
            "Converter subprocess crashed before returning structured result "
            f"(returncode={proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    if payload.get("ok") is True:
        return True, None
    return False, str(payload.get("error", "conversion failed"))


def _has_local_calibration_rows_cache() -> bool:
    """Return True when local Pile calibration rows cache is available."""
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    cache_dir = hf_home / "hub" / "datasets--NeelNanda--pile-10k" / "snapshots" / "main"
    if not cache_dir.exists():
        return False
    return any(cache_dir.glob("rows-offset-*-length-*.json"))


def _skip_reason_for_conversion_error(error: str | None) -> str | None:
    """Return a skip reason for non-regression infra conditions."""
    error_str = (error or "").lower()
    if "already quantized" in error_str:
        return "Test requires FP16 source model, got quantized model"
    if any(msg in error_str for msg in _MODEL_UNAVAILABLE_MESSAGES):
        return "Test requires an accessible source model for conversion"
    return None


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

    @pytest.fixture
    def conversion_lock(self):
        with _converter_alignment_lock():
            yield

    @pytest.fixture
    def isolated_talu_home(self, tmp_path, monkeypatch):
        talu_home = tmp_path / "talu_home"
        talu_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("TALU_HOME", str(talu_home))
        monkeypatch.setenv("BACKEND", "cpu")
        monkeypatch.setenv("TALU_CONVERT_CALIB_GPU", "0")
        return talu_home

    def test_tq4_64_conversion_no_bus_error(
        self, reference_model, tmp_path, conversion_lock, isolated_talu_home
    ):
        """TQ4_64 conversion should not cause Bus errors from alignment issues."""
        output_dir = tmp_path / "tq4_64_output"
        success, error = _run_convert_subprocess(
            reference_model,
            "tq4_64",
            str(output_dir),
        )
        if success:
            return
        if (skip_reason := _skip_reason_for_conversion_error(error)) is not None:
            pytest.skip(skip_reason)
        error_str = (error or "").lower()
        assert "bus error" not in error_str
        assert "sigbus" not in error_str
        pytest.fail(error or "conversion failed")

    def test_tq8_conversion_no_bus_error(
        self, reference_model, tmp_path, conversion_lock, isolated_talu_home
    ):
        """TQ8 conversion should not cause Bus errors from alignment issues."""
        output_dir = tmp_path / "tq8_output"
        success, error = _run_convert_subprocess(
            reference_model,
            "tq8",
            str(output_dir),
        )
        if success:
            return
        if (skip_reason := _skip_reason_for_conversion_error(error)) is not None:
            pytest.skip(skip_reason)
        error_str = (error or "").lower()
        assert "bus error" not in error_str
        assert "sigbus" not in error_str
        pytest.fail(error or "conversion failed")


class TestRepeatedConversionStability:
    """Tests that conversions are stable across multiple runs.

    Intermittent Bus errors often manifest as failures that only occur
    sometimes, depending on memory allocation patterns.
    """

    @pytest.fixture
    def reference_model(self, test_model_path):
        """Get path to reference model."""
        return test_model_path

    @pytest.fixture
    def conversion_lock(self):
        with _converter_alignment_lock():
            yield

    @pytest.fixture
    def isolated_talu_home(self, tmp_path, monkeypatch):
        talu_home = tmp_path / "talu_home"
        talu_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("TALU_HOME", str(talu_home))
        monkeypatch.setenv("BACKEND", "cpu")
        monkeypatch.setenv("TALU_CONVERT_CALIB_GPU", "0")
        return talu_home

    def test_repeated_tq4_64_conversions(
        self, reference_model, tmp_path, conversion_lock, isolated_talu_home, monkeypatch
    ):
        """Multiple TQ4_64 conversions should all succeed without crashes.

        This test runs multiple conversions to catch intermittent alignment
        issues that only manifest with certain memory allocation patterns.
        """
        if os.environ.get(_STABILITY_TESTS_ENV, "").lower() not in {"1", "true", "yes"}:
            pytest.skip(
                f"Stability loop is opt-in. Set {_STABILITY_TESTS_ENV}=1 to run this test."
            )
        if not _has_local_calibration_rows_cache():
            pytest.skip(
                "Test requires local calibration rows cache "
                "(datasets--NeelNanda--pile-10k rows-offset-*.json)."
            )

        num_iterations = 3
        for i in range(num_iterations):
            output_dir = tmp_path / f"tq4_64_run_{i}"
            run_home = tmp_path / f"talu_home_iter_{i}"
            run_home.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv("TALU_HOME", str(run_home))

            success, error = _run_convert_subprocess(
                reference_model,
                "tq4_64",
                str(output_dir),
            )
            if success:
                continue
            if (skip_reason := _skip_reason_for_conversion_error(error)) is not None:
                pytest.skip(skip_reason)
            error_str = (error or "").lower()
            assert "bus error" not in error_str, f"Bus error on iteration {i}"
            assert "sigbus" not in error_str, f"SIGBUS on iteration {i}"
            pytest.fail(f"Iteration {i} failed unexpectedly: {error}")

    def test_alternating_scheme_conversions(
        self, reference_model, tmp_path, conversion_lock, isolated_talu_home, monkeypatch
    ):
        """Alternating between schemes should not cause alignment issues.

        Different schemes use different group sizes with different alignment
        requirements. Switching between them exercises more code paths.
        """
        if os.environ.get(_STABILITY_TESTS_ENV, "").lower() not in {"1", "true", "yes"}:
            pytest.skip(
                f"Stability loop is opt-in. Set {_STABILITY_TESTS_ENV}=1 to run this test."
            )
        if not _has_local_calibration_rows_cache():
            pytest.skip(
                "Test requires local calibration rows cache "
                "(datasets--NeelNanda--pile-10k rows-offset-*.json)."
            )

        schemes = ["tq4", "tq4_64", "tq8"]

        for scheme in schemes:
            output_dir = tmp_path / f"{scheme}_run"
            run_home = tmp_path / f"talu_home_{scheme}"
            run_home.mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv("TALU_HOME", str(run_home))

            success, error = _run_convert_subprocess(
                reference_model,
                scheme,
                str(output_dir),
            )
            if success:
                continue
            if (skip_reason := _skip_reason_for_conversion_error(error)) is not None:
                pytest.skip(skip_reason)
            error_str = (error or "").lower()
            assert "bus error" not in error_str, f"Bus error with scheme {scheme}"
            assert "sigbus" not in error_str, f"SIGBUS with scheme {scheme}"
            pytest.fail(f"Scheme {scheme} failed unexpectedly: {error}")
