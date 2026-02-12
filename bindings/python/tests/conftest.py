"""
Global pytest fixtures for talu integration tests.

This module provides:
- Library build verification
- Fault handling for native crashes
- Test data helpers
- Shared session/tensor fixtures

=============================================================================
Skip vs Xfail Policy (FINALIZED)
=============================================================================

pytest.skip(): Infrastructure/environmental issues - NOT test failures:
  - Missing compiler or build tools
  - Library build failures
  - Missing test models (user hasn't downloaded them)
  - Missing optional dependencies (transformers, torch)
  These are prerequisites, not talu bugs.

pytest.xfail(): Known talu limitations we want to track:
  - Features not yet implemented (e.g., missing graph JSON for an architecture)
  - Known behavioral deviations from spec (e.g., Jinja2 scoping differences)
  - Error messages that don't yet include expected diagnostic info
  These appear in CI reports and alert when fixed (XPASS).

Decision tree for models:
  Model not cached/downloaded → skip (infrastructure - user needs to download)
  Model architecture not supported → xfail (known limitation we're tracking)
  Model tokenizer has known issues → xfail with reason in KNOWN_ISSUES

=============================================================================
Audit Mode (Zero-Xfail Run)
=============================================================================

To verify all xfails are still valid (none have been fixed without updating tests):
  pytest --runxfail  # Runs xfails as normal tests, fails if they pass

For a clean baseline run (skips all xfails):
  pytest -m "not xfail"  # Only runs tests expected to pass

CI should periodically run --runxfail to catch stale xfails (XPASS indicates
the issue is fixed and the test should be updated to a normal assertion).
"""

import ctypes
import faulthandler
import gc
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Enable faulthandler to trace native crashes (segfaults)
faulthandler.enable()


# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ZIG_OUT_LIB = PROJECT_ROOT / "zig-out" / "lib"


def _get_lib_path() -> Path:
    """Get the shared library path for the current platform."""
    if sys.platform == "win32":
        return ZIG_OUT_LIB / "talu.dll"
    elif sys.platform == "darwin":
        return ZIG_OUT_LIB / "libtalu.dylib"
    else:
        return ZIG_OUT_LIB / "libtalu.so"


# =============================================================================
# Build Fixtures
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def ensure_library_built():
    """
    Ensure the Zig library is built before running tests.

    This fixture runs once per test session and verifies the shared library
    exists. If not, it attempts to build it.

    Raises:
        pytest.skip: If build fails or library cannot be found
    """
    lib_path = _get_lib_path()

    if not lib_path.exists():
        print(f"\nLibrary not found at {lib_path}, attempting build...")
        try:
            result = subprocess.run(
                ["zig", "build"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            if result.returncode != 0:
                pytest.skip(f"Failed to build library:\n{result.stderr}")
        except FileNotFoundError:
            pytest.skip("Zig compiler not found. Install Zig to run tests.")
        except subprocess.TimeoutExpired:
            pytest.skip("Build timed out after 5 minutes.")

    if not lib_path.exists():
        pytest.skip(f"Library not found at {lib_path} after build attempt.")

    # Verify library can be loaded
    try:
        ctypes.CDLL(str(lib_path))
    except OSError as e:
        pytest.skip(f"Cannot load library: {e}")


# =============================================================================
# Talu Import Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def talu():
    """
    Import and return the talu module.

    This fixture ensures the library is built before importing.
    """
    import talu

    return talu


@pytest.fixture(scope="session")
def ops():
    """
    Import and return the talu.ops module.

    Provides access to low-level tensor operations.
    """
    from talu import ops

    return ops


# =============================================================================
# Model Fixtures
# =============================================================================

TEST_MODEL_URI_TEXT_RANDOM = os.environ.get(
    "TEST_MODEL_URI_TEXT_RANDOM", "llamafactory/tiny-random-Llama-3"
)
TEST_MODEL_URI_TEXT = os.environ.get("TEST_MODEL_URI_TEXT", "LiquidAI/LFM2-350M-GAF4")
TEST_MODEL_URI_TEXT_THINK = os.environ.get("TEST_MODEL_URI_TEXT_THINK", "Qwen/Qwen3-0.6B-GAF4")
TEST_MODEL_URI_EMBEDDING = os.environ.get(
    "TEST_MODEL_URI_EMBEDDING", "sentence-transformers/all-MiniLM-L6-v2-GAF8"
)

# HuggingFace model IDs for cross-model tokenizer/template comparison tests.
# Each entry maps a model family to its canonical HF model.
# Used by tokenizer reference tests to compare talu vs transformers output.
TEST_MODEL_HF_MINILM = os.environ.get(
    "TEST_MODEL_HF_MINILM", "sentence-transformers/all-MiniLM-L6-v2"
)

MODEL_REGISTRY = {
    "qwen3": {
        "name": "qwen3",
        "hf_id": os.environ.get("TEST_MODEL_HF_QWEN3", "Qwen/Qwen3-0.6B"),
        "model_types": ["qwen3", "qwen2.5", "qwen2", "qwen"],
    },
    "llama2": {
        "name": "llama2",
        "hf_id": os.environ.get("TEST_MODEL_HF_LLAMA2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        "model_types": ["llama2", "mistral", "yi", "vicuna", "tinyllama"],
    },
    "llama3": {
        "name": "llama3",
        "hf_id": os.environ.get("TEST_MODEL_HF_LLAMA3", "meta-llama/Llama-3.2-3B-Instruct"),
        "model_types": ["llama", "llama3", "llama3.1", "llama3.2"],
    },
    "gemma3": {
        "name": "gemma3",
        "hf_id": os.environ.get("TEST_MODEL_HF_GEMMA3", "google/gemma-3-1b-it"),
        "model_types": ["gemma3", "gemma3_text", "gemma2", "gemma"],
    },
    "phi4": {
        "name": "phi4",
        "hf_id": os.environ.get("TEST_MODEL_HF_PHI4", "microsoft/Phi-4-mini-instruct"),
        "model_types": ["phi3", "phi4", "phi"],
    },
    "granite3": {
        "name": "granite3",
        "hf_id": os.environ.get("TEST_MODEL_HF_GRANITE3", "ibm-granite/granite-3.3-2b-instruct"),
        "model_types": ["granite"],
    },
    "ministral3": {
        "name": "ministral3",
        "hf_id": os.environ.get(
            "TEST_MODEL_HF_MINISTRAL3", "mistralai/Ministral-3-3B-Instruct-2512"
        ),
        "model_types": ["ministral3", "mistral3"],
    },
}


@pytest.fixture(scope="session")
def test_model_path():
    """Return the text model URI (TEST_MODEL_URI_TEXT_RANDOM env var or default)."""
    return TEST_MODEL_URI_TEXT_RANDOM


# Note: The old `session` fixture that created Chat(model_path) is removed.
# Use `engine` fixture from tests/chat_session/conftest.py for LocalEngine,
# and create Chat() directly in tests (no model path needed).


# =============================================================================
# Memory Tracking Fixtures
# =============================================================================


@pytest.fixture
def memory_tracker():
    """
    Track memory usage for leak detection.

    Usage:
        def test_no_leak(memory_tracker):
            tracker = memory_tracker()
            initial = tracker.get_rss()

            # ... do stuff ...

            gc.collect()
            final = tracker.get_rss()
            assert final - initial < threshold
    """

    class MemoryTracker:
        @staticmethod
        def get_rss():
            """Get current RSS (Resident Set Size) in bytes."""
            try:
                import resource

                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            except ImportError:
                # Windows fallback
                import psutil

                return psutil.Process().memory_info().rss

        @staticmethod
        def force_gc():
            """Force garbage collection."""
            for _ in range(3):
                gc.collect()

    return MemoryTracker


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def random_seed():
    """Provide a fixed random seed for reproducible tests."""
    return 42


# =============================================================================
# Skip Markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "requires_model: marks tests requiring a model")
    config.addinivalue_line("markers", "memory: marks memory/leak detection tests")
    config.addinivalue_line("markers", "network: marks tests requiring network access")
    config.addinivalue_line(
        "markers", "integration: marks integration tests (subprocess-heavy, external deps)"
    )
