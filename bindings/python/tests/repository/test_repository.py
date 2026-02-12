"""
Tests for talu.repository module.

Tests the talu.repository module - REST-style model repository operations.

Note: Edge case tests (malformed cache, corrupted files, etc.) are in test_edge_cases.py.
"""

import json
import os

import pytest

from talu.repository import (
    cache_dir,
    cache_path,
    clear,
    delete,
    fetch,
    fetch_file,
    is_cached,
    is_model_id,
    list_files,
    list_models,
    resolve_path,
    search,
    size,
)

# =============================================================================
# Construction and Properties
# =============================================================================


class TestModuleProperties:
    """Tests for talu.repository module-level functions."""

    def test_cache_dir_returns_string(self):
        """cache_dir() returns a string ending with /hub."""
        path = cache_dir()
        assert isinstance(path, str)
        assert path.endswith("/hub")

    def test_cache_dir_respects_hf_home(self, monkeypatch, tmp_path):
        """cache_dir() respects HF_HOME environment variable."""
        custom_hf_home = str(tmp_path / "custom_hf")
        monkeypatch.setenv("HF_HOME", custom_hf_home)
        assert cache_dir() == f"{custom_hf_home}/hub"


class TestRepositoryImport:
    """Tests for talu.repository import paths."""

    def test_import_functions_from_repository(self):
        """Module-level functions are importable from talu.repository."""
        from talu.repository import (
            cache_dir,
            cache_path,
            clear,
            delete,
            fetch,
            fetch_file,
            is_cached,
            is_model_id,
            list_files,
            list_models,
            resolve_path,
            search,
            size,
        )

        assert callable(cache_dir)
        assert callable(is_model_id)

    def test_import_repository_module(self):
        """talu.repository module is importable."""
        import talu.repository

        assert hasattr(talu.repository, "cache_dir")
        assert hasattr(talu.repository, "is_model_id")


# =============================================================================
# Static Methods
# =============================================================================


class TestIsModelId:
    """Tests for is_model_id() static method."""

    def test_valid_model_ids(self):
        """Valid HuggingFace model IDs return True."""
        valid_ids = [
            "Foo/Bar-0B",
            "meta-llama/Llama-3.2-1B",
            "microsoft/Phi-4-mini-instruct",
            "google/gemma-3-1b-it",
            "org/model",
            "a/b",
        ]
        for model_id in valid_ids:
            assert is_model_id(model_id), f"Expected {model_id} to be valid"

    def test_invalid_model_ids(self):
        """Invalid model IDs return False."""
        invalid_ids = [
            "/path/to/model",
            "./models/qwen",
            "model-name",
            "org/sub/model",
            "",
        ]
        for path in invalid_ids:
            assert not is_model_id(path), f"Expected {path} to be invalid"


# =============================================================================
# Local Cache Operations
# =============================================================================


class TestListModelsOperation:
    """Tests for list_models() operation."""

    def test_list_models_returns_iterator(self):
        """list_models() returns an iterator."""
        result = list_models()
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_list_models_yields_strings(self):
        """list_models() yields string model IDs."""
        for model_id in list_models():
            assert isinstance(model_id, str)
            break  # Just check first item

    def test_list_models_yields_valid_model_ids(self):
        """list_models() yields valid model ID formats."""
        for model_id in list_models():
            assert "/" in model_id, f"Model ID should contain /: {model_id}"
            assert is_model_id(model_id), f"Should be valid model ID: {model_id}"
            break  # Just check first item


class TestListModelsSourceField:
    """Tests for list_models source field in the bindings layer."""

    def test_call_repo_list_models_yields_tuples(self):
        """call_repo_list_models yields (model_id, path, source) tuples."""
        from talu.repository import _call_repo_list_models as call_repo_list_models

        for entry in call_repo_list_models(True):
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            model_id, path, source = entry
            assert isinstance(model_id, str)
            assert isinstance(path, str)
            assert source in ("hub", "managed")
            break  # Just check first item

    def test_list_models_includes_talu_managed(self, tmp_path, monkeypatch):
        """list_models includes models from Talu managed cache."""
        from talu.repository import _call_repo_list_models as call_repo_list_models

        talu_home = tmp_path / "talu_home"
        model_dir = talu_home / "models" / "TestOrg" / "TestModel-GAF4"
        model_dir.mkdir(parents=True)

        # Create a weights file so require_weights passes
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 64)

        monkeypatch.setenv("TALU_HOME", str(talu_home))

        entries = list(call_repo_list_models(True))
        local_entries = [(mid, p, s) for mid, p, s in entries if s == "managed"]
        assert any(mid == "TestOrg/TestModel-GAF4" for mid, _, _ in local_entries), (
            f"Expected TestOrg/TestModel-GAF4 in local entries, got: {local_entries}"
        )


class TestGetPathOperation:
    """Tests for path() operation (REST: cache lookup only)."""

    def test_get_path_uncached_returns_none(self):
        """path() returns None for uncached models."""
        result = cache_path("definitely/not-a-real-model-12345")
        assert result is None

    def test_get_path_returns_string_or_none(self):
        """path() returns string or None."""
        result = cache_path("Foo/Bar-0B")
        assert result is None or isinstance(result, str)


class TestIsCachedOperation:
    """Tests for is_cached() operation (cache-only check)."""

    def test_is_cached_returns_bool(self):
        """is_cached() returns a boolean."""
        result = is_cached("definitely/not-a-real-model-12345")
        assert isinstance(result, bool)
        assert result is False

    def test_is_cached_uncached_returns_false(self):
        """is_cached() returns False for uncached models."""
        assert is_cached("definitely/not-a-real-model-12345") is False


class TestListFilesOperation:
    """Tests for list_files() operation."""

    def test_list_files_returns_iterator(self):
        """list_files() returns an iterator."""
        result = list_files("not-a-real-org-xyz/not-a-model-abc")
        assert hasattr(result, "__iter__")


class TestSizeOperation:
    """Tests for size() operation."""

    def test_size_returns_int(self):
        """size() returns an integer."""
        total = size()
        assert isinstance(total, int)
        assert total >= 0

    def test_size_nonexistent_model(self):
        """size() for non-existent model returns 0."""
        result_size = size("definitely/not-a-real-model-12345")
        assert result_size == 0


class TestDeleteOperation:
    """Tests for delete() operation (REST: _delete)."""

    def test_delete_nonexistent_returns_false(self):
        """delete() returns False for non-existent models."""
        assert delete("definitely/not-a-real-model-12345") is False


class TestClearOperation:
    """Tests for clear() operation."""

    def test_clear_is_callable(self):
        """clear() is a callable method."""
        assert callable(clear)


# =============================================================================
# Remote Operations
# =============================================================================


class TestFetchOperation:
    """Tests for fetch() operation (REST: _fetch).

    Note: These tests don't make actual network calls for non-existent models.
    """

    def test_fetch_invalid_returns_none(self):
        """fetch() returns None for invalid model IDs."""
        # Invalid model ID (no slash)
        result = fetch("invalid-no-slash")
        assert result is None


class TestRemoteOperations:
    """Tests for remote operations.

    Note: These tests may make network calls but use clearly invalid IDs
    to avoid downloading actual models.
    """

    def test_list_files_remote_returns_iterator(self):
        """list_files() for remote model ID returns an iterator."""
        result = list_files("not-a-real-org-xyz/not-a-model-abc")
        assert hasattr(result, "__iter__")

    def test_search_returns_iterator(self):
        """search() returns an iterator."""
        result = search("nonexistent-query-xyz-12345")
        assert hasattr(result, "__iter__")


# =============================================================================
# Deterministic Tests (with simple fixtures)
# =============================================================================


@pytest.fixture
def temp_hf_cache(tmp_path, monkeypatch):
    """Create a temporary HF cache with test models."""
    hf_home = tmp_path / "hf_home"
    hub_dir = hf_home / "hub"
    hub_dir.mkdir(parents=True)

    # Isolate both caches so real models don't leak into tests
    monkeypatch.setenv("HF_HOME", str(hf_home))
    talu_home = tmp_path / "talu"
    talu_home.mkdir(parents=True)
    monkeypatch.setenv("TALU_HOME", str(talu_home))

    return hub_dir


@pytest.fixture
def cached_model(temp_hf_cache):
    """Create a cached model in the temp HF cache."""
    model_dir = temp_hf_cache / "models--TestOrg--TestModel"
    snapshot_dir = model_dir / "snapshots" / "main"
    snapshot_dir.mkdir(parents=True)

    # Create minimal model files
    config = {"model_type": "test", "hidden_size": 768}
    (snapshot_dir / "config.json").write_text(json.dumps(config))
    (snapshot_dir / "model.safetensors").write_bytes(b"fake weights")

    return "TestOrg/TestModel"


class TestDeterministicListModels:
    """Deterministic tests for list_models() with temp cache."""

    def test_list_models_returns_cached_models(self, cached_model):
        """list_models() returns models that are cached."""
        models = list(list_models())
        assert cached_model in models

    def test_list_models_empty_cache(self, temp_hf_cache):
        """list_models() returns empty for empty cache."""
        models = list(list_models())
        assert models == []


class TestDeterministicGetPath:
    """Deterministic tests for path() with temp cache."""

    def test_get_path_returns_path_for_cached(self, cached_model):
        """path() returns path for cached model."""
        path = cache_path(cached_model)
        assert path is not None
        assert isinstance(path, str)
        assert "TestOrg" in path or "TestModel" in path

    def test_get_path_returns_none_for_uncached(self, temp_hf_cache):
        """path() returns None for uncached model."""
        path = cache_path("NotCached/Model")
        assert path is None


class TestDeterministicIsCached:
    """Deterministic tests for is_cached() with temp cache."""

    def test_is_cached_true_for_cached(self, cached_model):
        """is_cached() returns True for cached model."""
        assert is_cached(cached_model) is True

    def test_is_cached_false_for_uncached(self, temp_hf_cache):
        """is_cached() returns False for uncached model."""
        assert is_cached("NotCached/Model") is False


class TestDeterministicSize:
    """Deterministic tests for size() with temp cache."""

    def test_size_positive_for_cached(self, cached_model):
        """size() returns positive value for cached model."""
        result_size = size(cached_model)
        assert result_size > 0

    def test_size_zero_for_uncached(self, temp_hf_cache):
        """size() returns 0 for uncached model."""
        result_size = size("NotCached/Model")
        assert result_size == 0


class TestDeterministicDelete:
    """Deterministic tests for delete() with temp cache."""

    def test_delete_returns_true_for_cached(self, cached_model):
        """delete() returns True for cached model."""
        result = delete(cached_model)
        assert result is True
        # Verify it's gone from cache
        assert is_cached(cached_model) is False

    def test_delete_returns_false_for_uncached(self, temp_hf_cache):
        """delete() returns False for uncached model."""
        result = delete("NotCached/Model")
        assert result is False


class TestDeterministicClear:
    """Deterministic tests for clear() with temp cache."""

    def test_clear_removes_all_models(self, cached_model):
        """clear() removes all cached models."""
        # Verify model is cached
        assert is_cached(cached_model) is True

        # Clear all
        count = clear()
        assert count >= 1

        # Verify empty
        assert list(list_models()) == []

    def test_clear_returns_zero_for_empty(self, temp_hf_cache):
        """clear() returns 0 for empty cache."""
        count = clear()
        assert count == 0


# =============================================================================
# Comprehensive Deterministic Tests (with fake_hf_cache fixture)
# =============================================================================


class TestComprehensiveListModels:
    """Comprehensive tests for list_models() with fake_hf_cache."""

    def test_list_models_returns_all_cached_models(self, fake_hf_cache):
        """list_models() returns all models in fake cache."""
        cached_models = list(list_models())

        assert len(cached_models) == len(fake_hf_cache["models"])
        for model_id in fake_hf_cache["models"]:
            assert model_id in cached_models

    def test_list_models_returns_empty_for_empty_cache(self, empty_hf_cache):
        """list_models() returns empty iterator for empty cache."""
        cached_models = list(list_models())
        assert len(cached_models) == 0

    def test_all_listed_models_are_cached(self, fake_hf_cache):
        """All models from list_models() return True for is_cached()."""
        for model_id in list_models():
            assert is_cached(model_id), f"Expected {model_id} to be cached"


class TestComprehensiveGetPath:
    """Comprehensive tests for path() with fake_hf_cache."""

    def test_get_path_returns_correct_path(self, fake_hf_cache):
        """path() returns correct path for cached models."""
        for model_id in fake_hf_cache["models"]:
            path = cache_path(model_id)
            assert path is not None
            assert os.path.exists(path)
            expected = fake_hf_cache["info"][model_id]["path"]
            assert path == expected

    def test_get_path_returned_path_has_config(self, fake_hf_cache):
        """path() returns a path containing config.json."""
        for model_id in fake_hf_cache["models"]:
            path = cache_path(model_id)
            config_path = os.path.join(path, "config.json")
            assert os.path.exists(config_path), f"Expected config.json at {config_path}"


class TestComprehensiveIsCached:
    """Comprehensive tests for is_cached() with fake_hf_cache."""

    def test_is_cached_returns_true_for_cached_model(self, fake_hf_cache):
        """is_cached() returns True for models in fake cache."""
        for model_id in fake_hf_cache["models"]:
            assert is_cached(model_id) is True, f"Expected {model_id} to be cached"

    def test_is_cached_returns_false_for_uncached_model(self, fake_hf_cache):
        """is_cached() returns False for models not in fake cache."""
        assert is_cached("not/in-cache") is False


class TestComprehensiveSize:
    """Comprehensive tests for size() with fake_hf_cache."""

    def test_size_returns_int_for_cached_model(self, fake_hf_cache):
        """size() returns integer for cached models."""
        for model_id in fake_hf_cache["models"]:
            result_size = size(model_id)
            assert isinstance(result_size, int)
            assert result_size >= 0

    def test_size_returns_zero_for_uncached_model(self, fake_hf_cache):
        """size() returns 0 for uncached models."""
        result_size = size("not/in-cache")
        assert result_size == 0


class TestComprehensiveDelete:
    """Comprehensive tests for delete() with fake_hf_cache."""

    def test_delete_returns_true_for_cached_model(self, fake_hf_cache):
        """delete() returns True and removes cached model."""
        model_id = fake_hf_cache["models"][0]

        # Verify cached before
        assert is_cached(model_id) is True

        result = delete(model_id)
        assert result is True

        # Verify removed from cache after
        assert is_cached(model_id) is False
        assert cache_path(model_id) is None

    def test_delete_cleanup_integrity(self, fake_hf_cache):
        """delete() ensures no stale internal state remains.

        Contract: After delete(), the same Repository instance must report
        consistent state for the removed model. This verifies there's no
        internal path caching that would return stale data.
        """
        model_id = fake_hf_cache["models"][0]

        # Verify and cache various states before deletion
        assert is_cached(model_id) is True
        path_before = cache_path(model_id)
        assert path_before is not None
        _ = size(model_id)  # Potentially cached

        # Delete the model
        result = delete(model_id)
        assert result is True

        # Verify ALL methods return consistent "not found" state
        # on the SAME Repository instance (tests for stale caching)
        assert is_cached(model_id) is False, (
            "is_cached() returned True after delete() - possible stale cache"
        )
        assert cache_path(model_id) is None, (
            "path() returned path after delete() - possible stale cache"
        )
        assert size(model_id) == 0, (
            "size() returned non-zero after delete() - possible stale cache"
        )

        # Verify model is not in list
        cached_models = list(list_models())
        assert model_id not in cached_models, (
            "list_models() still contains deleted model - possible stale cache"
        )


class TestComprehensiveClear:
    """Comprehensive tests for clear() with fake_hf_cache."""

    def test_clear_removes_all_models_comprehensive(self, fake_hf_cache):
        """clear() removes all models from cache."""
        # Verify models are cached before clear
        initial_count = len(list(list_models()))
        assert initial_count > 0

        removed = clear()
        assert removed == initial_count

        # After clear, no models should be cached
        assert len(list(list_models())) == 0
        for model_id in fake_hf_cache["models"]:
            assert is_cached(model_id) is False


# =============================================================================
# Resource Management Tests
# =============================================================================


# =============================================================================
# Resolve Path Tests
# =============================================================================


class TestResolvePath:
    """Tests for resolve_path() operation."""

    def test_resolve_path_returns_string_for_cached(self, cached_model):
        """resolve_path() returns path for cached model."""
        path = resolve_path(cached_model, offline=True)
        assert isinstance(path, str)
        assert "TestOrg" in path or "TestModel" in path

    def test_resolve_path_error_for_nonexistent(self, temp_hf_cache):
        """resolve_path() raises IOError for non-existent model in offline mode."""
        from talu.exceptions import IOError as TaluIOError
        with pytest.raises((TaluIOError, OSError)):
            resolve_path("NotReal/Model-12345", offline=True)

    def test_resolve_path_with_token(self, cached_model):
        """resolve_path() accepts token parameter."""
        # Should work even with a fake token for cached models
        path = resolve_path(cached_model, offline=True, token="fake-token")
        assert isinstance(path, str)


# =============================================================================
# Search Tests
# =============================================================================


class TestSearchOperation:
    """Tests for search() operation.

    Note: These tests use clearly invalid queries to avoid real API calls.
    """

    def test_search_returns_iterator(self):
        """search() returns an iterator."""
        result = search("zzznonexistent999xyz")
        assert hasattr(result, "__iter__")

    def test_search_can_be_exhausted(self):
        """search() iterator can be fully consumed."""
        # This query should return empty results quickly
        results = list(search("zzznonexistent999xyz", limit=5))
        assert isinstance(results, list)
        # May be empty or contain results depending on network

    def test_search_with_limit(self):
        """search() respects limit parameter."""
        result = search("zzznonexistent999xyz", limit=3)
        results = list(result)
        assert len(results) <= 3

    def test_search_with_token(self):
        """search() accepts token parameter."""
        result = search("zzznonexistent999xyz", token="fake-token")
        # Should not crash, just return iterator
        assert hasattr(result, "__iter__")

    def test_search_with_endpoint_url(self):
        """search() accepts endpoint_url parameter."""
        result = search("zzznonexistent999xyz", endpoint_url="https://example.com")
        # Should not crash, just return iterator
        assert hasattr(result, "__iter__")


# =============================================================================
# Fetch with Callbacks Tests
# =============================================================================


class TestFetchWithCallbacks:
    """Tests for fetch() with callbacks and options.

    Note: These tests use invalid model IDs to avoid actual downloads.
    The tests verify the Python binding code paths for callback setup.
    """

    def test_fetch_with_progress_callback(self):
        """fetch() accepts on_progress callback."""
        progress_calls = []

        def on_progress(downloaded, total, filename):
            progress_calls.append((downloaded, total, filename))

        # Use invalid model - won't download anything
        result = fetch("invalid-no-slash", on_progress=on_progress)
        assert result is None
        # Callback may or may not be called for invalid models

    def test_fetch_with_endpoint_url(self):
        """fetch() accepts endpoint_url parameter."""
        result = fetch("invalid-no-slash", endpoint_url="https://example.com")
        assert result is None

    def test_fetch_accepts_token_parameter(self):
        """fetch() accepts token parameter."""
        result = fetch("invalid-no-slash", token="fake-token")
        assert result is None

    def test_fetch_with_force_no_segfault(self):
        """fetch() with force=True doesn't segfault.

        Regression test: The FetchOptions struct had a bogus file_start_callback
        field that caused memory corruption when force=True. This test ensures
        the struct now matches the Zig DownloadOptions layout.
        """
        # This previously caused a segfault due to struct mismatch
        result = fetch("invalid-no-slash", force=True)
        assert result is None

    def test_fetch_with_all_params_no_segfault(self):
        """fetch() with all parameters doesn't segfault.

        Regression test for struct layout fix.
        """
        progress_calls = []

        def on_progress(downloaded, total, filename):
            progress_calls.append((downloaded, total, filename))

        result = fetch(
            "invalid-no-slash",
            force=True,
            on_progress=on_progress,
            token="fake-token",
            endpoint_url="https://example.com",
        )
        assert result is None

    def test_fetch_force_with_valid_model_id_format(self):
        """fetch() with force=True and valid model ID format doesn't segfault.

        Tests the code path where the model ID is valid format but doesn't exist.
        """
        result = fetch("nonexistent/model-xyz-12345", force=True)
        assert result is None


# =============================================================================
# Resource Management Tests
# =============================================================================


class TestResourceCleanup:
    """Tests for C pointer resource cleanup in Repository methods.

    These tests verify that native resources (C pointers) are properly
    freed even under edge conditions and repeated operations.
    """

    def test_repeated_cache_path_calls_no_leak(self, cached_model):
        """Repeated cache_path() calls don't leak memory.

        Each call allocates a C string that must be freed.
        """
        for _ in range(100):
            path = cache_path(cached_model)
            assert path is not None

    def test_repeated_cache_path_calls_on_uncached_no_leak(self, temp_hf_cache):
        """Repeated cache_path() calls for uncached models don't leak.

        Early return path (None result) should not leak.
        """
        for _ in range(100):
            path = cache_path("NotCached/Model")
            assert path is None

    def test_repeated_list_models_no_leak(self, fake_hf_cache):
        """Repeated list_models() iterations don't leak memory.

        Each call allocates a list handle that must be freed via finally block.
        """
        for _ in range(50):
            models = list(list_models())
            assert len(models) > 0

    def test_partial_list_models_iteration_cleanup(self, fake_hf_cache):
        """Breaking out of list_models() early still cleans up resources.

        The finally block should run even if iteration is abandoned.
        """
        for _ in range(50):
            for i, _model_id in enumerate(list_models()):
                if i >= 1:
                    break  # Early exit

    def test_list_models_empty_cleanup(self, empty_hf_cache):
        """list_models() on empty cache handles cleanup correctly.

        Tests the early return path (code != 0 or no value).
        """
        for _ in range(50):
            models = list(list_models())
            assert models == []

    def test_repeated_is_cached_calls(self, cached_model):
        """Repeated is_cached() calls work correctly.

        is_cached() returns int directly, no pointer to free.
        """
        for _ in range(100):
            result = is_cached(cached_model)
            assert result is True

    def test_repeated_size_calls(self, cached_model):
        """Repeated size() calls work correctly.

        size() returns uint64 directly, no pointer to free.
        """
        for _ in range(100):
            result_size = size(cached_model)
            assert result_size > 0

    def test_cache_dir_property_repeated_access(self, temp_hf_cache):
        """cache_dir property repeated access doesn't leak.

        Each access allocates a C string that must be freed.
        """
        for _ in range(50):
            result_dir = cache_dir()
            assert result_dir is not None
            assert isinstance(result_dir, str)

    def test_list_files_cleanup_on_early_exit(self, cached_model):
        """Breaking out of list_files() early still cleans up resources."""
        for _ in range(20):
            for i, _filename in enumerate(list_files(cached_model)):
                if i >= 1:
                    break  # Early exit

    def test_list_files_empty_result_cleanup(self):
        """list_files() for non-existent model handles cleanup correctly."""
        for _ in range(20):
            files = list(list_files("NotReal/Model"))
            # May be empty or error - just ensure no crash
            assert isinstance(files, list)


class TestIteratorResourceManagement:
    """Tests for iterator resource management across multiple scenarios."""

    def test_concurrent_list_models_iterations(self, fake_hf_cache):
        """Multiple concurrent list_models() iterations don't interfere.

        Each iterator should have its own handle that gets freed independently.
        """
        # Start multiple iterators
        iter1 = list_models()
        iter2 = list_models()

        # Interleave consumption
        results1 = []
        results2 = []

        try:
            results1.append(next(iter1))
        except StopIteration:
            pass

        try:
            results2.append(next(iter2))
        except StopIteration:
            pass

        try:
            results1.append(next(iter1))
        except StopIteration:
            pass

        # Complete iteration
        results1.extend(iter1)
        results2.extend(iter2)

        # Both should have same models
        assert set(results1) == set(results2)

    def test_discarded_iterator_cleanup(self, fake_hf_cache):
        """Creating iterator without consuming cleans up correctly.

        Generator's finally block runs when generator is garbage collected.
        """
        import gc
        for _ in range(20):
            # Create iterator but don't consume it
            _ = list_models()
            # Force garbage collection
            gc.collect()

    def test_exception_during_iteration_cleanup(self, fake_hf_cache):
        """Exception raised during iteration still cleans up resources.

        The finally block should run even if consumer raises.
        """

        def consume_with_error():
            for i, model_id in enumerate(list_models()):
                if i >= 1:
                    raise ValueError("Test error")
                yield model_id

        for _ in range(10):
            try:
                list(consume_with_error())
            except ValueError:
                pass  # Expected
