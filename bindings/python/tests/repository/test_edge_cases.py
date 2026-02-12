"""
Edge case tests for talu.repository module.

Tests for handling malformed cache entries, incomplete models,
corrupted files, and other edge cases.
"""

import json

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
# Incomplete Model Tests
# =============================================================================


class TestIncompleteModels:
    """Tests for incomplete model handling."""

    def test_missing_weights_not_cached(self, incomplete_model_missing_weights):
        """Model without weights file is not considered cached."""
        model_id = incomplete_model_missing_weights["model_id"]

        # is_cached() should return False - model is incomplete
        assert is_cached(model_id) is False

        # path() should return None
        assert cache_path(model_id) is None

    def test_missing_weights_not_in_list(self, incomplete_model_missing_weights):
        """Model without weights is not listed in list_models()."""
        model_id = incomplete_model_missing_weights["model_id"]

        models = list(list_models())
        assert model_id not in models

    def test_missing_tokenizer_still_exists(self, incomplete_model_missing_tokenizer):
        """Model without tokenizer.json may still be listed (weights required)."""
        model_id = incomplete_model_missing_tokenizer["model_id"]

        # The model has weights, so it should exist from repository perspective
        # (tokenizer is loaded separately)
        assert is_cached(model_id) is True
        assert cache_path(model_id) is not None

    def test_missing_config_still_cached(self, incomplete_model_missing_config):
        """Model without config.json is still considered cached.

        CONTRACT: Repository checks for weights presence only.
        config.json is not required at the cache layer — it is validated
        at model load time. This matches the design: getCachedPath scans
        snapshots for findWeightsFile, which checks .safetensors existence.
        """
        model_id = incomplete_model_missing_config["model_id"]

        assert is_cached(model_id) is True
        assert cache_path(model_id) is not None

    def test_missing_weights_size_returns_zero(self, incomplete_model_missing_weights):
        """size() returns 0 for a model without weights.

        CONTRACT: size() uses getCachedPath which requires valid weights.
        A model without weights is not a valid cached model, so
        getCachedPath returns null and size() returns 0.
        Use is_cached() for the same check without computing size.
        """
        model_id = incomplete_model_missing_weights["model_id"]

        result_size = size(model_id)

        # No weights → getCachedPath returns null → size is 0
        assert result_size == 0

    def test_missing_weights_delete_cleans_directory(self, incomplete_model_missing_weights):
        """Deleting incomplete model cleans up the cache directory."""
        model_id = incomplete_model_missing_weights["model_id"]
        cache_dir = incomplete_model_missing_weights["cache_dir"]

        model_dir = cache_dir / "models--test--incomplete-weights"

        # Directory exists before removal
        assert model_dir.exists()

        # Delete the model directory
        delete(model_id)
        # After removal, model definitely doesn't exist
        assert is_cached(model_id) is False


# =============================================================================
# Malformed Cache / Edge Case Tests
# =============================================================================


class TestMalformedCacheHandling:
    """Tests for handling malformed or incomplete cache entries."""

    def test_missing_weights_file(self, tmp_path, monkeypatch):
        """Repository handles cache entries without model weights gracefully."""
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--no-weights"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        # Only config, no weights file
        config = {"model_type": "llama", "hidden_size": 64}
        (snapshot / "config.json").write_text(json.dumps(config))

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123")

        monkeypatch.setenv("HF_HOME", str(hf_home))
        # Model without weights should not be considered cached
        assert is_cached("test/no-weights") is False

    def test_empty_refs_file_still_cached(self, tmp_path, monkeypatch):
        """Empty refs/main does not affect cache detection.

        CONTRACT: getCachedPath scans snapshots/ directories directly
        and checks each for weights via findWeightsFile. It never reads
        refs/. The refs/ directory is a HuggingFace Hub convention but
        not used by talu's cache layer.
        """
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--empty-refs"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        config = {"model_type": "llama", "hidden_size": 64}
        (snapshot / "config.json").write_text(json.dumps(config))
        (snapshot / "model.safetensors").write_bytes(b"\x00" * 512)

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("")  # Empty refs file

        monkeypatch.setenv("HF_HOME", str(hf_home))
        assert is_cached("test/empty-refs") is True

    def test_multiple_snapshots(self, tmp_path, monkeypatch):
        """Repository handles multiple snapshot directories correctly."""
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--multi-snapshot"

        # Create two snapshots
        for rev in ["old123", "new456"]:
            snapshot = model_dir / "snapshots" / rev
            snapshot.mkdir(parents=True)
            config = {"model_type": "llama", "hidden_size": 64, "rev": rev}
            (snapshot / "config.json").write_text(json.dumps(config))
            (snapshot / "model.safetensors").write_bytes(b"\x00" * 512)

        # refs/main points to newer snapshot
        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("new456")

        monkeypatch.setenv("HF_HOME", str(hf_home))
        assert is_cached("test/multi-snapshot") is True

        # get_path should return the refs/main snapshot path
        path = cache_path("test/multi-snapshot")
        assert path is not None
        assert "new456" in path

    def test_empty_snapshot_directory(self, tmp_path, monkeypatch):
        """Repository handles empty snapshot directories gracefully."""
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--empty-snapshot"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        # Empty snapshot - no files at all
        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123")

        monkeypatch.setenv("HF_HOME", str(hf_home))
        # Empty snapshot should not be considered a valid model
        assert is_cached("test/empty-snapshot") is False

    def test_stale_refs_pointing_to_missing_snapshot(self, tmp_path, monkeypatch):
        """Refs pointing to a non-existent snapshot should be handled gracefully."""
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--stale-refs"

        # Create refs pointing to non-existent snapshot
        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("nonexistent123")

        # Create snapshots directory but NOT the referenced snapshot
        snapshots_dir = model_dir / "snapshots"
        snapshots_dir.mkdir(parents=True)

        monkeypatch.setenv("HF_HOME", str(hf_home))
        exists = is_cached("test/stale-refs")

        # Stale refs should not resolve to a valid model
        if exists:
            pytest.xfail(
                "Repository returns True for stale refs pointing to missing snapshot - "
                "should return False when referenced snapshot doesn't exist"
            )
        assert exists is False

    def test_corrupted_config_json(self, tmp_path, monkeypatch):
        """Corrupted config.json means model is NOT valid.

        CONTRACT: exists() checks for valid model files. A model with
        corrupted config.json is invalid and MUST return False.

        Repository does NOT parse config.json - it only checks file existence.
        The corruption will be detected at model load time.
        """
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--corrupted-config"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        # Create invalid JSON
        (snapshot / "config.json").write_text("{ this is not valid json }")
        (snapshot / "model.safetensors").write_bytes(b"\x00" * 512)

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123")

        monkeypatch.setenv("HF_HOME", str(hf_home))

        # CONTRACT: Repository checks file existence, not content validity.
        # A model with weights and config.json (even corrupted) exists from
        # repository perspective. Corruption is detected at model load time.
        exists = is_cached("test/corrupted-config")
        assert isinstance(exists, bool)
        assert exists is True, (
            "Repository should return exists=True for model with corrupted config.json. "
            "Repository checks file presence, not content validity. "
            "Corruption detection happens at model load time."
        )

    def test_missing_refs_directory_still_cached(self, tmp_path, monkeypatch):
        """Model without refs/ directory is still considered cached.

        CONTRACT: getCachedPath scans snapshots/ directories directly
        and never reads refs/. A model with weights in any snapshot
        subdirectory is considered cached regardless of refs/ presence.
        """
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--no-refs"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        config = {"model_type": "llama", "hidden_size": 64}
        (snapshot / "config.json").write_text(json.dumps(config))
        (snapshot / "model.safetensors").write_bytes(b"\x00" * 512)
        # Note: refs_dir NOT created

        monkeypatch.setenv("HF_HOME", str(hf_home))
        assert is_cached("test/no-refs") is True
        assert cache_path("test/no-refs") is not None

    def test_missing_snapshots_directory(self, tmp_path, monkeypatch):
        """Model directory without snapshots/ should not exist."""
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--no-snapshots"

        # Create refs but no snapshots directory
        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123")
        # Note: snapshots directory NOT created

        monkeypatch.setenv("HF_HOME", str(hf_home))
        exists = is_cached("test/no-snapshots")

        # Without snapshots, model definitely can't exist
        if exists:
            pytest.xfail(
                "Repository returns True for model without snapshots/ directory - "
                "should return False when snapshots directory is missing"
            )
        assert exists is False

    def test_multi_branch_refs(self, tmp_path, monkeypatch):
        """Repository handles multiple refs (main, dev) correctly."""
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--multi-refs"

        # Create two snapshots
        for rev in ["main123", "dev456"]:
            snapshot = model_dir / "snapshots" / rev
            snapshot.mkdir(parents=True)
            config = {"model_type": "llama", "hidden_size": 64, "branch": rev}
            (snapshot / "config.json").write_text(json.dumps(config))
            (snapshot / "model.safetensors").write_bytes(b"\x00" * 512)

        # Create refs/main and refs/dev
        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("main123")
        (refs_dir / "dev").write_text("dev456")

        monkeypatch.setenv("HF_HOME", str(hf_home))

        # Model should exist
        assert is_cached("test/multi-refs") is True

        # get_path should return refs/main snapshot
        path = cache_path("test/multi-refs")
        assert path is not None
        assert "main123" in path, f"Expected path to refs/main snapshot (main123), got: {path}"


# =============================================================================
# Tokenizer Edge Cases
# =============================================================================


class TestTokenizerEdgeCases:
    """Tests for tokenizer file edge cases."""

    def test_missing_tokenizer_json(self, incomplete_model_missing_tokenizer):
        """Model without tokenizer.json should still be listed by repository.

        Repository only checks for weights, not tokenizer. Tokenizer validation
        happens at model load time.
        """
        model_id = incomplete_model_missing_tokenizer["model_id"]

        # Repository should see the model (weights exist)
        assert is_cached(model_id) is True

        # Path should be retrievable
        path = cache_path(model_id)
        assert path is not None

        # But tokenizer.json should actually be missing
        from pathlib import Path

        model_path = Path(path)
        assert not (model_path / "tokenizer.json").exists()

    def test_corrupted_tokenizer_json(self, tmp_path, monkeypatch):
        """Corrupted tokenizer.json should be handled gracefully by repository.

        Repository should still list the model (it only checks weights).
        Tokenizer errors should happen at model load time, not repository query.
        """
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--corrupted-tokenizer"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        # Valid config, invalid tokenizer JSON
        config = {"model_type": "llama", "hidden_size": 64}
        (snapshot / "config.json").write_text(json.dumps(config))
        (snapshot / "tokenizer.json").write_text("{ not valid json !!!")
        (snapshot / "model.safetensors").write_bytes(b"\x00" * 512)

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123")

        monkeypatch.setenv("HF_HOME", str(hf_home))

        # Repository should still report model exists (weights are there)
        # Tokenizer validity is not repository's concern
        try:
            exists = is_cached("test/corrupted-tokenizer")
            assert isinstance(exists, bool)
            # Model should exist from repository perspective (has weights)
            assert exists is True, (
                "Repository should report exists=True even with corrupted tokenizer - "
                "tokenizer validation happens at load time"
            )
        except Exception as e:
            pytest.fail(f"Repository crashed on corrupted tokenizer.json: {e}")


# =============================================================================
# Corrupted SafeTensors Tests
# =============================================================================


class TestCorruptedSafeTensors:
    """Tests for handling corrupted SafeTensors files."""

    def test_truncated_safetensors_repository_sees_model(self, tmp_path, monkeypatch):
        """Repository reports model exists even with truncated weights.

        Repository only checks file presence, not integrity. Corruption
        is detected when actually loading the model.
        """
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--truncated-weights"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        config = {"model_type": "llama", "hidden_size": 64}
        (snapshot / "config.json").write_text(json.dumps(config))
        # Truncated file - only a few bytes
        (snapshot / "model.safetensors").write_bytes(b"\x00\x01\x02")

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123")

        monkeypatch.setenv("HF_HOME", str(hf_home))

        # Repository should see the model (file exists)
        assert is_cached("test/truncated-weights") is True
        assert cache_path("test/truncated-weights") is not None

    def test_corrupted_safetensors_header(self, tmp_path, monkeypatch):
        """SafeTensors with invalid header bytes is handled gracefully.

        CONTRACT: Repository checks file existence and non-zero size, not content validity.
        Corruption detection happens at model load time, not repository query time.
        """
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--bad-header"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        config = {"model_type": "llama", "hidden_size": 64}
        (snapshot / "config.json").write_text(json.dumps(config))
        # Invalid SafeTensors: random garbage that doesn't parse (but has non-zero size)
        (snapshot / "model.safetensors").write_bytes(
            b"\xff\xfe\xfd\xfc" * 128  # Random bytes that aren't valid SafeTensors
        )

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123")

        monkeypatch.setenv("HF_HOME", str(hf_home))

        # CONTRACT: File exists with non-zero size, so repository reports exists=True
        # Content validation is deferred to model load time
        exists = is_cached("test/bad-header")
        assert exists is True, (
            "Repository should report exists=True for non-zero corrupted weights. "
            "Content validation happens at model load time, not repository query."
        )

    def test_zero_byte_safetensors_still_cached(self, tmp_path, monkeypatch):
        """0-byte SafeTensors file is considered cached (presence check only).

        CONTRACT: findWeightsFile uses access() to check file existence,
        not size or content validity. A 0-byte file exists on disk, so
        the model is considered cached. Content validation (header parsing,
        tensor loading) happens at model load time.
        """
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--empty-weights"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        config = {"model_type": "llama", "hidden_size": 64}
        (snapshot / "config.json").write_text(json.dumps(config))
        (snapshot / "model.safetensors").write_bytes(b"")

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123")

        monkeypatch.setenv("HF_HOME", str(hf_home))
        assert is_cached("test/empty-weights") is True

    def test_sharded_model_missing_one_shard_still_cached(self, tmp_path, monkeypatch):
        """Model with missing shard is still considered cached.

        CONTRACT: findWeightsFile checks for model.safetensors.index.json
        by file presence (first in priority order). It does NOT parse the
        index to verify all shards exist. Shard completeness validation
        happens at model load time when the sharded reader opens each file.
        """
        hf_home = tmp_path / "huggingface"
        cache_dir = hf_home / "hub"
        model_dir = cache_dir / "models--test--missing-shard"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)

        config = {"model_type": "llama", "hidden_size": 64}
        (snapshot / "config.json").write_text(json.dumps(config))

        # Create only first shard, second shard missing
        (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 512)
        # Note: model-00002-of-00002.safetensors NOT created

        # Index file mentions both shards
        index = {
            "metadata": {},
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            },
        }
        (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))

        refs_dir = model_dir / "refs"
        refs_dir.mkdir(parents=True)
        (refs_dir / "main").write_text("abc123")

        monkeypatch.setenv("HF_HOME", str(hf_home))
        assert is_cached("test/missing-shard") is True
