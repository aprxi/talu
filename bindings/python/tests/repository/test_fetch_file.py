"""Tests for Repository.fetch_file()."""

import json

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


class TestFetchFile:
    """Tests for single-file fetch."""

    def test_invalid_model_id_returns_none(self):
        """Invalid model ID (no slash) returns None."""
        path = fetch_file("invalid-no-slash", "config.json")
        assert path is None

    def test_cached_file_returned(self, tmp_path, monkeypatch):
        """Returns cached file path when file already exists."""
        hf_home = tmp_path / "huggingface"
        monkeypatch.setenv("HF_HOME", str(hf_home))

        # Create fake cache with snapshots/main/config.json
        model_dir = hf_home / "hub" / "models--test--model-a"
        snapshot_dir = model_dir / "snapshots" / "main"
        snapshot_dir.mkdir(parents=True)

        config = {"hidden_size": 64, "num_hidden_layers": 2}
        config_path = snapshot_dir / "config.json"
        config_path.write_text(json.dumps(config))
        result = fetch_file("test/model-a", "config.json")
        assert result is not None
        assert result.endswith("config.json")

        with open(result) as f:
            loaded = json.load(f)
        assert loaded["hidden_size"] == 64

    def test_cached_file_idempotent(self, tmp_path, monkeypatch):
        """Fetching the same file twice returns the same path."""
        hf_home = tmp_path / "huggingface"
        monkeypatch.setenv("HF_HOME", str(hf_home))

        model_dir = hf_home / "hub" / "models--test--model-a"
        snapshot_dir = model_dir / "snapshots" / "main"
        snapshot_dir.mkdir(parents=True)

        config = {"hidden_size": 64}
        (snapshot_dir / "config.json").write_text(json.dumps(config))
        path1 = fetch_file("test/model-a", "config.json")
        path2 = fetch_file("test/model-a", "config.json")
        assert path1 == path2

    def test_different_filenames(self, tmp_path, monkeypatch):
        """Can fetch different files from the same model."""
        hf_home = tmp_path / "huggingface"
        monkeypatch.setenv("HF_HOME", str(hf_home))

        model_dir = hf_home / "hub" / "models--test--model-a"
        snapshot_dir = model_dir / "snapshots" / "main"
        snapshot_dir.mkdir(parents=True)

        (snapshot_dir / "config.json").write_text('{"a": 1}')
        (snapshot_dir / "tokenizer.json").write_text('{"b": 2}')
        p1 = fetch_file("test/model-a", "config.json")
        p2 = fetch_file("test/model-a", "tokenizer.json")
        assert p1 is not None
        assert p2 is not None
        assert p1 != p2
        assert p1.endswith("config.json")
        assert p2.endswith("tokenizer.json")
