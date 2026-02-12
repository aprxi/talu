"""Tests for talu.db module."""

from __future__ import annotations

from array import array
from pathlib import Path

import pytest

from talu.db import Database, VectorStore
from talu.exceptions import ValidationError


class TestDatabase:
    """Tests for Database class."""

    def test_default_construction(self) -> None:
        """Database defaults to :memory:."""
        db = Database()
        assert db.location == ":memory:"

    def test_explicit_memory(self) -> None:
        """Database accepts explicit :memory:."""
        db = Database(":memory:")
        assert db.location == ":memory:"

    def test_taludb_location(self, tmp_path: Path) -> None:
        """Database accepts talu:// locations."""
        location = f"talu://{tmp_path}/db"
        db = Database(location)
        assert db.location == location

    def test_empty_location_raises(self) -> None:
        """Database rejects empty talu:// locations."""
        with pytest.raises(ValidationError):
            Database("talu://")

    def test_unsupported_location_raises(self) -> None:
        """Database rejects unsupported locations."""
        with pytest.raises(ValidationError, match="not yet supported"):
            Database("sqlite:test.db")

    def test_repr(self) -> None:
        """Database has informative repr."""
        db = Database()
        assert "Database" in repr(db)
        assert ":memory:" in repr(db)


class TestVectorStore:
    """Tests for VectorStore class."""

    def test_init(self, tmp_path: Path) -> None:
        """VectorStore initializes at path."""
        store = VectorStore(str(tmp_path / "vectors"))
        store.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """VectorStore works as context manager."""
        with VectorStore(str(tmp_path / "vectors")) as store:
            assert store is not None

    def test_append_and_load(self, tmp_path: Path) -> None:
        """Append vectors and load them back."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            ids = array("Q", [1, 2, 3])
            vectors = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

            store.append_batch(ids, vectors, dims=3)

            loaded = store.load()
            assert loaded.count == 3
            assert loaded.dims == 3
            assert len(loaded.ids) == 3
            assert len(loaded.vectors) == 9
        finally:
            store.close()

    def test_search(self, tmp_path: Path) -> None:
        """Search returns top-k similar vectors."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            ids = array("Q", [1, 2, 3])
            vectors = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            store.append_batch(ids, vectors, dims=3)

            query = array("f", [1.0, 0.0, 0.0])
            found_ids, scores = store.search(query, k=2)

            assert len(found_ids) <= 2
            assert 1 in found_ids  # Should find closest match (ID 1 has [1,0,0])
        finally:
            store.close()

    def test_search_batch(self, tmp_path: Path) -> None:
        """Batch search returns results for multiple queries."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            ids = array("Q", [1, 2, 3])
            vectors = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            store.append_batch(ids, vectors, dims=3)

            # Two queries: [1,0,0] and [0,1,0]
            queries = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            found_ids, scores, count_per_query = store.search_batch(
                queries, dims=3, query_count=2, k=2
            )

            assert count_per_query <= 2
            assert len(found_ids) == count_per_query * 2
        finally:
            store.close()

    def test_scan(self, tmp_path: Path) -> None:
        """Scan yields all vectors with scores."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            ids = array("Q", [1, 2, 3])
            vectors = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            store.append_batch(ids, vectors, dims=3)

            query = array("f", [1.0, 0.0, 0.0])
            results = list(store.scan(query))

            assert len(results) == 3
            # Check that we got (id, score) tuples
            for vec_id, score in results:
                assert isinstance(vec_id, int)
                assert isinstance(score, float)
        finally:
            store.close()

    def test_scan_batch(self, tmp_path: Path) -> None:
        """Batch scan returns all scores for multiple queries."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            ids = array("Q", [1, 2, 3])
            vectors = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            store.append_batch(ids, vectors, dims=3)

            queries = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            found_ids, scores, total_rows = store.scan_batch(queries, dims=3, query_count=2)

            assert total_rows == 3
            assert len(found_ids) == 3
            assert len(scores) == 6  # 3 rows * 2 queries
        finally:
            store.close()

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """Closing multiple times is safe."""
        store = VectorStore(str(tmp_path / "vectors"))
        store.close()
        store.close()  # Should not raise

    def test_operations_after_close_raise(self, tmp_path: Path) -> None:
        """Operations on closed store raise StateError."""
        from talu.exceptions import StateError

        store = VectorStore(str(tmp_path / "vectors"))
        store.close()

        with pytest.raises(StateError):
            store.load()

    def test_empty_append(self, tmp_path: Path) -> None:
        """Empty append is a no-op."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            ids: array[int] = array("Q")
            vectors: array[float] = array("f")
            store.append_batch(ids, vectors, dims=3)

            loaded = store.load()
            assert loaded.count == 0
        finally:
            store.close()

    def test_set_durability_async_os(self, tmp_path: Path) -> None:
        """Setting async_os durability does not raise."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            store.set_durability("async_os")
        finally:
            store.close()

    def test_set_durability_full(self, tmp_path: Path) -> None:
        """Setting full durability does not raise."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            store.set_durability("full")
        finally:
            store.close()

    def test_set_durability_invalid_raises(self, tmp_path: Path) -> None:
        """Invalid durability mode raises ValueError."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            with pytest.raises(ValueError, match="invalid durability mode"):
                store.set_durability("bogus")
        finally:
            store.close()

    def test_invalid_vector_length_raises(self, tmp_path: Path) -> None:
        """Mismatched vector length raises ValidationError."""
        store = VectorStore(str(tmp_path / "vectors"))
        try:
            ids = array("Q", [1, 2])
            # 2 ids but only 5 floats (should be 6 for dims=3)
            vectors = array("f", [1.0, 0.0, 0.0, 0.0, 1.0])

            with pytest.raises(ValidationError, match="does not match"):
                store.append_batch(ids, vectors, dims=3)
        finally:
            store.close()
