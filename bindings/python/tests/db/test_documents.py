"""Tests for talu.db DocumentStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from talu.db import (
    ChangeAction,
    ChangeRecord,
    CompactionStats,
    DocumentRecord,
    DocumentSearchResult,
    DocumentStore,
    DocumentSummary,
)
from talu.exceptions import StateError


class TestDocumentStore:
    """Tests for DocumentStore class."""

    def test_init(self, tmp_path: Path) -> None:
        """DocumentStore initializes at path."""
        store = DocumentStore(str(tmp_path / "docs"))
        store.close()

    def test_context_manager(self, tmp_path: Path) -> None:
        """DocumentStore works as context manager."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            assert store is not None

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """Closing multiple times is safe."""
        store = DocumentStore(str(tmp_path / "docs"))
        store.close()
        store.close()  # Should not raise

    def test_operations_after_close_raise(self, tmp_path: Path) -> None:
        """Operations on closed store raise StateError."""
        store = DocumentStore(str(tmp_path / "docs"))
        store.close()

        with pytest.raises(StateError):
            store.list()


class TestDocumentCRUD:
    """Tests for document CRUD operations."""

    def test_create_and_get(self, tmp_path: Path) -> None:
        """Create a document and retrieve it."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-123",
                doc_type="prompt",
                title="Test Prompt",
                doc_json='{"content": "Hello world"}',
            )

            doc = store.get("doc-123")
            assert doc is not None
            assert doc.doc_id == "doc-123"
            assert doc.doc_type == "prompt"
            assert doc.title == "Test Prompt"
            assert doc.doc_json == '{"content": "Hello world"}'

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Getting a nonexistent document returns None."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            doc = store.get("nonexistent")
            assert doc is None

    def test_create_with_all_fields(self, tmp_path: Path) -> None:
        """Create a document with all optional fields."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-full",
                doc_type="prompt",
                title="Full Document",
                doc_json='{"system": "You are helpful"}',
                tags_text="coding review",
                parent_id="doc-parent",
                marker="active",
                group_id="group-1",
                owner_id="user-1",
            )

            doc = store.get("doc-full")
            assert doc is not None
            assert doc.tags_text == "coding review"
            assert doc.parent_id == "doc-parent"
            assert doc.marker == "active"
            assert doc.group_id == "group-1"
            assert doc.owner_id == "user-1"

    def test_update(self, tmp_path: Path) -> None:
        """Update an existing document."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-update",
                doc_type="prompt",
                title="Original Title",
                doc_json='{"version": 1}',
            )

            store.update(
                "doc-update",
                title="Updated Title",
                doc_json='{"version": 2}',
            )

            doc = store.get("doc-update")
            assert doc is not None
            assert doc.title == "Updated Title"
            assert doc.doc_json == '{"version": 2}'

    def test_delete(self, tmp_path: Path) -> None:
        """Delete a document."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-delete",
                doc_type="prompt",
                title="To Delete",
                doc_json="{}",
            )

            assert store.get("doc-delete") is not None

            store.delete("doc-delete")

            # After delete, get returns None
            assert store.get("doc-delete") is None

    def test_list_empty(self, tmp_path: Path) -> None:
        """List returns empty for new store."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            docs = store.list()
            assert docs == []

    def test_list_with_documents(self, tmp_path: Path) -> None:
        """List returns created documents."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-1",
                doc_type="prompt",
                title="First",
                doc_json="{}",
            )
            store.create(
                doc_id="doc-2",
                doc_type="persona",
                title="Second",
                doc_json="{}",
            )

            docs = store.list()
            assert len(docs) == 2
            assert all(isinstance(d, DocumentSummary) for d in docs)

    def test_list_filter_by_type(self, tmp_path: Path) -> None:
        """List filters by document type."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-1",
                doc_type="prompt",
                title="Prompt",
                doc_json="{}",
            )
            store.create(
                doc_id="doc-2",
                doc_type="persona",
                title="Persona",
                doc_json="{}",
            )

            prompts = store.list(doc_type="prompt")
            assert len(prompts) == 1
            assert prompts[0].doc_type == "prompt"

    def test_list_filter_by_owner(self, tmp_path: Path) -> None:
        """List filters by owner ID."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-1",
                doc_type="prompt",
                title="User 1 Doc",
                doc_json="{}",
                owner_id="user-1",
            )
            store.create(
                doc_id="doc-2",
                doc_type="prompt",
                title="User 2 Doc",
                doc_json="{}",
                owner_id="user-2",
            )

            user1_docs = store.list(owner_id="user-1")
            assert len(user1_docs) == 1
            assert user1_docs[0].doc_id == "doc-1"

    def test_list_limit(self, tmp_path: Path) -> None:
        """List respects limit parameter."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            for i in range(5):
                store.create(
                    doc_id=f"doc-{i}",
                    doc_type="prompt",
                    title=f"Document {i}",
                    doc_json="{}",
                )

            docs = store.list(limit=2)
            assert len(docs) == 2


class TestDocumentBlobStreaming:
    """Tests for externalized blob access and streaming."""

    def test_get_blob_ref_returns_none_for_inline_payload(self, tmp_path: Path) -> None:
        """Inline payloads should not expose a blob reference."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-inline",
                doc_type="prompt",
                title="Inline",
                doc_json='{"content":"small"}',
            )
            assert store.get_blob_ref("doc-inline") is None

    def test_iter_blob_chunks_reads_externalized_payload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stream reads should reconstruct single-blob externalized payloads."""
        monkeypatch.setenv("TALU_DB_DOC_JSON_EXTERNALIZE_THRESHOLD_BYTES", "32")

        payload = '{"content":"' + ("A" * 320) + '"}'
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-external",
                doc_type="prompt",
                title="Externalized",
                doc_json=payload,
            )

            blob_ref = store.get_blob_ref("doc-external")
            assert blob_ref is not None
            assert blob_ref.startswith("sha256:")

            streamed = b"".join(store.iter_blob_chunks(blob_ref, chunk_size=17))
            assert streamed == payload.encode("utf-8")

    def test_read_blob_supports_multipart_refs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multipart blob refs should stream and reassemble transparently."""
        monkeypatch.setenv("TALU_DB_DOC_JSON_EXTERNALIZE_THRESHOLD_BYTES", "32")
        monkeypatch.setenv("TALU_DB_BLOB_MULTIPART_CHUNK_SIZE_BYTES", "48")

        payload = '{"content":"' + ("multipart-" * 80) + '"}'
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-multipart",
                doc_type="prompt",
                title="Multipart",
                doc_json=payload,
            )

            blob_ref = store.get_blob_ref("doc-multipart")
            assert blob_ref is not None
            assert blob_ref.startswith("multi:")

            loaded = store.read_blob(blob_ref, chunk_size=13)
            assert loaded == payload.encode("utf-8")


class TestDocumentSearch:
    """Tests for document search operations."""

    def test_search_empty(self, tmp_path: Path) -> None:
        """Search on empty store returns empty."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            results = store.search("hello")
            assert results == []

    def test_search_finds_match(self, tmp_path: Path) -> None:
        """Search finds matching documents."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-1",
                doc_type="prompt",
                title="Python Tutorial",
                doc_json='{"content": "Learn Python programming"}',
            )
            store.create(
                doc_id="doc-2",
                doc_type="prompt",
                title="Rust Guide",
                doc_json='{"content": "Learn Rust programming"}',
            )

            results = store.search("Python")
            assert len(results) >= 1
            assert all(isinstance(r, DocumentSearchResult) for r in results)
            assert any(r.doc_id == "doc-1" for r in results)

    def test_search_filter_by_type(self, tmp_path: Path) -> None:
        """Search filters by document type."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-1",
                doc_type="prompt",
                title="Programming Guide",
                doc_json='{"topic": "coding"}',
            )
            store.create(
                doc_id="doc-2",
                doc_type="persona",
                title="Coding Assistant",
                doc_json='{"role": "coder"}',
            )

            results = store.search("coding", doc_type="prompt")
            # Should only return prompts
            for r in results:
                assert r.doc_type == "prompt"

    def test_search_batch(self, tmp_path: Path) -> None:
        """Batch search multiple queries."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-python",
                doc_type="prompt",
                title="Python Tutorial",
                doc_json='{"content": "Learn Python programming"}',
            )
            store.create(
                doc_id="doc-rust",
                doc_type="prompt",
                title="Rust Guide",
                doc_json='{"content": "Learn Rust programming"}',
            )

            queries = [
                {"id": "q1", "text": "Python"},
                {"id": "q2", "text": "Rust"},
            ]
            results = store.search_batch(queries)

            assert isinstance(results, dict)
            assert "q1" in results
            assert "q2" in results
            assert isinstance(results["q1"], list)
            assert isinstance(results["q2"], list)

    def test_search_batch_empty(self, tmp_path: Path) -> None:
        """Batch search on empty store returns empty results."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            queries = [{"id": "q1", "text": "nonexistent"}]
            results = store.search_batch(queries)

            assert isinstance(results, dict)
            assert "q1" in results
            assert results["q1"] == []

    def test_search_batch_with_type_filter(self, tmp_path: Path) -> None:
        """Batch search with document type filter."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-1",
                doc_type="prompt",
                title="Programming Prompt",
                doc_json='{"topic": "coding"}',
            )
            store.create(
                doc_id="doc-2",
                doc_type="persona",
                title="Programming Persona",
                doc_json='{"role": "coder"}',
            )

            queries = [
                {"id": "q1", "text": "Programming", "type": "prompt"},
            ]
            results = store.search_batch(queries)

            assert isinstance(results, dict)
            # Results should only include prompts
            if results["q1"]:
                assert "doc-1" in results["q1"]


class TestDocumentTags:
    """Tests for document tag operations."""

    def test_add_and_get_tags(self, tmp_path: Path) -> None:
        """Add tags to a document and retrieve them."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-tags",
                doc_type="prompt",
                title="Tagged Doc",
                doc_json="{}",
            )

            store.add_tag("doc-tags", "tag-1")
            store.add_tag("doc-tags", "tag-2")

            tags = store.get_tags("doc-tags")
            assert "tag-1" in tags
            assert "tag-2" in tags

    def test_remove_tag(self, tmp_path: Path) -> None:
        """Remove a tag from a document."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-tags",
                doc_type="prompt",
                title="Tagged Doc",
                doc_json="{}",
            )

            store.add_tag("doc-tags", "tag-1")
            store.add_tag("doc-tags", "tag-2")

            store.remove_tag("doc-tags", "tag-1")

            tags = store.get_tags("doc-tags")
            assert "tag-1" not in tags
            assert "tag-2" in tags

    def test_get_by_tag(self, tmp_path: Path) -> None:
        """Get documents by tag."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-1",
                doc_type="prompt",
                title="Doc 1",
                doc_json="{}",
            )
            store.create(
                doc_id="doc-2",
                doc_type="prompt",
                title="Doc 2",
                doc_json="{}",
            )

            store.add_tag("doc-1", "shared-tag")
            store.add_tag("doc-2", "shared-tag")

            doc_ids = store.get_by_tag("shared-tag")
            assert "doc-1" in doc_ids
            assert "doc-2" in doc_ids


class TestDocumentTTL:
    """Tests for document TTL operations."""

    def test_set_ttl(self, tmp_path: Path) -> None:
        """Set TTL for a document."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-ttl",
                doc_type="prompt",
                title="Temporary Doc",
                doc_json="{}",
            )

            # Set 1 hour TTL - verify operation completes without error
            store.set_ttl("doc-ttl", 3600)

            # Verify document still exists after setting TTL
            doc = store.get("doc-ttl")
            assert doc is not None

    def test_remove_ttl(self, tmp_path: Path) -> None:
        """Remove TTL from a document."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-ttl",
                doc_type="prompt",
                title="Temporary Doc",
                doc_json="{}",
            )

            store.set_ttl("doc-ttl", 3600)
            store.set_ttl("doc-ttl", 0)  # Remove TTL

            doc = store.get("doc-ttl")
            assert doc is not None
            assert doc.expires_at_ms == 0

    def test_count_expired(self, tmp_path: Path) -> None:
        """Count expired documents."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            # Initially no expired docs
            count = store.count_expired()
            assert count >= 0


class TestDocumentCDC:
    """Tests for document CDC operations."""

    def test_get_changes_empty(self, tmp_path: Path) -> None:
        """Get changes on empty store."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            changes = store.get_changes()
            assert isinstance(changes, list)

    def test_get_changes_after_create(self, tmp_path: Path) -> None:
        """Get changes after creating documents."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-cdc",
                doc_type="prompt",
                title="CDC Test",
                doc_json="{}",
            )

            # CDC may return changes depending on backend configuration
            changes = store.get_changes(since_seq=0)
            assert isinstance(changes, list)
            assert all(isinstance(c, ChangeRecord) for c in changes)

            # If changes are returned, verify structure
            if len(changes) > 0:
                create_change = next((c for c in changes if c.doc_id == "doc-cdc"), None)
                if create_change:
                    assert create_change.action == ChangeAction.CREATE
                    assert create_change.seq_num > 0

    def test_change_action_enum(self) -> None:
        """ChangeAction enum has expected values."""
        assert ChangeAction.CREATE == 1
        assert ChangeAction.UPDATE == 2
        assert ChangeAction.DELETE == 3


class TestDocumentDelta:
    """Tests for document delta versioning."""

    def test_create_delta(self, tmp_path: Path) -> None:
        """Create a delta version."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            # Create base document
            store.create(
                doc_id="doc-base",
                doc_type="prompt",
                title="Base Document",
                doc_json='{"version": 1}',
            )

            # Create delta
            store.create_delta(
                base_doc_id="doc-base",
                new_doc_id="doc-delta",
                delta_json='[{"op": "replace", "path": "/version", "value": 2}]',
            )

            # Verify delta exists
            assert store.is_delta("doc-delta")
            assert not store.is_delta("doc-base")

    def test_get_base_id(self, tmp_path: Path) -> None:
        """Get base ID for a delta document."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-base",
                doc_type="prompt",
                title="Base",
                doc_json="{}",
            )

            store.create_delta(
                base_doc_id="doc-base",
                new_doc_id="doc-delta",
                delta_json="[]",
            )

            base_id = store.get_base_id("doc-delta")
            assert base_id == "doc-base"

            # Non-delta returns None
            base_id = store.get_base_id("doc-base")
            assert base_id is None

    def test_get_delta_chain(self, tmp_path: Path) -> None:
        """Get delta chain for a document."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            # Create base document
            store.create(
                doc_id="doc-v1",
                doc_type="prompt",
                title="Version 1",
                doc_json='{"version": 1}',
            )

            # Create delta chain: v1 -> v2 -> v3
            store.create_delta(
                base_doc_id="doc-v1",
                new_doc_id="doc-v2",
                delta_json='{"version": 2}',
                title="Version 2",
            )
            store.create_delta(
                base_doc_id="doc-v2",
                new_doc_id="doc-v3",
                delta_json='{"version": 3}',
                title="Version 3",
            )

            # Get chain from v3
            chain = store.get_delta_chain("doc-v3")
            assert isinstance(chain, list)
            assert len(chain) == 3

            # First element is the requested doc, last is the base
            assert chain[0].doc_id == "doc-v3"
            assert chain[-1].doc_id == "doc-v1"

            # All are DocumentRecord instances
            assert all(isinstance(doc, DocumentRecord) for doc in chain)

    def test_get_delta_chain_single(self, tmp_path: Path) -> None:
        """Get delta chain for a non-delta document."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-single",
                doc_type="prompt",
                title="Single Doc",
                doc_json="{}",
            )

            chain = store.get_delta_chain("doc-single")
            assert isinstance(chain, list)
            assert len(chain) == 1
            assert chain[0].doc_id == "doc-single"


class TestDocumentCompaction:
    """Tests for document compaction operations."""

    def test_get_compaction_stats(self, tmp_path: Path) -> None:
        """Get compaction statistics."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            store.create(
                doc_id="doc-1",
                doc_type="prompt",
                title="Doc 1",
                doc_json="{}",
            )

            stats = store.get_compaction_stats()
            assert isinstance(stats, CompactionStats)
            assert stats.total_documents >= 1
            assert stats.active_documents >= 1

    def test_get_garbage_candidates(self, tmp_path: Path) -> None:
        """Get garbage collection candidates."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            candidates = store.get_garbage_candidates()
            assert isinstance(candidates, list)

    def test_purge_expired(self, tmp_path: Path) -> None:
        """Purge expired documents."""
        with DocumentStore(str(tmp_path / "docs")) as store:
            purged = store.purge_expired()
            assert purged >= 0


class TestDataClasses:
    """Tests for data class structures."""

    def test_document_record_fields(self) -> None:
        """DocumentRecord has expected fields."""
        doc = DocumentRecord(
            doc_id="test",
            doc_type="prompt",
            title="Test",
            doc_json="{}",
        )
        assert doc.doc_id == "test"
        assert doc.doc_type == "prompt"
        assert doc.title == "Test"
        assert doc.doc_json == "{}"
        assert doc.tags_text is None
        assert doc.parent_id is None
        assert doc.marker is None
        assert doc.group_id is None
        assert doc.owner_id is None

    def test_document_summary_fields(self) -> None:
        """DocumentSummary has expected fields."""
        summary = DocumentSummary(
            doc_id="test",
            doc_type="prompt",
            title="Test",
        )
        assert summary.doc_id == "test"
        assert summary.doc_type == "prompt"
        assert summary.title == "Test"
        assert summary.marker is None

    def test_search_result_fields(self) -> None:
        """DocumentSearchResult has expected fields."""
        result = DocumentSearchResult(
            doc_id="test",
            doc_type="prompt",
            title="Test",
            snippet="...match...",
        )
        assert result.doc_id == "test"
        assert result.snippet == "...match..."

    def test_change_record_fields(self) -> None:
        """ChangeRecord has expected fields."""
        change = ChangeRecord(
            seq_num=123,
            doc_id="test",
            action=ChangeAction.CREATE,
            timestamp_ms=1000,
        )
        assert change.seq_num == 123
        assert change.doc_id == "test"
        assert change.action == ChangeAction.CREATE

    def test_compaction_stats_fields(self) -> None:
        """CompactionStats has expected fields."""
        stats = CompactionStats(
            total_documents=100,
            active_documents=90,
            expired_documents=5,
            deleted_documents=5,
            tombstone_count=10,
            delta_versions=20,
            estimated_garbage_bytes=1024,
        )
        assert stats.total_documents == 100
        assert stats.active_documents == 90
        assert stats.estimated_garbage_bytes == 1024
