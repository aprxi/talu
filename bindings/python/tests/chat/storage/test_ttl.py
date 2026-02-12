from talu._native import CStorageRecord
from talu.chat._bindings import c_storage_record_to_item_record


class TestTtlStorage:
    """Verify ttl_ts storage propagation and defaults."""

    def test_cstorage_record_maps_ttl_ts(self):
        record = CStorageRecord()
        record.item_id = 1
        record.session_id = None
        record.item_type = 0
        record.role = 0
        record.status = 2
        record.hidden = False
        record.pinned = False
        record.parent_item_id = 0
        record.has_parent = False
        record.origin_item_id = 0
        record.has_origin = False
        record.origin_session_id = None
        record.finish_reason = None
        record.prefill_ns = 0
        record.generation_ns = 0
        record.input_tokens = 0
        record.output_tokens = 0
        record.ttl_ts = 42
        record.created_at_ms = 0
        record.content_json = b"{}"
        record.metadata_json = None

        item = c_storage_record_to_item_record(record)
        assert item["ttl_ts"] == 42
