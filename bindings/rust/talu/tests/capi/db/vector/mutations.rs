//! Vector mutation/state integration tests.

use crate::capi::db::common::TestContext;
use talu::vector::{ChangeOp, VectorStore};

#[test]
fn upsert_delete_fetch_roundtrip() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 3;
    store
        .upsert(&[1, 2], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dims)
        .expect("upsert failed");

    let fetched = store.fetch(&[1, 2, 3], true).expect("fetch failed");
    assert_eq!(fetched.dims, dims);
    assert_eq!(fetched.ids, vec![1, 2]);
    assert_eq!(fetched.missing_ids, vec![3]);
    let values = fetched.vectors.expect("values present");
    assert_eq!(values.len(), 6);

    let deleted = store.delete(&[2, 99]).expect("delete failed");
    assert_eq!(deleted.deleted_count, 1);
    assert_eq!(deleted.not_found_count, 1);

    let after_delete = store.fetch(&[2], true).expect("fetch after delete");
    assert!(after_delete.ids.is_empty());
    assert_eq!(after_delete.missing_ids, vec![2]);

    store
        .upsert(&[2], &[0.0, 0.0, 1.0], dims)
        .expect("upsert revive failed");
    let revived = store.fetch(&[2], true).expect("fetch revived");
    assert_eq!(revived.ids, vec![2]);
    assert_eq!(revived.missing_ids.len(), 0);
    assert_eq!(revived.vectors.expect("values"), vec![0.0, 0.0, 1.0]);
}

#[test]
fn stats_and_compact_remove_tombstones() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 3;
    store
        .upsert(&[1, 2], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dims)
        .expect("upsert failed");
    let _ = store.delete(&[2]).expect("delete failed");

    let before = store.stats().expect("stats before");
    assert_eq!(before.visible_count, 1);
    assert_eq!(before.tombstone_count, 1);
    assert_eq!(before.total_count, 2);

    let compact = store.compact(dims).expect("compact failed");
    assert_eq!(compact.kept_count, 1);
    assert_eq!(compact.removed_tombstones, 1);

    let after = store.stats().expect("stats after");
    assert_eq!(after.visible_count, 1);
    assert_eq!(after.tombstone_count, 0);
    assert_eq!(after.total_count, 1);

    let q = [1.0, 0.0, 0.0];
    let result = store.search(&q, 2).expect("search after compact");
    assert!(!result.ids.is_empty());
    assert_eq!(result.ids[0], 1);
}

#[test]
fn changes_pagination_reports_ordered_events() {
    let ctx = TestContext::new();
    let store = VectorStore::open(ctx.db_path()).expect("open failed");

    let dims: u32 = 2;
    store.upsert(&[7], &[1.0, 0.0], dims).expect("upsert");
    let _ = store.delete(&[7]).expect("delete");
    let _ = store.compact(dims).expect("compact");

    let page1 = store.changes(0, 2).expect("changes page1");
    assert_eq!(page1.events.len(), 2);
    assert!(page1.events[0].seq < page1.events[1].seq);
    assert_eq!(page1.events[0].op, ChangeOp::Upsert);
    assert_eq!(page1.events[1].op, ChangeOp::Delete);
    assert!(page1.has_more);

    let page2 = store.changes(page1.next_since, 10).expect("changes page2");
    assert!(!page2.events.is_empty());
    assert_eq!(page2.events[0].op, ChangeOp::Compact);
}
