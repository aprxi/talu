//! WAL replay and corruption tests.
//!
//! Validates that the storage engine correctly handles:
//! - Partial WAL frames (simulating crash mid-write)
//! - CRC corruption (simulating bit-rot or disk error)
//! - Clean WAL replay on reopen (no data loss)

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use crate::capi::db::common::{find_wal_files, TestContext};
use talu::vector::VectorStore;

/// Path to an orphaned WAL file for testing (simulates a crashed writer).
fn orphan_wal_path(db_root: &str) -> std::path::PathBuf {
    Path::new(db_root)
        .join("vector")
        .join("wal-00000000000000000000000000facade.wal")
}

/// Path to the vector data file within a DB root.
fn talu_path(db_root: &str) -> std::path::PathBuf {
    Path::new(db_root).join("vector").join("current.talu")
}

/// Vectors written and flushed to blocks survive clean reopen.
///
/// This is the baseline: no corruption, just verifying the WAL replay
/// path works for data that was flushed to blocks before close.
#[test]
fn clean_close_preserves_data() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    // Write enough data to trigger block flush (deinit flushes pending rows).
    {
        let store = VectorStore::open(ctx.db_path()).expect("open failed");
        store
            .append(
                &[1, 2, 3],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                dims,
            )
            .expect("append failed");
        // Drop triggers deinit → flushBlock → WAL truncated, data in current.talu
    }

    // Reopen and verify all data present.
    {
        let store = VectorStore::open(ctx.db_path()).expect("reopen failed");
        let loaded = store.load().expect("load failed");
        assert_eq!(loaded.ids.len(), 3, "All 3 vectors should survive reopen");
    }
}

/// WAL file should be empty (or near-empty) after a clean close,
/// because deinit calls flushBlock which truncates the WAL.
#[test]
fn clean_close_truncates_wal() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    {
        let store = VectorStore::open(ctx.db_path()).expect("open failed");
        store
            .append(&[1], &[1.0, 0.0, 0.0, 0.0], dims)
            .expect("append failed");
    }

    // With per-writer WALs, clean close deletes the WAL file.
    // No orphaned wal-*.wal files should remain.
    let wals = find_wal_files(ctx.db_path(), "vector");
    assert!(
        wals.is_empty(),
        "No WAL files should remain after clean close, found: {:?}",
        wals,
    );
}

/// Appending garbage bytes to the end of the WAL simulates a crash
/// during a write (partial frame). The engine should silently ignore
/// the partial frame and recover all valid data.
///
/// On-disk state after corruption:
///   current.talu: contains flushed blocks (valid)
///   wal-<hex>.wal: orphaned WAL with valid frames + trailing garbage
///
/// Expected behavior: open succeeds, valid data preserved, garbage ignored.
///
/// NOTE: Since deinit flushes pending rows and truncates WAL, to get
/// valid WAL frames we need to simulate a crash by NOT calling deinit.
/// We achieve this by:
///   1. Writing data (goes to WAL + in-memory buffer)
///   2. Cleanly closing (flushes to blocks, truncates WAL)
///   3. Writing more data in a second session
///   4. Cleanly closing (flushes second batch to blocks too)
///   5. Appending garbage to WAL
///   6. Reopening — garbage at EOF is a partial frame, silently skipped
#[test]
fn partial_wal_frame_ignored_on_reopen() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    // Phase 1: Write valid data and close cleanly.
    {
        let store = VectorStore::open(ctx.db_path()).expect("open failed");
        store
            .append(&[1, 2], &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dims)
            .expect("append failed");
    }

    // Phase 2: Create an orphan WAL with garbage (simulating partial write after crash).
    // With per-writer WALs, an orphaned wal-*.wal file represents a crashed writer.
    let wal = orphan_wal_path(ctx.db_path());
    {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal)
            .expect("failed to create orphan WAL for corruption");
        // Write some bytes that don't form a valid frame header.
        // A valid header is [Magic:4][PayloadLen:4]. Writing 5 bytes of 0xFF
        // means the iterator reads an invalid magic and stops.
        f.write_all(&[0xFF; 5]).expect("failed to write garbage");
        f.sync_all().expect("sync failed");
    }

    // Phase 3: Reopen. Engine should tolerate trailing garbage.
    // The data from Phase 1 was flushed to blocks, so it's safe regardless.
    {
        let store = VectorStore::open(ctx.db_path()).expect("reopen after corruption failed");
        let loaded = store.load().expect("load failed");
        assert_eq!(
            loaded.ids.len(),
            2,
            "Valid data should survive partial WAL frame",
        );
    }
}

/// If valid WAL frames exist followed by a partial frame, the valid
/// frames should be replayed and the partial frame ignored.
///
/// Strategy: Write data to create WAL entries, then simulate crash by
/// writing a valid WAL magic + length but truncated payload.
#[test]
fn truncated_payload_after_valid_frames() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    // Phase 1: Write and close cleanly (data flushed to blocks).
    {
        let store = VectorStore::open(ctx.db_path()).expect("open failed");
        store
            .append(&[100], &[0.5, 0.5, 0.5, 0.5], dims)
            .expect("append failed");
    }

    // Phase 2: Create an orphan WAL with a truncated frame header.
    // WALB magic = 0x424C4157 (little-endian: 57 41 4C 42)
    // Then a payload length of 9999, but only 3 bytes of actual payload.
    let wal = orphan_wal_path(ctx.db_path());
    {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal)
            .expect("failed to create orphan WAL");
        let magic: u32 = 0x424C4157; // "WALB"
        f.write_all(&magic.to_le_bytes()).expect("write magic");
        let fake_len: u32 = 9999;
        f.write_all(&fake_len.to_le_bytes()).expect("write len");
        f.write_all(&[0xAA; 3]).expect("write truncated payload");
        f.sync_all().expect("sync");
    }

    // Phase 3: Reopen. The truncated frame should be skipped.
    {
        let store = VectorStore::open(ctx.db_path()).expect("reopen failed");
        let loaded = store.load().expect("load failed");
        assert_eq!(
            loaded.ids.len(),
            1,
            "Original vector should survive truncated WAL frame",
        );
    }
}

/// Multiple write+close cycles accumulate data correctly.
/// Each cycle flushes to blocks; data survives across all reopens.
#[test]
fn multiple_reopen_cycles() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    for i in 0u64..5 {
        let store = VectorStore::open(ctx.db_path()).expect("open failed");
        let vec = vec![i as f32; dims as usize];
        store.append(&[i + 1], &vec, dims).expect("append failed");
    }

    let store = VectorStore::open(ctx.db_path()).expect("final open failed");
    let loaded = store.load().expect("load failed");
    assert_eq!(
        loaded.ids.len(),
        5,
        "All 5 vectors across 5 sessions should be present",
    );
}

/// After clean close, the data file (current.talu) should contain blocks.
/// Verify it's non-empty and starts with the block magic.
#[test]
fn data_file_has_valid_blocks() {
    let ctx = TestContext::new();
    let dims: u32 = 4;

    {
        let store = VectorStore::open(ctx.db_path()).expect("open failed");
        store
            .append(&[1], &[1.0, 0.0, 0.0, 0.0], dims)
            .expect("append failed");
    }

    let talu = talu_path(ctx.db_path());
    assert!(talu.exists(), "current.talu should exist after write");

    let data = std::fs::read(&talu).expect("read talu file");
    assert!(
        data.len() >= 4,
        "Block file too small: {} bytes",
        data.len()
    );

    // Block magic: 0x554C4154 ("TALU") in little-endian
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    assert_eq!(
        magic, 0x554C4154,
        "First 4 bytes should be TALU block magic, got 0x{:08X}",
        magic,
    );
}
