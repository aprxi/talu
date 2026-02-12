//! Physical storage topology tests.
//!
//! Verifies on-disk layout invariants required for S3 replication:
//! - Append-only block files (existing bytes never modified)
//! - Namespace isolation (chat/ and vector/ are independent)
//! - Manifest paths are relative (portable across machines)

mod immutability;
mod manifest;
mod rotation;
