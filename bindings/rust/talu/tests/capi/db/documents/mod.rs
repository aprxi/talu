//! Documents storage CAPI tests.
//!
//! Tests the Rust wrapper over the documents C API for:
//! - CRUD operations (create, get, update, delete)
//! - Listing and search
//! - Tag operations
//! - TTL and expiration
//! - Change tracking (CDC)
//! - Memory safety and error handling

mod lifecycle;
mod memory;
mod search;
mod tags;
mod ttl;
