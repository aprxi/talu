//! Talu server integration tests.
//!
//! Requires a pre-built `talu` CLI binary.
//! Run: `cargo test --test server`

mod server {
    pub mod auth;
    pub mod common;
    pub mod compliance;
    pub mod console;
    pub mod conversations;
    pub mod documents;
    pub mod files;
    pub mod plugins;
    pub mod responses;
    pub mod search;
    pub mod tags;
}
