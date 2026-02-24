//! Talu server integration tests.
//!
//! Requires a pre-built `talu` CLI binary.
//! Run: `cargo test --test server`

mod server {
    pub mod auth;
    pub mod code;
    pub mod common;
    pub mod compliance;
    pub mod console;
    pub mod conversations;
    pub mod db;
    pub mod file;
    pub mod files;
    pub mod openapi_docs;
    pub mod plugins;
    pub mod repo;
    pub mod responses;
    pub mod search;
    pub mod settings;
    pub mod tags;
}
