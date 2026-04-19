//! Talu server integration tests.
//!
//! Requires a pre-built `talu` CLI binary.
//! Run: `cargo test --test server`

mod server {
    pub mod auth;
    pub mod common;
    pub mod completions;
    pub mod cors;
    pub mod openapi_docs;
    pub mod responses;
    pub mod tokenizer;
}
