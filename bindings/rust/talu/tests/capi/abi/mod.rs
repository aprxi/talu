//! ABI signature validation tests.
//!
//! These tests verify that auto-generated FFI bindings in `talu_sys` have
//! correct function signatures. They catch generator bugs that produce
//! incorrect pointer/value semantics.
//!
//! **Background**: The binding generator (`zig build gen-bindings-rust`) parses
//! Zig C API exports and generates Rust FFI bindings. A bug in the generator
//! can produce incorrect signatures (e.g., taking a struct by value instead of
//! by pointer), which causes memory corruption at runtime.
//!
//! **How it works**: These tests use compile-time type assertions to verify
//! function pointer types. If the generator produces incorrect signatures,
//! the code fails to compile with a type mismatch error.
//!
//! Run: `cargo test --test capi -- capi::abi`

mod signatures;
