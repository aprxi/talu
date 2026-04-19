//! C-API integration test suite.
//!
//! Exercises the `talu` safe Rust wrappers over the C API
//! (`core/src/capi/`). Covers inference, router, tokenizer, template,
//! and responses surfaces.
//!
//! Run: `cargo test --test capi`

mod capi {
    pub mod abi;

    pub mod responses {
        pub mod chat;
        pub mod common;
        pub mod content;
        pub mod conversation;
        pub mod iterator;
        pub mod serialization;
        pub mod structured_output;
        pub mod tool_calling;
        pub mod variants;
    }

    pub mod router;
    pub mod template;
    pub mod tokenizer;
}
