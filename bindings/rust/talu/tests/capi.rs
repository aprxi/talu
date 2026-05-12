//! C-API integration test suite.
//!
//! Exercises the `talu` safe Rust wrappers over the C API
//! (`core/src/capi/`). Covers inference, tokenizer, template, and
//! responses surfaces.
//!
//! Run: `cargo test --test capi`

mod capi {
    pub mod abi;

    pub mod responses {
        pub mod chat;
        pub mod common;
        pub mod conversation;
        pub mod engine;
        pub mod items;
        pub mod layout;
        pub mod serialization;
        pub mod structured_output;
        pub mod tool_calling;
        pub mod validation;
        pub mod write;
    }

    pub mod template;
    pub mod tokenizer;
}
