//! C-API integration test suite.
//!
//! Exercises the `talu` safe Rust wrappers over the C API
//! (`core/src/capi/`). Covers both the Responses API and TaluDB.
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

    pub mod db {
        pub mod blobs;
        pub mod chat;
        pub mod common;
        pub mod concurrency;
        pub mod documents;
        pub mod recovery;
        pub mod stress;
        pub mod topology;
        pub mod vector;
    }

    pub mod policy;
    pub mod template;
    pub mod tensor;
    pub mod tokenizer;
}
