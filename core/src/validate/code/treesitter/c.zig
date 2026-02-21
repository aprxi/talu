//! Raw C bindings for tree-sitter.
//!
//! Re-exports the tree-sitter C API from tree_sitter/api.h.
//! Other treesitter modules import this file for C type access.

pub const c = @cImport({
    @cInclude("tree_sitter/api.h");
});
