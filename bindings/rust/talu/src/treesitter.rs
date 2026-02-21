//! Safe wrappers for tree-sitter code parsing, highlighting, and analysis.
//!
//! All functions return raw JSON strings produced by the Zig core.
//! This enables zero-copy forwarding in HTTP handlers — pass the string
//! directly to the response body without serde round-trips.

use std::ffi::{c_char, c_void, CStr, CString};

use crate::error::error_from_last_or;
use crate::Result;

/// Helper: call a Zig C API function that writes a NUL-terminated JSON string
/// into an out-pointer, then convert it to an owned Rust String and free the
/// C-allocated original.
///
/// # Safety contract
/// The closure must call a C API function that:
/// - Writes a NUL-terminated `[*:0]u8` into the provided `*mut c_void` on success
/// - Returns 0 on success, non-zero on failure
/// - Allocates via `std.heap.c_allocator` (freed by `talu_text_free`)
unsafe fn call_returning_json(
    f: impl FnOnce(*mut c_void) -> i32,
    fallback_err: &str,
) -> Result<String> {
    let mut out_ptr: *mut c_char = std::ptr::null_mut();
    let rc = f(&mut out_ptr as *mut _ as *mut c_void);
    if rc != 0 {
        return Err(error_from_last_or(fallback_err));
    }
    if out_ptr.is_null() {
        return Err(error_from_last_or(fallback_err));
    }
    let c_str = CStr::from_ptr(out_ptr);
    // SAFETY: The Zig core produces valid UTF-8 JSON. Skip redundant validation.
    let result = String::from_utf8_unchecked(c_str.to_bytes().to_vec());
    talu_sys::talu_text_free(out_ptr);
    Ok(result)
}

/// Parse and highlight source code, returning a JSON array of tokens.
///
/// Returns `[{"s":0,"e":3,"t":"syntax-keyword"}, ...]` where s=start byte,
/// e=end byte, t=CSS class name.
pub fn highlight(source: &[u8], language: &str) -> Result<String> {
    let c_lang = CString::new(language)?;
    // SAFETY: source ptr/len are valid; c_lang is a valid CString; out_json is a valid out-param.
    unsafe {
        call_returning_json(
            |out| {
                talu_sys::talu_treesitter_highlight(
                    source.as_ptr(),
                    source.len() as u32,
                    c_lang.as_ptr(),
                    out,
                )
            },
            "Highlight failed",
        )
    }
}

/// Parse and highlight with rich output (positions, node kinds, text).
///
/// Returns `[{"s":0,"e":3,"t":"syntax-keyword","nk":"keyword","tx":"def",
/// "sr":0,"sc":0,"er":0,"ec":3}, ...]`.
pub fn highlight_rich(source: &[u8], language: &str) -> Result<String> {
    let c_lang = CString::new(language)?;
    // SAFETY: source ptr/len are valid; c_lang is a valid CString; out_json is a valid out-param.
    unsafe {
        call_returning_json(
            |out| {
                talu_sys::talu_treesitter_highlight_rich(
                    source.as_ptr(),
                    source.len() as u32,
                    c_lang.as_ptr(),
                    out,
                )
            },
            "Rich highlight failed",
        )
    }
}

/// Parse source code and return a JSON AST representation.
///
/// Returns `{"language":"<lang>","tree":{...recursive node JSON...}}`.
pub fn parse_to_json(source: &[u8], language: &str) -> Result<String> {
    let c_lang = CString::new(language)?;

    // Create parser
    // SAFETY: c_lang is a valid CString.
    let parser = unsafe { talu_sys::talu_treesitter_parser_create(c_lang.as_ptr()) };
    if parser.is_null() {
        return Err(error_from_last_or("Failed to create parser"));
    }

    // Parse source
    let mut tree: *mut c_void = std::ptr::null_mut();
    // SAFETY: parser is a valid handle; source ptr/len are valid; tree is a valid out-param.
    let rc = unsafe {
        talu_sys::talu_treesitter_parse(
            parser,
            source.as_ptr(),
            source.len() as u32,
            &mut tree as *mut _ as *mut c_void,
        )
    };
    if rc != 0 || tree.is_null() {
        // SAFETY: parser is a valid handle.
        unsafe { talu_sys::talu_treesitter_parser_free(parser) };
        return Err(error_from_last_or("Parse failed"));
    }

    // Convert tree to JSON
    // SAFETY: tree is a valid handle; source ptr/len are valid; c_lang is valid.
    let result = unsafe {
        call_returning_json(
            |out| {
                talu_sys::talu_treesitter_tree_json(
                    tree,
                    source.as_ptr(),
                    source.len() as u32,
                    c_lang.as_ptr(),
                    out,
                )
            },
            "Tree to JSON failed",
        )
    };

    // Cleanup
    // SAFETY: tree and parser are valid handles.
    unsafe {
        talu_sys::talu_treesitter_tree_free(tree);
        talu_sys::talu_treesitter_parser_free(parser);
    }

    result
}

/// Execute an S-expression query against source code.
///
/// Returns a JSON array of matches:
/// `[{"id":0,"captures":[{"name":"fn","start":0,"end":3,"text":"def"}]}, ...]`.
pub fn query(source: &[u8], language: &str, pattern: &str) -> Result<String> {
    let c_lang = CString::new(language)?;

    // Create parser and parse
    // SAFETY: c_lang is a valid CString.
    let parser = unsafe { talu_sys::talu_treesitter_parser_create(c_lang.as_ptr()) };
    if parser.is_null() {
        return Err(error_from_last_or("Failed to create parser"));
    }

    let mut tree: *mut c_void = std::ptr::null_mut();
    // SAFETY: parser is valid; source ptr/len are valid; tree is a valid out-param.
    let rc = unsafe {
        talu_sys::talu_treesitter_parse(
            parser,
            source.as_ptr(),
            source.len() as u32,
            &mut tree as *mut _ as *mut c_void,
        )
    };
    if rc != 0 || tree.is_null() {
        unsafe { talu_sys::talu_treesitter_parser_free(parser) };
        return Err(error_from_last_or("Parse failed"));
    }

    // Compile query
    let mut query_handle: *mut c_void = std::ptr::null_mut();
    // SAFETY: c_lang is valid; pattern ptr/len are valid; query_handle is a valid out-param.
    let rc = unsafe {
        talu_sys::talu_treesitter_query_create(
            c_lang.as_ptr(),
            pattern.as_bytes().as_ptr(),
            pattern.len() as u32,
            &mut query_handle as *mut _ as *mut c_void,
        )
    };
    if rc != 0 || query_handle.is_null() {
        unsafe {
            talu_sys::talu_treesitter_tree_free(tree);
            talu_sys::talu_treesitter_parser_free(parser);
        }
        return Err(error_from_last_or("Query compilation failed"));
    }

    // Execute query
    // SAFETY: query_handle, tree are valid; source ptr/len are valid.
    let result = unsafe {
        call_returning_json(
            |out| {
                talu_sys::talu_treesitter_query_exec(
                    query_handle,
                    tree,
                    source.as_ptr(),
                    source.len() as u32,
                    out,
                )
            },
            "Query execution failed",
        )
    };

    // Cleanup
    // SAFETY: all handles are valid.
    unsafe {
        talu_sys::talu_treesitter_query_free(query_handle);
        talu_sys::talu_treesitter_tree_free(tree);
        talu_sys::talu_treesitter_parser_free(parser);
    }

    result
}

/// Extract callable definitions and import aliases from source code.
///
/// Returns `{"callables":[...],"aliases":[...]}`.
pub fn extract_callables(
    source: &[u8],
    language: &str,
    file_path: &str,
    project_root: &str,
) -> Result<String> {
    let c_lang = CString::new(language)?;
    let c_file_path = CString::new(file_path)?;
    let c_project_root = CString::new(project_root)?;
    // SAFETY: source ptr/len are valid; all CStrings are valid; out_json is a valid out-param.
    unsafe {
        call_returning_json(
            |out| {
                talu_sys::talu_treesitter_extract_callables(
                    source.as_ptr(),
                    source.len() as u32,
                    c_lang.as_ptr(),
                    c_file_path.as_ptr(),
                    c_project_root.as_ptr(),
                    out,
                )
            },
            "Callable extraction failed",
        )
    }
}

/// Extract call sites with import-aware resolution.
///
/// Returns a JSON array of call site objects with resolved paths.
pub fn extract_call_sites(
    source: &[u8],
    language: &str,
    definer_fqn: &str,
    file_path: &str,
    project_root: &str,
) -> Result<String> {
    let c_lang = CString::new(language)?;
    let c_definer = CString::new(definer_fqn)?;
    let c_file_path = CString::new(file_path)?;
    let c_project_root = CString::new(project_root)?;
    // SAFETY: source ptr/len are valid; all CStrings are valid; out_json is a valid out-param.
    unsafe {
        call_returning_json(
            |out| {
                talu_sys::talu_treesitter_extract_call_sites(
                    source.as_ptr(),
                    source.len() as u32,
                    c_lang.as_ptr(),
                    c_definer.as_ptr(),
                    c_file_path.as_ptr(),
                    c_project_root.as_ptr(),
                    out,
                )
            },
            "Call site extraction failed",
        )
    }
}

// ---------------------------------------------------------------------------
// Session-oriented handles (RAII wrappers for incremental parsing)
// ---------------------------------------------------------------------------

/// Owned handle for a tree-sitter parser. Freed on drop.
///
/// Not `Send`/`Sync` — tree-sitter parsers are single-threaded.
pub struct ParserHandle {
    ptr: *mut c_void,
}

impl ParserHandle {
    /// Create a parser for the given language.
    pub fn new(language: &str) -> Result<Self> {
        let c_lang = CString::new(language)?;
        // SAFETY: c_lang is a valid CString.
        let ptr = unsafe { talu_sys::talu_treesitter_parser_create(c_lang.as_ptr()) };
        if ptr.is_null() {
            return Err(error_from_last_or("Failed to create parser"));
        }
        Ok(Self { ptr })
    }

    /// Parse source code, returning an owned tree.
    ///
    /// If `old_tree` is provided, tree-sitter reuses unchanged subtrees
    /// for faster incremental re-parsing.
    pub fn parse(&self, source: &[u8], old_tree: Option<&TreeHandle>) -> Result<TreeHandle> {
        let mut tree_ptr: *mut c_void = std::ptr::null_mut();
        let old = old_tree.map_or(std::ptr::null_mut(), |t| t.ptr);
        // SAFETY: self.ptr is valid; source ptr/len are valid; old is valid or null.
        let rc = unsafe {
            talu_sys::talu_treesitter_parse_incremental(
                self.ptr,
                source.as_ptr(),
                source.len() as u32,
                old,
                &mut tree_ptr as *mut _ as *mut c_void,
            )
        };
        if rc != 0 || tree_ptr.is_null() {
            return Err(error_from_last_or("Parse failed"));
        }
        Ok(TreeHandle { ptr: tree_ptr })
    }

    /// Raw pointer access (for advanced use).
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for ParserHandle {
    fn drop(&mut self) {
        // SAFETY: self.ptr is valid (set in new, never modified).
        unsafe { talu_sys::talu_treesitter_parser_free(self.ptr) };
    }
}

// SAFETY: ParserHandle is a C heap pointer. Moving between threads is safe.
// Concurrent access is NOT safe — callers must use external synchronization (Mutex).
unsafe impl Send for ParserHandle {}

/// Describes a text edit for incremental parsing.
///
/// Both byte offsets and (row, column) coordinates are required.
/// Row and column are 0-indexed.
#[derive(Debug, Clone, Copy)]
pub struct InputEdit {
    pub start_byte: u32,
    pub old_end_byte: u32,
    pub new_end_byte: u32,
    pub start_row: u32,
    pub start_column: u32,
    pub old_end_row: u32,
    pub old_end_column: u32,
    pub new_end_row: u32,
    pub new_end_column: u32,
}

/// Owned handle for a parsed syntax tree. Freed on drop.
pub struct TreeHandle {
    ptr: *mut c_void,
}

impl TreeHandle {
    /// Inform tree-sitter that the source has been edited.
    ///
    /// You **must** call this before passing the tree to `ParserHandle::parse()`
    /// as `old_tree` for correct incremental parsing. Without it, tree-sitter
    /// cannot identify which nodes to reuse and will do significantly more work.
    pub fn edit(&mut self, edit: InputEdit) {
        // SAFETY: self.ptr is valid; all parameters are plain integers.
        unsafe {
            talu_sys::talu_treesitter_tree_edit(
                self.ptr,
                edit.start_byte,
                edit.old_end_byte,
                edit.new_end_byte,
                edit.start_row,
                edit.start_column,
                edit.old_end_row,
                edit.old_end_column,
                edit.new_end_row,
                edit.new_end_column,
            );
        }
    }

    /// Highlight this tree's source code, returning compact JSON tokens.
    ///
    /// `source` must be the same bytes this tree was parsed from.
    pub fn highlight(&self, source: &[u8], language: &str) -> Result<String> {
        let c_lang = CString::new(language)?;
        self.highlight_with_c_lang(source, &c_lang)
    }

    /// Like [`highlight`](Self::highlight), but accepts a pre-allocated `CStr`.
    ///
    /// Use this in hot loops to avoid per-call `CString` allocation.
    pub fn highlight_with_c_lang(&self, source: &[u8], c_lang: &CStr) -> Result<String> {
        // SAFETY: self.ptr is valid; source ptr/len are valid; c_lang is valid.
        unsafe {
            call_returning_json(
                |out| {
                    talu_sys::talu_treesitter_highlight_from_tree(
                        self.ptr,
                        source.as_ptr(),
                        source.len() as u32,
                        c_lang.as_ptr(),
                        out,
                    )
                },
                "Highlight from tree failed",
            )
        }
    }

    /// Highlight this tree with rich output (positions, node kinds, text).
    ///
    /// `source` must be the same bytes this tree was parsed from.
    pub fn highlight_rich(&self, source: &[u8], language: &str) -> Result<String> {
        let c_lang = CString::new(language)?;
        self.highlight_rich_with_c_lang(source, &c_lang)
    }

    /// Like [`highlight_rich`](Self::highlight_rich), but accepts a pre-allocated `CStr`.
    ///
    /// Use this in hot loops to avoid per-call `CString` allocation.
    pub fn highlight_rich_with_c_lang(&self, source: &[u8], c_lang: &CStr) -> Result<String> {
        // SAFETY: self.ptr is valid; source ptr/len are valid; c_lang is valid.
        unsafe {
            call_returning_json(
                |out| {
                    talu_sys::talu_treesitter_highlight_rich_from_tree(
                        self.ptr,
                        source.as_ptr(),
                        source.len() as u32,
                        c_lang.as_ptr(),
                        out,
                    )
                },
                "Rich highlight from tree failed",
            )
        }
    }

    /// Execute an S-expression query against this tree.
    ///
    /// Returns a JSON array of matches. Uses the existing parsed tree
    /// instead of re-parsing, making it suitable for real-time queries
    /// against a live session.
    pub fn query(&self, source: &[u8], language: &str, pattern: &str) -> Result<String> {
        let c_lang = CString::new(language)?;
        self.query_with_c_lang(source, &c_lang, pattern)
    }

    /// Like [`query`](Self::query), but accepts a pre-allocated `CStr` for the language.
    pub fn query_with_c_lang(&self, source: &[u8], c_lang: &CStr, pattern: &str) -> Result<String> {
        // Compile query
        let mut query_handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: c_lang is valid; pattern ptr/len are valid; query_handle is a valid out-param.
        let rc = unsafe {
            talu_sys::talu_treesitter_query_create(
                c_lang.as_ptr(),
                pattern.as_bytes().as_ptr(),
                pattern.len() as u32,
                &mut query_handle as *mut _ as *mut c_void,
            )
        };
        if rc != 0 || query_handle.is_null() {
            return Err(error_from_last_or("Query compilation failed"));
        }

        // Execute query against this tree
        // SAFETY: query_handle and self.ptr are valid; source ptr/len are valid.
        let result = unsafe {
            call_returning_json(
                |out| {
                    talu_sys::talu_treesitter_query_exec(
                        query_handle,
                        self.ptr,
                        source.as_ptr(),
                        source.len() as u32,
                        out,
                    )
                },
                "Query execution failed",
            )
        };

        // SAFETY: query_handle is valid.
        unsafe { talu_sys::talu_treesitter_query_free(query_handle) };

        result
    }

    /// Raw pointer access (for advanced use).
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for TreeHandle {
    fn drop(&mut self) {
        // SAFETY: self.ptr is valid (set in parse, never modified).
        unsafe { talu_sys::talu_treesitter_tree_free(self.ptr) };
    }
}

// SAFETY: TreeHandle is a C heap pointer to an immutable tree. Moving between
// threads is safe. The tree is read-only after creation.
unsafe impl Send for TreeHandle {}

/// Get the list of supported languages (comma-separated).
pub fn languages() -> Result<String> {
    // SAFETY: out_str is a valid out-param.
    unsafe { call_returning_json(|out| talu_sys::talu_treesitter_languages(out), "Failed to get languages") }
}

/// Detect language from a filename or extension.
///
/// Returns the canonical language name (e.g., "python", "javascript").
pub fn language_from_filename(filename: &str) -> Result<String> {
    let c_filename = CString::new(filename)?;
    // SAFETY: c_filename is a valid CString; out_lang is a valid out-param.
    unsafe {
        call_returning_json(
            |out| talu_sys::talu_treesitter_language_from_filename(c_filename.as_ptr(), out),
            "Language detection failed",
        )
    }
}
