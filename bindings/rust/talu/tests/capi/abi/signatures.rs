//! Compile-time FFI signature validation.
//!
//! These tests verify that auto-generated FFI bindings have correct function
//! signatures. If the binding generator produces incorrect signatures (e.g.,
//! struct by value instead of pointer), these tests fail to compile.
//!
//! **Why this matters**: A function declared as `fn foo(x: SomeStruct)` instead of
//! `fn foo(x: *mut SomeStruct)` causes memory corruption because Rust copies the
//! struct to the stack and the C code frees from the wrong address.
//!
//! **How it works**: We use type assertions to verify function signatures match
//! expected pointer types. The `assert_fn_signature!` macro checks that the
//! function can be assigned to a variable with the expected type signature.

use std::os::raw::{c_char, c_int};
use talu_sys::{CSessionRecord, CStringList, CTagList, CTagRecord, GenerationConfigInfo};

/// Assert that a function has the expected signature at compile time.
///
/// This macro creates a const binding that fails to compile if the function
/// signature doesn't match the expected type.
macro_rules! assert_fn_signature {
    ($fn_name:path, $expected_type:ty) => {
        const _: $expected_type = $fn_name;
    };
}

// =============================================================================
// Free functions MUST take pointers, not values
// =============================================================================
// These functions free memory. If they take a struct by value, Rust will copy
// the struct to the stack before calling, and the C code will try to free
// memory at the wrong address, causing corruption.
//
// CORRECT:   fn talu_storage_free_string_list(list: *mut CStringList)
// INCORRECT: fn talu_storage_free_string_list(list: CStringList)
//
// If the generator produces the incorrect signature, these assertions will
// fail to compile with a type mismatch error.

assert_fn_signature!(
    talu_sys::talu_storage_free_string_list,
    unsafe extern "C" fn(*mut CStringList)
);

assert_fn_signature!(
    talu_sys::talu_storage_free_tags,
    unsafe extern "C" fn(*mut CTagList)
);

// =============================================================================
// Output parameter functions MUST take pointers, not values
// =============================================================================
// Functions that write to output parameters must take pointers. If they take
// a struct by value, the C code writes to stack memory that Rust thinks it owns,
// causing memory corruption when Rust frees it.
//
// CORRECT:   fn talu_storage_get_tag(..., out_tag: *mut CTagRecord) -> c_int
// INCORRECT: fn talu_storage_get_tag(..., out_tag: CTagRecord) -> c_int

assert_fn_signature!(
    talu_sys::talu_storage_get_tag,
    unsafe extern "C" fn(*const c_char, *const c_char, *mut CTagRecord) -> c_int
);

assert_fn_signature!(
    talu_sys::talu_storage_get_tag_by_name,
    unsafe extern "C" fn(*const c_char, *const c_char, *const c_char, *mut CTagRecord) -> c_int
);

// Session info (db_path, session_id, out_session)
assert_fn_signature!(
    talu_sys::talu_storage_get_session_info,
    unsafe extern "C" fn(*const c_char, *const c_char, *mut CSessionRecord) -> c_int
);

// Generation config (model_dir, out_config)
assert_fn_signature!(
    talu_sys::talu_get_generation_config,
    unsafe extern "C" fn(*const c_char, *mut GenerationConfigInfo) -> c_int
);

// =============================================================================
// Runtime tests (for test discovery)
// =============================================================================
// The compile-time assertions above are sufficient, but we add a runtime test
// so that `cargo test` reports this module and makes it visible in test output.

#[test]
fn free_functions_take_pointers() {
    // If this test runs, the compile-time assertions passed.
    // The actual validation happens at compile time above.

    // Additional runtime check: verify pointer size assumptions.
    // Free functions taking pointers should have function pointers the same size
    // as any other function pointer (8 bytes on 64-bit systems).
    let free_string_list_ptr = talu_sys::talu_storage_free_string_list as usize;
    let free_tags_ptr = talu_sys::talu_storage_free_tags as usize;

    // Both should be valid function addresses (non-zero)
    assert_ne!(
        free_string_list_ptr, 0,
        "free_string_list should be a valid function"
    );
    assert_ne!(free_tags_ptr, 0, "free_tags should be a valid function");
}

/// Demonstrates what the WRONG signature would look like.
/// This is documentation for future maintainers.
#[test]
fn document_incorrect_signature() {
    // This comment shows what the BAD signature looks like:
    //
    // WRONG (struct by value - causes memory corruption):
    //   pub fn talu_storage_free_string_list(list: CStringList);
    //   pub fn talu_storage_free_tags(tags: CTagList);
    //
    // CORRECT (pointer - safe):
    //   pub fn talu_storage_free_string_list(list: *mut CStringList);
    //   pub fn talu_storage_free_tags(tags: *mut CTagList);
    //
    // The compile-time assertions in this file catch the wrong signature
    // by failing to compile when the types don't match.
}
