# talu-sys

Low-level FFI bindings to the talu C API.

## Auto-Generation

**This crate's `src/lib.rs` is auto-generated. Do not edit it manually.**

### How It Works

The bindings are generated from Zig source files by a custom generator:

```
core/src/capi/*.zig          →  gen_bindings_rust.zig  →  talu-sys/src/lib.rs
core/src/router/*.zig        →        (parser)         →
core/src/converter/*.zig     →                         →
```

The generator (`core/tests/helpers/gen_bindings_rust.zig`) parses Zig's C API exports and produces Rust FFI declarations with proper `#[repr(C)]` types.

### Regenerating Bindings

From the repository root:

```bash
zig build gen-bindings-rust
```

This will:
1. Parse all `pub export fn` declarations in the scanned directories
2. Parse all `pub const ... = extern struct` definitions
3. Parse all `pub const ... = extern union` definitions
4. Parse all `pub const ... = enum(...)` definitions
5. Generate `bindings/rust/talu-sys/src/lib.rs`

### When to Regenerate

Regenerate bindings after any changes to:
- `core/src/capi/*.zig` - Main C API
- `core/src/router/capi_bridge.zig` - Router types
- `core/src/converter/scheme.zig` - Converter enums

### Type Mapping

| Zig Type | Rust Type |
|----------|-----------|
| `c_int`, `i32` | `c_int` |
| `usize` | `usize` |
| `bool` | `bool` |
| `[*:0]const u8` | `*const c_char` |
| `?*anyopaque` | `*mut c_void` |
| `*StructName` | `*mut StructName` |
| `*const StructName` | `*const StructName` |
| `extern struct` | `#[repr(C)] struct` |
| `extern union` | `#[repr(C)] union` |
| `enum(i32)` | `#[repr(i32)] enum` |

### Source Tracking

Each generated type includes a `/// Source:` comment indicating which Zig file it came from, making it easy to trace definitions back to their origin.

## Usage

This crate provides raw, unsafe FFI bindings. For a safe API, use the `talu` crate instead.

```rust
use talu_sys::*;
use std::ffi::CString;
use std::os::raw::c_void;

// Raw FFI - unsafe, manual memory management
let model_path = CString::new("/path/to/model").unwrap();
let mut handle: *mut c_void = std::ptr::null_mut();
unsafe {
    let rc = talu_backend_create(model_path.as_ptr(), &mut handle as *mut _ as *mut c_void);
    if rc != 0 {
        // Handle error
    }
    // ... use handle ...
    talu_backend_free(handle);
}
```

## See Also

- `../talu/` - Safe Rust wrappers
- `../cli/` - Command-line interface
- `../POLICY.md` - Coding standards
