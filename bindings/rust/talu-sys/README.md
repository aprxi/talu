# talu-sys

Low-level FFI bindings to the [talu](https://github.com/aprxi/talu) C API.

This crate provides raw `extern "C"` function declarations and `#[repr(C)]` type
definitions for linking against `libtalu`, the shared library that powers talu's
inference engine.

**Most users should depend on the [`talu`](https://crates.io/crates/talu) crate
instead**, which provides safe, idiomatic Rust wrappers with RAII resource management
and `Result`-based error handling.

## How it works

At build time, `talu-sys` locates the native `libtalu` shared library:

1. **`TALU_LIB_DIR` env var** — if set, links against the library in that directory
   (used for local development and custom builds).
2. **Automatic download** — if `TALU_LIB_DIR` is not set, downloads the pre-built
   library from the matching [GitHub Release](https://github.com/aprxi/talu/releases)
   based on the crate version and target platform.

### Supported targets

| Target | Library |
|--------|---------|
| `x86_64-unknown-linux-gnu` | `libtalu.so` |
| `aarch64-apple-darwin` | `libtalu.dylib` |

### Downstream crates

`talu-sys` exposes `DEP_TALU_LIB_DIR` via Cargo's
[`links`](https://doc.rust-lang.org/cargo/reference/build-scripts.html#the-links-manifest-key)
mechanism, so downstream crates can locate the library for rpath embedding.

## Example

```rust
use talu_sys::*;
use std::ffi::CString;
use std::os::raw::c_void;

let model_path = CString::new("/path/to/model").unwrap();
let mut handle: *mut c_void = std::ptr::null_mut();
unsafe {
    let rc = talu_backend_create(model_path.as_ptr(), &mut handle as *mut _ as *mut c_void);
    if rc != 0 {
        // handle error
    }
    // ... use handle ...
    talu_backend_free(handle);
}
```

## License

MIT
