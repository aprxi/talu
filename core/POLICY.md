# Zig Core Policy

> This document is the Architectural Specification for `core/src/`.
> It translates [AGENTS.md](../AGENTS.md) into enforceable Zig patterns.

**Scope:** `core/src/` directory.

---

## Definitions

| Term | Meaning |
|------|---------|
| **API-public** | Functions/types exported via `src/capi/` for external consumers (Python, C, FFI). Strict documentation, stability, and safety requirements. |
| **Module-public** | Zig `pub fn`/`pub const` visible within the codebase but not exported externally. Internal implementation. |

When this document says "public" without qualification, it means **API-public**.

---

## ⚡ Quick Invariants (The Non-Negotiables)

*If you violate these, the architecture breaks. Do not merge.*

1. **C API = Logic Boundary:** `src/capi/` is thin glue code (<40 LOC/fn). All logic lives in internal modules (`src/tokenizer`, `src/inference`).
2. **Zeroed Padding:** Extern structs crossing the FFI boundary MUST be initialized with `std.mem.zeroes()` to prevent data leaks.
3. **Stable Errors:** Numeric error codes are permanent. Never change or reuse an existing code.
4. **No Hot-Path Logging:** Logging is **forbidden** in hot paths (per-token/per-batch code). See §5 for the full definition.
5. **Allocators:** Tests MUST use `std.testing.allocator`. Production code MUST accept an allocator argument (no global allocator reliance).
6. **Defer Safety:** Never `defer free(ptr)` and then `return ptr` (Use-After-Free).
7. **Atomic Sync:** Changes affecting the C ABI surface MUST update `_native.py` (via `zig build gen-bindings`) in the same PR.

---

## 1. The C API (`src/capi/`)

**Concept:** The C API is the **only** external interface. It must be readable by non-Zig developers (Python/Rust maintainers). Minimize Zig-specific syntax; when Zig idioms are necessary, add brief inline comments explaining what they do.

### The Logic Budget

`talu_*` functions are **glue code**. They translate ABI types to internal Zig types and delegate immediately.

- **Allowed:** Pointer checks, null initialization, error mapping, simple casts.
- **Forbidden:** Loops (logic), complex switches, parsing logic, file I/O, network calls.
- **Heuristic:** If a C API function is >40 LOC, move the logic to an internal module.

**Where to put extracted logic:**

| Logic type | Target module |
|------------|---------------|
| Generic FFI conversions | `src/helpers/ffi.zig` |
| Domain-specific FFI types | The relevant domain module |
| Domain logic | The relevant `src/` domain module |
| Path operations | `src/io/` modules |

**Note:** Adding a private `fn helper()` to a capi file to "keep the export thin" is still a violation—the helper belongs in a core module.

### Function Naming

Names follow: `talu_<module>_<action>`.

```zig
// CORRECT
talu_session_create
talu_tokenizer_encode
talu_chat_append

// INCORRECT
talu_sess_new      // Unclear abbreviation
talu_encode        // Missing module prefix
```

### Parameter Naming

Parameters MUST be self-explanatory. Use full words, not abbreviations.

```zig
// CORRECT
pub export fn talu_tokenizer_encode(
    tokenizer: *Tokenizer,
    text: [*:0]const u8,
    out_tokens: *[*]u32,
    out_count: *usize,
) callconv(.c) i32;

// INCORRECT
pub export fn talu_tokenizer_encode(
    t: *Tokenizer,      // What is 't'?
    s: [*:0]const u8,   // What is 's'?
    out: *[*]u32,       // Out of what?
    n: *usize,          // 'n' of what?
) callconv(.c) i32;
```

### Function Structure

1. **Clear & Initialize:** Call `clearError()`, then set all output pointers to `null` / `0` **before any fallible operation**.
2. **Delegate:** Call the internal module.
3. **Handle Error:** If error, call `setError()`, return code.
4. **Success:** Assign output, return 0.

**Safety invariant:** Out-params must be initialized before any `try` or `catch`. This prevents returning garbage pointers on error paths.

```zig
pub export fn talu_chat_create(out_chat: *?*Chat) callconv(.c) i32 {
    clearError();
    out_chat.* = null; // Before any fallible work
    const chat = Chat.init(std.heap.c_allocator) catch |err| {
        setError(err, "Failed to create chat");
        return errorToCode(err);
    };
    out_chat.* = chat;
    return 0;
}
```

### Strings & Handles

- **Input strings:** `[*:0]const u8` (sentinel-terminated)
- **Output strings:** Always NUL-terminated
- **Opaque handles:** Prefer `opaque {}` over `extern struct` to avoid locking ABI layout

### Error Handling

**Rule:** All error code assignment goes through `error_codes.errorToCode()`.

- **Internal:** Return `!T`. Call `capi_error.setContext()` for details.
- **Boundary:** Catch error, call `capi_error.setError()`, return `i32`.
- **Stability:** Existing error codes are **immutable**. You may add new codes, but never change the meaning of an existing integer.
- **Naming:** Be specific: `error.WeightShapeMismatch`, not `error.BadInput`.
- **No module-specific mappers:** Do not write `fn mapXError()` functions in capi modules. All mapping goes through `errorToCode()`.

---

## 2. Memory Safety & Management

**Concept:** We are responsible for preventing leaks and data exposure.

### Extern Structs (Security)

**Rule:** `extern struct` instances must be initialized with `std.mem.zeroes` to clear padding bytes. Stack garbage in padding leaks secrets across the FFI boundary.

```zig
// CORRECT
var config = std.mem.zeroes(TaluConfig);
config.version = 1;

// FORBIDDEN (Leaks stack garbage in padding)
var config = TaluConfig{ .version = 1 };
```

### Allocator Discipline

- **Rule:** Functions that allocate MUST accept an `Allocator` parameter.
- **Exceptions:** `src/capi/` uses `c_allocator` (FFI requirement). Global singletons may use `page_allocator`. Hot-path scratch buffers should be caller-provided, not allocated.

### ArrayListUnmanaged

Use `ArrayListUnmanaged` when the struct does not own its allocator (allocator passed per-call). Use `ArrayList` only when the struct stores and owns the allocator.

### Arenas

Arenas MUST be scoped to request/session lifecycles. Never use for long-lived state.

### Errdefer Pattern

If a function performs multiple allocations, use `errdefer` to prevent leaks on partial failure.

```zig
const a = try allocator.alloc(u8, 10);
errdefer allocator.free(a); // Frees 'a' if next line fails
const b = try allocator.alloc(u8, 10);
```

**Exceptions:** Arena allocations (arena frees atomically), ownership transfer (`return result`), container append (container owns it).

### The "Defer-Return" Trap

**Forbidden:**
```zig
defer allocator.free(res);
return res; // RETURNS DANGLING POINTER
```

**Correct:** Use `errdefer` (runs only on error) or transfer ownership explicitly.

### `undefined` Variables

Permitted only when immediately overwritten. Comment required when the overwrite is non-obvious (conditional, inside loops, or multiple statements away).

### Pointer Stability

Structures exporting pointers to C MUST NOT reallocate while pointers are live. Document when pointers become invalid:

```zig
/// Returns pointer valid until next append(), clear(), or deinit().
pub fn getPtr(self: *Self) [*]T { ... }
```

### Alignment

Use `@alignCast` for aligned access or `@memcpy` for unaligned:

```zig
// Aligned
const floats: []f32 = @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0..len];

// Unaligned
var value: f32 = undefined;
@memcpy(std.mem.asBytes(&value), bytes[0..4]);
```

### Ownership

Document ownership in doc comments: "Caller owns returned X" or "Returned slice borrows from input."

---

## 3. Testing Strategy

**Concept:** Verify logic correctness without network dependency.

### Coverage

Every `pub fn` MUST have at least one test that exercises it with representative input and asserts correctness. Functions that return errors MUST additionally test each error condition.

**Enforcement:** `core/tests/check_coverage.sh`

### Allocator

Tests **MUST** use `std.testing.allocator` to detect leaks.

### Test Naming

Test names MUST contain the exact function name being tested:

```zig
test "softmaxContiguous sums to 1.0" { ... }           // CORRECT
test "softmaxContiguous handles empty input" { ... }  // CORRECT
test "Softmax correctness" { ... }                    // INCORRECT: wrong casing
```

### Prohibited Test Patterns

```zig
// No assertion
test "runs" { _ = someFunction(); }

// Tautology
test "returns" { try std.testing.expect(result.len >= 0); }

// No value check
test "works" { try std.testing.expect(result != null); }

// Unverified error
test "fails" { _ = someFunction(bad) catch return; }
```

### Integration Test Policy

All behavioral types exported from a module's `root.zig` must have integration tests in `core/tests/`.

**Behavioral type:** A struct with `pub fn` methods (excluding format/hash/eql interface methods).

**Exemption:** Data-only types (structs with no `pub fn`) do not need separate integration tests. They are tested via functions that create or consume them.

**Location:** Test directories mirror the module structure: `src/foo/root.zig` → `tests/foo/`.

**Naming:** snake_case test files for PascalCase structs: `Session` → `session_test.zig`, `ThreadPool` → `thread_pool_test.zig`.

**Coverage:** Every `pub fn` in a behavioral type must be exercised in its integration test file.

### Fuzz Testing

**API-public** functions parsing untrusted input (SafeTensors headers, JSON schemas, chat templates, grammars) MUST have fuzz tests in `src/capi/`.

```zig
test "fuzz talu_safetensors_parse" {
    try std.testing.fuzz(.{}, struct {
        fn testOne(input: []const u8) !void {
            var out: ?*Header = null;
            _ = talu_safetensors_parse(input.ptr, input.len, &out);
            if (out) |h| talu_safetensors_free(h);
        }
    }.testOne);
}
```

### Network Isolation

Network-dependent code must be split into:
1. **Fetch (Untestable):** Performs I/O, returns raw bytes. Tests verify signature/compile only.
2. **Parse (Testable):** Takes bytes, returns Struct. Full mock tests + fuzzing required.

### Numerical Correctness

Use `std.testing.expectApproxEqAbs`.

| Operation | Tolerance |
|-----------|-----------|
| Indices, counts, parsing, tokenization, quantization | 0 (bit-exact) |
| Elementwise f32 (relu, silu, gelu) | 1e-5 |
| Reductions f32 (softmax, norm) | 1e-4 |
| Matmul f32 | 1e-3 |
| Matmul bf16/f16 | 0.05 |
| Dequantized Q8_0 | 0.5 |
| Dequantized Q4_K/Q4_0 | 1.0 |

Never loosen tolerance to pass a failing test—fix the bug or skip with an issue.

### SIMD

Every SIMD implementation MUST have a scalar reference and an equivalence test.

### Skipped Tests

Skipped tests MUST have a GitHub issue number and empty body:

```zig
test "linearQ4_0 small matrix" {
    // SKIP(#247): Q4_0 nibble encoding bug
    if (true) return error.SkipZigTest;
}
```

---

## 4. Module Public API Contract

**Concept:** `root.zig` is the single source of truth for what a module exposes for cross-module use.

### Rule

Every module's `root.zig` defines its public API contract. Behavioral types (structs with `pub fn` methods) intended for use outside the module **must** be directly re-exported from that module's `root.zig`.

If a type is not re-exported from `root.zig`, it is internal-only. External code must not reach into sub-files to import it.

### Rationale

This enforces a soft internal policy where `root.zig` is the authoritative list of what is intended for cross-module consumption. It makes the public surface of each module discoverable, auditable, and intentional—not accidental.

### Correct Pattern

**Aggregator `root.zig`** — re-exports sub-modules and commonly-used types:

```zig
// core/src/compute/root.zig

// ===== Public API =====
pub const ops = @import("ops/root.zig");
pub const simd = @import("simd/root.zig");
pub const quant = @import("quant/root.zig");
pub const device = @import("device.zig");
pub const parallel = @import("parallel.zig");

// Re-export commonly used types at module level
pub const Device = device.Device;
pub const ThreadPool = parallel.ThreadPool;
pub const TensorView = ops.tensor_view.TensorView;
```

**Leaf `root.zig`** — exports functions and types directly:

```zig
// core/src/io/config/root.zig

pub const GenerationConfig = generation.GenerationConfig;
pub const loadGenerationConfig = generation.loadGenerationConfig;
pub fn loadConfig(...) !Config { ... }
pub fn checkArchitecture(...) !ArchitectureCheck { ... }
```

### Enforcement

When adding a new behavioral type to a module, add it to the module's `root.zig` re-exports in the same change. The Developer Checklist includes this as a gate.

---

## 5. Performance & Hot Paths

**Concept:** Correctness includes performance. Overhead in kernels is unacceptable.

### What Is "Hot Path"?

Hot path = code executed repeatedly per-token or per-batch during inference or tokenization:

- Matrix multiplication kernels
- Attention computation loops
- Tight encoding/decoding loops
- Activation functions, normalization, softmax
- Any code inside a per-token or per-element loop

**Primary hot-path directories:** `src/compute/`, `src/inference/kernels/`, `src/tokenizer/`.

**NOT hot path** (logging allowed):

- Initialization and model loading
- Error handling and fallback paths
- One-time setup and configuration
- Completion-boundary logs (one per generation)

### The Ban List

In hot paths, you MUST NOT:
1. **Log:** No logging whatsoever—not even project `log.*`. Even checked logging has overhead. Log at the caller level.
2. **Allocate:** No `allocator.alloc`. Pass scratch buffers from caller.
3. **Syscall:** No I/O or locking.

### Data Layout

1. Order struct fields by access frequency
2. Align SIMD data to register size (32B AVX2, 64B AVX-512)
3. Pad thread-local mutable state to cache line (64B)

### Validation & Safety

Validate at public API boundaries. Internal functions MAY use `@setRuntimeSafety(false)` with documented invariants.

### Branching

Hoist invariants outside loops. Use `@branchHint(.cold)` for error paths.

### Documentation

Hot path code MUST document: complexity (Big O), alignment requirements, register strategy (for SIMD).

---

## 6. Logging Discipline

**Concept:** Logs are structural data, not strings.

### Rules

1. **Source:** Use `core/src/log.zig`. **Never** use `std.log`.
2. **Looping:** Logs inside loops MUST include incremental context (e.g., `progress=i, total=n`). Do not spam identical messages.
3. **Structure:** Use `.name = val` syntax for structured attributes.

```zig
// CORRECT
log.debug("converter", "Processing", .{ .layer = i, .name = name }, @src());

// FORBIDDEN
std.log.debug("Processing {d} {s}", .{i, name});
```

### Scopes

| Scope | Description |
|-------|-------------|
| `fetch` | Remote repository operations |
| `load` | Loading model/weights into memory |
| `tokenizer` | Tokenization operations |
| `template` | Chat template rendering |
| `inference` | Inference execution |
| `converter` | Model conversion/quantization |
| `cli` | CLI argument parsing, dispatch |

---

## 7. Thread Safety

**Concept:** API-public types must document their threading guarantees.

Every **API-public** type (exported via `src/capi/`) MUST document thread safety using one of:

| Annotation | Meaning |
|------------|---------|
| `NOT thread-safe` | Single-threaded use only |
| `Immutable after initialization` | Safe to share after init |
| `Single-writer, multi-reader` | Caller coordinates access |
| `Fully thread-safe` | Internal synchronization |

Mutable module-level variables (`var` at file scope) MUST document thread safety.

**Exemptions** (no documentation needed): `const` globals, `std.atomic.*` types, variables protected by a documented mutex.

`threadlocal` variables MUST document that pointers/values cannot be shared across threads (they usually cannot).

---

## 8. Module Structure & Naming

### Module Structure

Every directory MUST have `root.zig` exporting its public API. External code imports through `root.zig` only.

**Exception:** `src/helpers/` modules are simple utilities imported directly (no `helpers/root.zig`).

Behavioral types intended for cross-module use MUST be re-exported from the module's `root.zig`. See **§4 Module Public API Contract** for the full rule and examples.

Test-only exports MUST be namespaced: `pub const testing = struct { ... };`

### Naming Conventions

| Category | Pattern | Examples |
|----------|---------|----------|
| Index | `*_idx` | `row_idx`, `token_idx` |
| Count | `*_count`, `n_*` | `row_count`, `n_heads` |
| Buffer | `*_buf` | `scratch_buf` |
| Pointer | `*_ptr` | `input_ptr` |

Single-letter variables (`i`, `j`) permitted only in loops ≤3 lines where bounds provide context.

No shadowing.

---

## 9. Formatting

1. `zig fmt` clean
2. ASCII only (no Unicode in identifiers)
3. No trailing whitespace
4. Single newline at EOF

---

## 10. Code Quality

### Prohibited Patterns

- **Convenience Wrappers:** Do not write `fn foo() { fooEx(null); }`. Update call sites.
- **Alias Exports:** Do not use `pub const Old = New;` for backward compatibility.
- **Undefined:** `var x: T = undefined;` only if immediately overwritten.
- **Fallback paths:** No `if (use_legacy_mode) { ... } else { ... }`.

**Exception:** Architecture-specific code paths (Metal vs CPU, AVX-512 vs AVX2) are permitted.

### The Policy Linter

The linter enforces these rules.

- `zig build lint` MUST pass.
- Suppressions (`// lint:ignore`) require a comment explaining safety.

---

## 11. API Stability

### Zero-Legacy Policy (Current Phase)

The C API currently operates under a **zero-legacy policy**. Breaking changes are permitted without deprecation cycles.

**Rationale:** We control all bindings (Python, Rust). Every binding test suite runs on every change. Breaking C API changes are immediately propagated and fixed atomically.

**Exception — Error Codes Are Permanent:** Even under zero-legacy, existing numeric error codes are never changed or reused. New codes can be added; existing codes keep their meaning forever.

**Binding Impact:** Python bindings have their own deprecation policy for external consumers—Python can absorb breaking C API changes internally and present a stable API to users.

### Future Stability (Post-Public)

Once bindings are public and versioned: error codes permanent, signatures stable, removal requires deprecation with defined removal version.

---

## 12. Documentation

### Required

1. File header (`//!`) - Every file must have a header explaining its purpose
2. Safety invariants (alignment, pointer validity, initialization order)
3. **C API** (`src/capi/`): Every exported function MUST have `///` doc comment

### When to Document

Doc comments (`///`) are valuable when they explain **why** or **how**, not **what**:

```zig
// GOOD: Explains non-obvious behavior
/// Returns -1 if token not in vocabulary (not an error, use for fallback logic).
pub fn tokenToId(self: *Self, token: []const u8) i32 { ... }

// BAD: Restates the obvious
/// Initializes the tokenizer.
pub fn init(allocator: Allocator) !Self { ... }
```

### Prohibited

1. **Filler documentation** - Doc comments that restate what the code already says
2. **Comments restating code** - `x += 1; // increment x`
3. **TODO/FIXME/HACK/XXX comments** - Not allowed in committed code. Complete the work or track in an issue. Use `// lint:ignore no-todo` with justification if temporary.
4. **Commented-out code** - Delete it or keep it

---

## Developer Checklist

Before marking complete:

**Architecture:**
- [ ] Logic lives in `src/`, not `src/capi/`?
- [ ] C API changes reflected in `_native.py`?
- [ ] No convenience wrappers or alias exports?
- [ ] Module has `root.zig` with public exports?
- [ ] Behavioral types exported from module `root.zig`?

**Safety:**
- [ ] `extern struct` init uses `std.mem.zeroes`?
- [ ] `undefined` variables immediately initialized?
- [ ] `errdefer` used for multi-step allocations?
- [ ] Pointer stability documented for FFI types?
- [ ] Ownership documented in doc comments?

**Testing:**
- [ ] Tests use `std.testing.allocator`?
- [ ] Test names contain exact function name?
- [ ] Integration tests exist in `core/tests/` for each behavioral type?
- [ ] Parsers have fuzz tests?
- [ ] No `std.log` or logging in hot paths (§5)?
- [ ] SIMD has scalar reference + equivalence test?

**Thread Safety:**
- [ ] API-public types document thread safety?
- [ ] Mutable globals document synchronization?
