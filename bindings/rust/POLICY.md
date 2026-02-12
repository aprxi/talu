# Rust Policy

> This document is the Architectural Specification for `bindings/rust/`.
> It translates [AGENTS.md](../../AGENTS.md) into enforceable Rust patterns.

**Scope:** `bindings/rust/` directory.

---

## Definitions

| Term | Meaning |
|------|---------|
| **Public Rust API** | Symbols exported from `bindings/rust/talu/src/lib.rs` (and documented modules). These are user-facing contracts. |
| **FFI boundary** | Raw C ABI declarations in `bindings/rust/talu-sys/`. Unsafe, generated, and not hand-written business logic. |
| **Safe wrapper** | `bindings/rust/talu/` layer that owns safety, RAII, error conversion, and idiomatic Rust API shape. |
| **CLI layer** | `bindings/rust/cli/` crate for terminal/server UX. It is a consumer of `talu`, not a second FFI layer. |

When this document says "public API", it means the `talu` crate's exported surface.

---

## Quick Invariants (The Non-Negotiables)

*If you violate these, the architecture breaks. Do not merge.*

1. **One logic owner:** Domain logic lives in `core/src/`. Rust bindings are boundary layers (convert/validate/forward), not parallel implementations.
2. **Strict layer split:** `talu-sys` is raw FFI only, `talu` is safe API, `cli` uses `talu` (not direct `talu-sys`).
3. **Atomic FFI sync:** Any C API surface change is landed with regenerated `talu-sys` (`zig build gen-bindings-rust`) and updated `talu`/`cli` code and tests in the same change.
4. **Unsafe discipline:** Every unsafe operation must have a concrete safety invariant and smallest possible unsafe scope.
5. **Deterministic lifecycle:** Native resources are released via RAII (`Drop`) and ownership is explicit at type boundaries.
6. **Error fidelity:** Non-zero C return codes must map to `Result::Err` with preserved `talu_last_error()` context.
7. **No panic path for runtime failures:** Invalid input, IO/network failure, model errors, and C API errors return typed errors; they do not panic.
8. **No binding drift:** Rust behavior, tests, and docs must stay aligned with core C API semantics.

**Anti-circumvention:** These are architecture invariants, not style suggestions. Do not bypass them through indirection, wrapper layering, or code placement tricks to make checks pass.

---

## 1. Workspace Architecture

**Concept:** Three crates, one direction of dependency flow.

| Crate | Role | Constraints |
|------|------|-------------|
| `talu-sys/` | Generated raw ABI | Auto-generated declarations, C types, link setup. No business logic. |
| `talu/` | Safe Rust SDK | RAII handles, `Result` APIs, typed Rust structures over C ABI. |
| `cli/` | CLI and server entrypoints | UX, argument parsing, terminal/server glue. Calls into `talu`. |

**Dependency direction:**
- `talu-sys` -> no dependency on `talu` or `cli`
- `talu` -> depends on `talu-sys`
- `cli` -> depends on `talu` (and CLI-specific crates)

### Hard boundary rules

- Do not hand-edit generated FFI in `talu-sys/src/lib.rs`.
- Do not duplicate core rules (provider behavior, parsing logic, defaults) in Rust.
- Do not add direct `talu-sys` usage in CLI command logic.
- If `cli` needs new capability, add it to `talu` safe wrappers first (or core C API if missing there).

---

## 2. FFI and Unsafe Discipline

**Concept:** Unsafe is allowed only with explicit, local proofs.

### 2.1 Unsafe scope and documentation

- Keep unsafe blocks as small as possible.
- Every unsafe block must have a `// SAFETY:` comment stating exact preconditions and why they hold.
- Safety comments must explain this call site, not generic function behavior.

```rust
// SAFETY: `ptr` came from `talu_responses_create` and is still owned by `self`.
unsafe { talu_sys::talu_responses_free(self.ptr) };
```

### 2.2 String and pointer lifetime rules

- `CString` used for C calls must outlive the call and any C-side retained pointer.
- If C may retain pointers, store backing `CString` in the owning Rust struct.
- Convert incoming C strings with `CStr::from_ptr` only after null checks.
- Use `to_string_lossy()` where UTF-8 validity is not guaranteed.

### 2.3 Null and return code handling

- Check null pointers immediately after C calls that return pointers.
- Check non-zero return codes immediately.
- Retrieve `talu_last_error()` context before issuing other FFI calls that may overwrite thread-local error state.

### 2.4 Raw pointer exposure

- Public APIs should return safe Rust types, not raw pointers.
- Raw pointer escape hatches (`as_ptr`, `from_raw_*`) must be explicit and documented.
- `from_raw_owned`/similar constructors must be `unsafe` and state ownership transfer contract precisely.

---

## 3. Resource Lifecycle and Ownership

**Concept:** Ownership is explicit; cleanup is deterministic.

### 3.1 RAII is required

- Any type that owns a native allocation must implement `Drop`.
- `Drop` must safely no-op on null and never panic.
- Public constructors must return owning Rust wrappers, not bare raw pointers.

### 3.2 Owned vs borrowed handles

- Separate owned and borrowed handle types when needed (`Owned` vs `Ref` semantics).
- Borrowed wrappers must never free foreign-owned pointers.
- Ownership expectations must be clear in docs and type names.

### 3.3 Concurrency and `Send`/`Sync`

- Default stance: do not mark FFI-backed types `Send`/`Sync` unless C-layer guarantees are documented.
- Any `unsafe impl Send`/`unsafe impl Sync` requires:
  - explicit safety comment with C API threading assumptions
  - tests that exercise realistic cross-thread usage

---

## 4. Error Semantics

**Concept:** Errors are part of API contract; context must survive boundary crossing.

### 4.1 Library error contract (`talu`)

- All fallible public operations return `Result<T, talu::Error>`.
- Prefer specific error variants and stable messages over opaque "failed" text.
- Preserve C-side error context using helpers like `error_from_last_or`.

### 4.2 No panic for recoverable failures

- Forbidden for runtime paths: `unwrap()`, `expect()`, `panic!()` on user input, model state, IO/network, or C return codes.
- Allowed in tests/benchmarks or for unreachable programmer invariants (`unreachable!`) with clear rationale.

### 4.3 CLI error handling

- `cli` may use `anyhow` for top-level command ergonomics.
- At API boundaries with `talu`, preserve source error context (do not replace with untyped generic text).
- User-facing error text should remain actionable and stable.

---

## 5. Public API Discipline (`talu`)

**Concept:** Exported symbols are contractual; internal refactors must not leak.

- Treat `talu/src/lib.rs` exports as stable public contract.
- Public API changes require:
  - regression tests (failing before change)
  - docs updates in the same change
  - migration notes for behavior changes
- Keep APIs idiomatic Rust:
  - ownership and borrowing explicit
  - avoid hidden global mutable state
  - avoid surprise side effects in getters

### Contract alignment with core

- Rust wrappers must preserve C API meaning (error conditions, lifecycle behavior, state transitions).
- Do not repurpose or reinterpret core errors without updating tests/docs.

---

## 6. CLI Layer Rules (`cli`)

**Concept:** CLI is presentation and orchestration, not duplicate domain runtime.

- Command parsing, formatting, and terminal UX belong in `cli`.
- Business logic duplicated from core or shared binding surfaces does not belong in `cli`.
- Machine-readable/script output contracts must be deterministic.
- Interactive UX must degrade clearly in non-TTY/script contexts.

### Temporarily disabled/hidden commands

- If a command is intentionally disabled, keep behavior explicit:
  - remove or gate command at parser/dispatch surface
  - keep internal code only when explicitly intended
  - document status in code comments and tests where applicable

---

## 7. Testing Strategy

**Concept:** Rust tests define wrapper and CLI contracts; core tests define domain internals.

### 7.1 Required testing behavior

- Every new public function/type in `talu` or command behavior in `cli` must add or update tests.
- Bug fixes must include regression tests that fail before the fix.
- Tests must be deterministic:
  - no sleeps/timeouts for correctness synchronization
  - timeouts only as deadlock guards

### 7.2 Recommended gates for Rust changes

Run relevant scopes for touched crates:

```bash
cargo test -p talu --lib
cargo test -p talu-cli --lib
```

When touching formatting/lints:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features
```

If C ABI changed, also run:

```bash
zig build gen-bindings-rust
```

### 7.3 Test scope guidance

- Unit tests for wrapper invariants (error mapping, ownership, conversion helpers).
- Integration tests for end-to-end wrapper behavior over real C library surface.
- CLI tests for command parsing/dispatch/output contracts.
- Do not rely on network or external mutable state unless the test is explicitly marked and isolated.

---

## 8. Dependency Hygiene

**Concept:** Keep runtime dependency footprint minimal and justified.

- `talu-sys`: generated FFI and link setup only.
- `talu`: keep dependencies small and directly justified by public API needs.
- `cli`: CLI/server UX dependencies allowed when they do not leak into `talu` public runtime.
- Do not add crates that duplicate capabilities already provided by core/bindings unless there is a clear, documented reason.

Any new runtime dependency should include:
- purpose and why std/core cannot cover it
- scope (which crate uses it and why)
- maintenance impact

---

## 9. Change Hygiene and Sync Rules

- Keep core C API changes and Rust binding updates atomic in one PR.
- If C signatures/semantics changed:
  - regenerate `talu-sys`
  - adapt `talu` wrappers
  - adapt `cli` behavior if affected
  - update tests and docs
- Never land "temporary drift" between layers.

---

## Developer Checklist

Before marking complete:

**Architecture**
- [ ] Change is in the right crate (`talu-sys` vs `talu` vs `cli`)?
- [ ] No duplicated core domain logic in Rust layers?
- [ ] CLI consumes `talu` APIs, not raw FFI?

**Safety and Lifecycle**
- [ ] Unsafe blocks have concrete `// SAFETY:` justifications?
- [ ] Owned native pointers are wrapped and dropped deterministically?
- [ ] Ownership transfer and pointer lifetime rules are explicit?

**Errors and Contracts**
- [ ] Fallible APIs return `Result`, not panic paths?
- [ ] C errors preserve `talu_last_error()` context?
- [ ] Public API behavior changes covered by tests and docs?

**Quality**
- [ ] Relevant Rust tests run for changed crates?
- [ ] Formatting/lint gates run where applicable?
- [ ] C ABI changes regenerated and synchronized?
