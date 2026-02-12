# Python Policy

> This document is the Architectural Specification for `bindings/python/`.
> It translates [AGENTS.md](../../AGENTS.md) into enforceable Python patterns.

**Scope:** `bindings/python/` directory.

---

## ⚡ Quick Invariants (The Non-Negotiables)

*If you violate these, the architecture breaks. Do not merge.*

1.  **Zero Dependencies:** Production code (`talu/`) imports **only** the stdlib. Type-only deps live in `TYPE_CHECKING` blocks.
2.  **Native Containment:** `ctypes` is confined to `_native.py` (auto-generated) and `_bindings.py` (with `Justification:` docstring). Never in logic files.
3.  **Explicit Lifecycle:** Every native-pointer wrapper implements `close()` (idempotent), `__enter__`/`__exit__`, and `__del__` (fail-safe).
4.  **No Logic in Async:** Async classes are thin I/O wrappers. Core logic lives in sync helpers. Blocking FFI uses `run_in_executor`.
5.  **Deterministic Tests:** No `time.sleep()`. No timeouts for correctness (only as `# DEADLOCK_GUARD` fail-fast). No network in default tests.
6.  **One Correct Path:** No `legacy_mode` flags. Deprecate → warn → delete. Never parallel paths.
7.  **Typed Errors:** Map Zig codes to specific exceptions via `check()`. Never **plain** `Exception` or **plain** `RuntimeError`.
8.  **Strict Boundaries:** Config flows down (Python → C). Errors flow up (C → Python). Python never shadows Zig state.

**Anti-circumvention:** These rules define architectural invariants, not lint targets. Do not bypass them through indirection (aliases, wrappers, dynamic imports, conditional branches, or relocation of code) to "make the checker pass." If a change appears to require violating an invariant, stop and escalate to maintainers with the concrete constraint you hit.

---

## 1. Zero-Dependency Architecture

**Concept:** The binding is infrastructure. It runs anywhere Python 3.10+ runs, instantly.

| Scope | Allowed Imports | Purpose |
| :--- | :--- | :--- |
| `talu/` | stdlib only | Production. **No runtime deps.** |
| `if TYPE_CHECKING:` | `typing_extensions`, etc. | Type hints only, zero runtime cost. |
| `tests/` | stdlib, pytest | API contract tests. |
| `tests/reference/` | torch, numpy, transformers | Numerical correctness vs. reference implementations. |

**Rules:**
- No third-party imports in `talu/` — not at module level, not inside functions. "stdlib only" means everywhere.
- Support third-party inputs via duck typing (Buffer Protocol, DLPack), not imports.
- Optional features that need numpy/torch belong in `tests/` or user code, not `talu/`.

### Prohibited Patterns

- **No module-level mutable state.** Constants are allowed; mutable globals are not.
- **No wildcard imports.** `from module import *` is forbidden.
- **No broad exception catches.** `except Exception:` and bare `except:` are forbidden in library code, **except inside `__del__` only**.

---

## 2. Native Boundary (Python ↔ Zig)

**Concept:** Python orchestrates; Zig computes. The FFI layer is strictly contained and always in sync.

### Atomic Sync Guarantee

When `core/src/` changes, bindings update atomically:
- `zig build gen-bindings` regenerates `_native.py` with all struct definitions and signatures.
- Python bindings **must** fully comply immediately — no "catch up later."
- CI enforces this via tests and auditing. Drift is a merge blocker.

### File Roles

| File | Role | Constraints |
| :--- | :--- | :--- |
| `_native.py` | Struct definitions, signatures | Auto-generated (`zig build gen-bindings`). Do not edit. |
| `_bindings.py` | Library loading, error handling, manual FFI | Requires `Justification:` in docstring. Contains `ERROR_MAP` and `check()`. |
| All others | Business logic | Must not import `ctypes`. |

**Exception:** `ctypes` imports inside `if TYPE_CHECKING:` blocks are allowed for annotations.

**Violation codes:**
| Code | Description |
|------|-------------|
| `missing-justification` | `_bindings.py` without `Justification:` in docstring |
| `ctypes-import` | ctypes import in non-allowed file |
| `ctypes-structure` | Structure/Union definition in non-allowed file |
| `ctypes-cfunctype` | CFUNCTYPE in non-allowed file |
| `ctypes-library-load` | CDLL in non-allowed file |
| `ctypes-signature` | argtypes/restype in non-allowed file |

**No pass-through wrappers:** Do not write trivial wrappers in `_bindings.py` that just call a C function and return the result. If the C API needs a friendlier signature, fix it in Zig and regenerate `_native.py`.

### Error Handling

Zig returns `i32` codes. Python checks every non-zero return code immediately via `check()`.

```python
# CORRECT: Atomic check
code = lib.talu_create_session(config)
check(code)  # Raises typed exception, clears thread-local error

# WRONG: Logic between call and check
code = lib.talu_create_session(config)
log.debug("created")  # Thread-local error state may be corrupted
check(code)
```

### Error Code Mapping (Strict One-Path)

Every Zig error code **must** be explicitly mapped in `_bindings.py:ERROR_MAP`. No range-based fallback.

**Normative source:** `_bindings.py:ERROR_MAP` is the source of truth. The table below is illustrative, not exhaustive.

**Zig-originated errors** (examples from `ERROR_MAP`):

| Code | Exception | String Code |
| :--- | :--- | :--- |
| 100 | `ModelNotFoundError` | `MODEL_NOT_FOUND` |
| 900 | `MemoryError` (builtin) | — |
| 901 | `ValidationError` | `INVALID_ARGUMENT` |
| 902 | `ValidationError` | `INVALID_HANDLE` |
| 999 | `TaluError` | `INTERNAL_ERROR` |

See `_bindings.py:ERROR_MAP` for the complete mapping.

**Python-only errors** (no Zig code, `.original_code` is `None`):

| Exception | Use Case |
| :--- | :--- |
| `StateError` | Use-after-close, invalid object state (Python-side lifecycle misuse) |
| `InteropError` | DLPack/NumPy interchange failures |

`StateError` must never be raised from Zig return codes; it is reserved for Python-side state validation (e.g., use-after-close). Zig handle/state violations map to `ValidationError` and preserve `.original_code`.

**Unmapped code = bug:**
- If `check()` receives a code not in `ERROR_MAP`, it raises `TaluError` with `.code="UNMAPPED_ERROR"` and preserves `.original_code`.
- This signals drift between Zig and Python — a merge blocker.
- `UNMAPPED_ERROR` must not be used as a general-purpose error; it exists only to signal drift.
- Fix by adding the mapping to `ERROR_MAP` + adding a test.

### Exception Attributes

All `TaluError` subclasses expose:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `.code` | `str` | Stable string code for programmatic handling (`"MODEL_NOT_FOUND"`). |
| `.original_code` | `int \| None` | Zig integer code. `None` for Python-only errors. |
| `.details` | `dict` | Structured context (includes `zig_code` key for Zig errors). |

**Invariant:** Never raise plain `Exception` or plain `RuntimeError`. Always use typed exceptions that inherit from `TaluError`.

---

## 3. Resource Lifecycle

**Concept:** We manage raw memory. GC is a safety net, not a cleanup mechanism.

### The "Holy Trinity" Pattern

Every native-pointer wrapper must implement:

1. **`close()`** — Frees the pointer, sets it to `None`. Idempotent (safe to call multiple times).
2. **Context Manager (`__enter__`/`__exit__`)** — `__enter__` returns `self`; `__exit__` calls `close()`.
3. **`__del__`** — Calls `close()` as fail-safe. Must not raise (wrap in `try/except`).

### Safety Rules

- **Use-After-Close:** Methods on closed objects raise `StateError(code="STATE_ERROR")`.
- **Copying:** Raise `TypeError` in `__copy__` and `__reduce__` unless explicitly supported.
- **Double-close:** `close()` must be safe to call multiple times (idempotent).

---

## 4. Async & Sync Discipline

**Concept:** Two API surfaces (`Client`, `AsyncClient`) powered by shared logic.

| Surface | Implementation |
| :--- | :--- |
| Sync (`Client`) | Blocking FFI calls directly. |
| Async (`AsyncClient`) | `async def` + `loop.run_in_executor()` for blocking FFI. |

**Rules:**
- Core logic (validation, config parsing) lives in **sync helpers**. Async wraps them.
- Async methods **must not** call blocking FFI directly (blocks event loop).
- Naming: `Async` prefix (`AsyncClient`, `AsyncChat`, `AsyncStreamingResponse`).

---

## 5. Public API & Stability

**Concept:** `__all__` is the contract. Everything else is internal.

### Definition

- **Public:** Symbols in `talu.__all__` or `talu.<subpackage>.__all__`. Stability guarantees apply.
- **Internal:** `_`-prefixed or not in `__all__`. Can change without notice.

**Re-exports are forbidden.** A public symbol must be defined in its canonical module. Do not import-and-export symbols to provide alternate import paths (including `__init__.py` aggregators). If a path change is required, it is a breaking change and the old path must be removed.

**No unauthorized compatibility layers.** Agents must not introduce compatibility layers, alternate public paths, or deprecation scaffolding unless explicitly directed by maintainers for a specific issue/PR.

### Deprecation Lifecycle

```python
warnings.warn(
    "old_function() is deprecated. Use new_function(). Removal in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Deprecation checklist:**
- [ ] Warning includes **replacement path** and **removal version**.
- [ ] Uses `stacklevel=2` (points to caller, not internal code).
- [ ] Test asserts the warning is emitted.
- [ ] Docstring updated with deprecation note.

**Timeline:** Deprecated in vX.Y → survives vX.Y+1 → removed in vX+1.0.

**Isolation:** Complex legacy wrappers go in `talu/_compat.py` (internal, not exported). Legacy markers allowed **only** if: warning emitted + issue referenced + isolated in `_compat.py` + tests assert warning.

### Documentation

Public symbols require Google-style docstrings: Summary, Args, Returns, Raises, Example.

**Content boundaries:**
- Describe behavior, inputs, outputs, errors.
- Do NOT document Zig internals, memory layout, or implementation details.

**Concurrency sections** only for: `Client`, `AsyncClient`, `Tokenizer`, `Router`, `Chat`, `AsyncChat`, `StreamingResponse`, `AsyncStreamingResponse`. Omit for immutable value objects.

---

## 6. Testing Strategy

**Concept:** Separate logic verification (API tests) from numerical validation (reference tests).

### Test Suites

| Suite | Path | Python | Deps | Focus |
| :--- | :--- | :--- | :--- | :--- |
| API | `tests/` (excl. `reference/`) | 3.10+ | None | State, errors, lifecycle, config. |
| Reference | `tests/reference/` | latest | torch, numpy | Numerical correctness vs. PyTorch. |

### Structure (Mirroring)

Tests **must** mirror source: `talu/chat/session.py` → `tests/chat/test_session.py`.

Each `tests/<module>/__init__.py` must exist and include a docstring containing `Maps to:`.

### Determinism

| Forbidden | Use Instead |
| :--- | :--- |
| `time.sleep()` | `threading.Event`, `queue.Queue` |
| `event.wait(timeout=X)` for logic | Blocking `wait()` or redesign |
| `thread.join(timeout=X)` for logic | `join()` without timeout |

**Deadlock guards allowed:** `join(timeout=30)` + `if alive: pytest.fail("hung")` — but test must pass without the timeout. Mark with `# DEADLOCK_GUARD` comment for auditability.

### Robustness Tests

- Resource cleanup: 50+ iteration loops + `gc.collect()`.
- Partial iteration: break early, verify cleanup.
- Exception recovery: raise mid-operation, verify resources freed.

### Numerical Correctness

- Never compare floats with `==`.
- API tests: use `pytest.approx()`.
- Reference tests: use `torch.testing.assert_close()`.
- Reuse tolerances from `tests/conftest.py` (`float32_tolerance`, `matmul_tolerance`, `quantized_tolerance`).

### Markers

| Marker | Meaning | CI Behavior |
| :--- | :--- | :--- |
| `@pytest.mark.slow` | Long-running | Included; skip with `-m "not slow"` |
| `@pytest.mark.network` | Requires internet | Excluded from default CI |
| `@pytest.mark.gpu` | Requires GPU | Skipped on CPU-only |
| `@pytest.mark.requires_model` | Needs downloaded model | Skipped if unavailable |

### xfail Governance

```python
@pytest.mark.xfail(reason="Edge case #456, fix by v1.5", strict=True)
def test_edge(): ...
```

**Rules:**
- Reason must reference issue number.
- Must have target version or review date.
- Use `strict=True` (unexpected pass = signal).
- Stale xfails (>90 days) must be resolved or escalated.

### Skip Rules

- `@pytest.mark.skip` forbidden for bypassing failures.
- Allowed only for platform conditionals: `@pytest.mark.skipif(sys.platform == "win32")`.

### Coverage Pragmas

`# pragma: no cover` must not be used on branches that perform meaningful behavior (resource frees, state transitions, returns, error mapping). It is only acceptable for provably unreachable code and must include a short invariant note explaining why it's unreachable.

**Allowed:**
```python
if ptr is None:  # pragma: no cover - C API guarantees non-null on success
    raise TaluError("unexpected null", code="INTERNAL_ERROR")
```

**Forbidden:**
```python
if ptr is None:  # pragma: no cover
    return None  # This is API behavior, not unreachable code
```

**Cleanup rule:** If cleanup is required, structure the code so cleanup occurs on a path that is covered by normal execution, not hidden behind an untested guard.

**NULL pointer handling:** Success from C API must not produce NULL. If a defensive NULL check is needed, treat it as an internal error (raise typed exception), not a valid return value. Returning `None` is an API-level semantic decision that must be tested; raising a typed error is a safety decision for provably impossible conditions.

---

## 7. Domain Rules

### ModelSpec

- **Flow:** `ModelSpec` → `normalize_to_handle()` → `talu_backend_create()`.
- Router converts Python objects to C structs. No string re-parsing.
- String model arguments are deprecated; use `ModelSpec`.

### Multimodal Content

- Text: plain `str`.
- Media: `InputImage.from_file()`, `InputAudio.from_file()` → dict with `type`, `data`, `mime`.

### Logging

- Use `talu._logging.logger`. No `print()`.
- Include `extra={"scope": "..."}` for structured context.

---

## Developer Checklist

Before marking complete:

**Architecture:**
- [ ] No new runtime dependencies in `talu/`?
- [ ] `ctypes` only in `_native.py` or `_bindings.py`?
- [ ] Async/sync logic shared, not duplicated?

**Safety:**
- [ ] `close()` idempotent (safe to call twice)?
- [ ] Every FFI return code checked immediately?
- [ ] Errors are typed (`ModelError`), not generic?

**Correctness:**
- [ ] Test added that would fail on old behavior?
- [ ] Test mirrors source structure?
- [ ] Test is deterministic (no sleeps)?

**Contract:**
- [ ] New public symbol in `__all__`?
- [ ] Docstring has Args, Returns, Raises, Example?
