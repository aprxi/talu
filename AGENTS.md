# Job description

*Goal: Feature velocity and functional correctness.*

*   **Tasks:** Add features, fix bugs, test
*   **Mindset:** "Does it work? Does it break anything? Did I write tests? Does it follow repository policy?"
*   **Responsibilities:**
    1.  **Build:** Build library (`zig build release -Drelease`).
    2.  **Test (Runtime):** Run unit/integration tests to verify logic (`make test`, `pytest`).
    3.  **Test Compliance (Existence):** Ensure every new function/class has a corresponding test file.

## Doing a task

Before you change anything:
* Read this file `AGENTS.md`.
* Read the nearest area policy for the files you will touch:
  * `bindings/python/` → `bindings/python/POLICY.md`
  * `core/src/` → `core/POLICY.md`

Then:
1. Confirm scope (Python / Zig / docs) and the correct entrypoints.
2. Make the smallest correct change. Do not add legacy/compat paths.
3. If behavior or public surface changes, update tests and docs in the same change.
4. Run the gate(s) and any area-specific checks.
5. Final pass: verify policy “musts” (lifecycle/safety, errors, tests, docs).


## Repository Policy (baseline)
These baseline rules are mandatory for all agents and contributors working in this repo.

### 1) Clarity is a feature

* Use explicit control flow, precise names, and locally-verifiable invariants (no hidden global assumptions).
* Make correctness/safety assumptions explicit (ownership, lifetime, concurrency expectations, input formats, error semantics).
* Comments explain *why* (intent, hazards, invariants), not "per doc X".
* Fight boilerplate: keep recurring guidance at module/package boundaries; don't scatter rules in-line.

### 2) Correctness and future-proof APIs

* No shortcuts: if we can't do it correctly, we don't merge it.
* Build for long-term correctness and performance: clear contracts, simple designs, and changes that are easy to audit.
* The user-facing public API is the bindings' exported surface; treat it as a contract.
* Public API changes require regression tests (would fail before) and user-facing docs updated to match.
* Design to prevent misuse: explicit lifecycle/ownership, typed errors on misuse, predictable state transitions.
* Tests define the contract: change behavior intentionally; never weaken tests to "make CI green".
* Inline docs (e.g., docstrings) are contract inputs: keep them accurate, language-idiomatic, and aligned with tests.

### 3) One correct path (no new legacy)

* Prefer deletion over divergence: provide one correct way; remove old paths rather than accumulate alternatives.
* Internal legacy is inexcusable: do not add compat/legacy paths for internal interfaces.
* Deprecation/compatibility applies only to external public APIs (bindings): centralized, owned, and removal target defined.

### 4) Gates and change hygiene

* Run the relevant lint/test gate first; a failing gate blocks review conclusions. Audits are qualitative inspection only (tests are CI/maintainer scope).
* Stay aligned: land changes atomically (core + affected bindings + tests + docs together), with no partial migrations/drift.
* Automate relentlessly: anything that can be enforced by gates/lint/tests should be.
* CI is required to detect interface drift.

### 5) Test rigor and determinism

* Tests can't use timing as synchronization for correctness (sleep/timeouts/polling). Timeouts are permitted only as deadlock guards, and the test must pass without relying on the timeout.
* Flaky tests are broken tests: remove nondeterminism; don't paper over with sleeps/retries/timeouts as synchronization.
* Bugfixes add regression coverage: include a test that fails before the fix.
* Use debugging as an opportunity: turn validated hypotheses into focused deterministic regression guards when they represent realistic boundary/lifecycle/error risks.
* Fix root causes, not symptoms.

### 6) Clean API interfaces

* Core owns domain logic; boundary layers are glue only (validate/convert/forward).
* Separate I/O/network from parsing/transform so parsing/transform is fully testable.
* `core/src/capi/` is a strict boundary and internal to our bindings (not a public ABI promise).
* No drift: C API signature/behavior changes must update all affected bindings **and tests** in the same change.
* Error semantics must remain stable across the boundary: don't repurpose error codes; update bindings/tests when meanings change.
* Contracts stay aligned: code + tests define behavior; docs must match.

### 7) Safety & lifecycle discipline

* Keep resource lifecycles pristine: ownership/lifetime must be explicit; resource release is required to be deterministic and idempotent.
* Cleanup must be explicit; finalizers are a safety net, no primary cleanup path.
* Errors must be diagnosable: preserve what failed, why, and key context; don't replace specifics with generic failures without cause.
* Use-after-close is a hard failure: it must reliably raise a typed error; close must be idempotent.

### 8) External public APIs (bindings)

* Treat as a contract (tests + docs track behavior) and as stable and versioned.
* Breaking changes require a deprecation/migration plan (with tests proving deprecation behavior).
* Concurrency docs only where user expectations change (shareable resources, sessions, streams).
* Internal refactors must not leak into public docs; docs describe usage and guarantees.

### 9) Shipping bar

* Ship for a real user job; must fit project direction.
* Prefer exposing capabilities the underlying system already supports when it materially helps users.
* Don't ship features with disproportionate complexity/risk/maintenance for marginal value.
* Every shipped feature needs a clear contract (tests + docs) and deterministic behavior.

### 10) Dependency hygiene (all bindings)

* Zero runtime dependencies for distributed bindings is a requirement.
* Build-time/dev deps are allowed; compiled/link-time deps are only acceptable when they do not become runtime dependencies.


# Build
Build & test entrypoints for canonical commands. Do not guess.

## Core (Zig)

    zig build release -Drelease           # build library + CLI + copy to Python
    zig build test -Drelease              # run unit tests
    zig build test-integration -Drelease  # run integration tests

See `core/POLICY.md` for test requirements and conventions.

## Bindings

### Python

Run from `bindings/python/`:

    uv sync                                              # install deps
    uv run pytest tests/<module>/                        # test a module
    uv run pytest tests/<module>/test_<file>.py          # test a file
    uv run pytest tests/ --ignore=tests/reference        # API tests
    uv run pytest tests/reference/                       # reference tests (vs PyTorch)
    uv run pytest tests/                                 # all tests (slow, use sparingly)

Match test scope to your changes. Changed `talu/chat/`? Run `tests/chat/`.

See `bindings/python/POLICY.md` for test requirements and conventions.

## UI

make -C ui && ./zig-out/bin/talu serve --html-dir ui/dist
