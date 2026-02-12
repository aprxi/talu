# Contributing

Welcome, you or your automations. Same standards.

## Start here

1. Read [AGENTS.md](AGENTS.md).
2. Follow the nearest area policy:
   - [core/POLICY.md](core/POLICY.md) — Zig/C API standards
   - [bindings/python/POLICY.md](bindings/python/POLICY.md) — Python standards
3. Area policy rules override global defaults.

## What a good PR looks like

A good PR explains the **why**.

* State the problem clearly and why it matters (user impact, correctness/safety, contract/expectation mismatch, maintenance pain).
* Describe the intended use case(s). If it's a bug: describe the expected behavior and how the current behavior violates it.
* Point back to the relevant default(s) when that helps frame the change.
* Ship it, own it.

All automated gates (linters, quality checks, tests) must pass. If a gate is wrong, fix the gate in a separate PR.
