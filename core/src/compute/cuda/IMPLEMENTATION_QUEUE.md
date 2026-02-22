# CUDA Implementation Queue
> **Scope:** `core/src/compute/cuda/` and CUDA backend wiring in `core/src/inference/backend/cuda/`
> **Goal:** Correct, stable, maintainable CUDA execution path before micro-optimization.

## 1. Stabilize Typed Dense Matvec Contract (Completed)
- Split dense typed matvec kernel into dtype-specific entry points (`f16` and `bf16`).
- Remove runtime dtype branching from kernel argument contract.
- Keep one explicit launch path per dtype in backend wiring.

## 2. Canonicalize Dense Typed Weight Layout (Completed)
- Define one internal orientation/layout contract for dense typed weights.
- Enforce layout at upload/prepack boundary only.
- Eliminate implicit orientation assumptions in hot path code.

## 3. Harden Kernel Surface Initialization (Completed)
- Resolve required kernels during backend init with fail-fast typed errors.
- Keep required operation names and symbol mapping centralized and auditable.
- Validate sideload manifest ABI/version compatibility before module use.

## 4. Solidify Artifact Boundary (Completed)
- Keep canonical source at `core/src/compute/cuda/kernels/kernels.cu`.
- Keep canonical embedded artifact at `core/assets/cuda/kernels.fatbin`.
- Keep sideload artifacts manifest-driven and checksum-verified.

## 5. Tighten Launch and Arg Validation (Completed)
- Ensure each wrapper validates dimensions, sizes, and alignment-sensitive assumptions.
- Keep `ArgPack` as the only kernel launch argument path.
- Add explicit tests for wrapper argument rejection paths.

## 6. Clean Kernel Naming and ABI Hygiene (Completed)
- Keep stable C ABI symbol names with `talu_` prefix.
- No symbol version suffixes in names; use manifest ABI versioning.
- Keep operation names in registry/manifest aligned with wrapper names.

## 7. Remove Redundant Synchronization Boundaries (Completed)
- Keep synchronization only at correctness boundaries.
- Avoid unnecessary host/device barriers between independent launches.
- Preserve deterministic behavior for smoke and parity checks.

## 8. Expand Deterministic Smoke Coverage (Completed)
- Ensure each required kernel class has a deterministic smoke check.
- Keep smoke checks lightweight and bounded.
- Ensure init fails clearly if required kernel smoke fails.

## 9. Prepare Fusion Scaffolding (Correctness-First) (In Progress)
- Keep fused/unfused boundaries explicit and swappable in one place.
- Only enable fused paths after deterministic parity validation.
- Keep fused kernels math-only and layout-contract driven.
