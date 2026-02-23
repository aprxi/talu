# CUDA Compute Policy
> **Status:** Enforced  
> **Scope:** `core/src/compute/cuda/`

This policy defines how CUDA kernels are authored, built, named, loaded, and tested in this repository.

## 1. Design Goals

1. Keep zero runtime linker dependencies for CUDA (`libcuda` driver API only).
2. Keep compute code domain-agnostic (math/tensor only; no LLM semantics).
3. Keep one clear, auditable kernel path for each operation.
4. Keep artifacts modular and replaceable without changing core inference logic.

## 2. Runtime Dependency Rules

1. Do not link `libcudart`, `libcudnn`, `libflashattn`, or other CUDA runtime ML libraries into the distributed binary.
2. CUDA interaction must flow through driver API loading in `device.zig` and kernel/module loading in `module.zig`.
3. GPU handles (contexts, modules, function handles) must be owned by backend instances. No global CUDA state.

## 3. Kernel Artifact Strategy

1. Embedded base module: `core/assets/cuda/kernels.fatbin`.
2. Sideload module: architecture-specific `.cubin` payload + JSON manifest + SHA-256 verification.
3. Sideload cache location:
   Linux/macOS/Windows app-data path by default via `std.fs.getAppDataDir`.
   Override with `TALU_CUDA_CACHE_DIR`.
4. Remote sideload base URL is optional and configured by `TALU_CUDA_KERNEL_BASE_URL`.

## 4. Build Rules

1. `zig build gen-cuda-kernels` is the canonical kernel generation step.
2. Kernel module generation must be reproducible and checked in with source updates.
3. Kernel compilation entrypoint is `core/src/compute/cuda/kernels/kernels.cu`.
4. Kernel implementations should live under `core/src/compute/cuda/kernels/ops/` and be aggregated by the entrypoint.
5. Generated module file is `core/assets/cuda/kernels.fatbin`.

## 5. Naming Rules

1. CUDA kernel entry names must be stable C ABI symbols with `talu_` prefix.
2. Do not embed version suffixes like `_v1` in kernel symbol names.
3. Kernel evolution is tracked by git history and manifest ABI metadata, not symbol suffixes.
4. Zig wrappers must expose `embedded_module` and `embedded_symbol` names (not `embedded_ptx`).

## 6. Compute Boundary Rules

1. CUDA kernels and wrappers in `compute/cuda` may only express tensor math, memory layout, indexing, and numerics.
2. Forbidden concepts in this layer: tokenization, prompt handling, chat/session orchestration, model-family special cases.
3. Any model/topology policy belongs in `core/src/inference/` and must call compute primitives through explicit interfaces.

## 7. Safety and Lifecycle Rules

1. All module/buffer allocations must have explicit deterministic cleanup.
2. Error paths must preserve typed errors; do not hide CUDA driver failures behind generic errors.
3. SHA-256 verification is mandatory for sideloaded artifacts before use.
4. Kernel launch arguments must use the validated argument packer in `args.zig`; avoid ad hoc pointer packing.

## 8. Testing and Verification Rules

1. New CUDA wrapper modules require focused unit tests for argument validation and error behavior.
2. Build gate minimum for CUDA changes:
   `zig build gen-cuda-kernels -Dcuda=true`
   `zig build release -Drelease -Dcuda=true` (or `make cuda`)
3. Runtime smoke checks in CUDA backend init must remain deterministic and bounded.
   They are opt-in via build flag `-Dcuda-startup-selftests=true` (default: disabled).

## 9. Policy Compliance Checklist

Use this checklist in every CUDA compute change before review:

- [ ] Runtime dependency check: no new runtime link dependency beyond CUDA driver API (`libcuda`).
- [ ] Artifact check: changes keep `kernels.cu` as canonical compile entrypoint and `kernels.fatbin` as canonical embedded module.
- [ ] Source layout check: kernel logic lives in `kernels/ops/` modules, not a growing monolithic entrypoint file.
- [ ] Build command check: `zig build gen-cuda-kernels -Dcuda=true` succeeds.
- [ ] Build command check: `zig build release -Drelease -Dcuda=true` succeeds (or `make cuda` succeeds).
- [ ] Naming check: no kernel symbol ends with version suffixes (forbidden: `_v1`).
- [ ] Wrapper naming check: wrappers use `embedded_module`/`embedded_symbol` terminology.
- [ ] Compute boundary check: changed files in `compute/cuda` contain no inference-domain concepts (tokens/prompts/chat/model-family policy).
- [ ] Lifecycle check: all new module/buffer resources have deterministic cleanup on success and error paths.
- [ ] Sideload integrity check: sideloaded artifacts remain SHA-256 verified before load.
- [ ] Test check: new/changed public CUDA wrapper behavior has direct unit coverage for validation/error behavior.
