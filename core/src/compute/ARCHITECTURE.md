# Compute Architecture & Policy
> **Status:** Enforced  
> **Scope:** `core/src/compute/`

Backend-specific policy addenda:
- CUDA: `core/src/compute/cuda/POLICY.md`

## 1. The Core Philosophy
`core/src/compute` is a **pure mathematical, memory-layout, and hardware-abstraction library**. 

It is strictly a tensor math engine, **not** an LLM engine. It has absolutely no knowledge of Large Language Models, text generation, prompts, model families (e.g., Llama, Qwen), or inference orchestration.

**The Golden Rule:** If you can't use a kernel in an audio-processing pipeline or a computer vision classifier without changing its name or signature, it does not belong in `compute`.

### Separation of Concerns
*   **`models/` owns Policy & Topology:** Interprets `config.json`, calculates heuristic values (e.g., YaRN inverse frequencies), and defines the sequence of operations (the graph).
*   **`inference/` owns Orchestration:** Manages the KV cache lifecycle, schedules batches, handles the decode loop, and executes the graph.
*   **`compute/` owns FLOPs & Bytes:** Executes raw matrix multiplications, vector additions, SIMD instructions, and hardware-specific lazy graph constructions (e.g., Metal/MLX).

---

## 2. The Strict Boundary

To prevent architectural rot and entanglement, `compute` operates under strict guardrails, enforced by CI linting (`core/tests/helpers/lint/root.zig`).

### Dependency Isolation
*   `compute` **MUST NOT** import anything from `models/` or `inference/`.
*   `compute` **MUST NOT** import `protocol/` or `responses/`.
*   `inference` **MUST NOT** bypass the boundary; it may only import `compute` via `compute/root.zig` (no deep reaching into `compute/cpu/memory.zig`).

### Vocabulary & Naming Conventions
Domain-specific language leaks context and restricts reusability. Computations must be described in geometric or mathematical terms.

| Forbidden (LLM Domain) | Allowed (Math/Geometry) | Example in Codebase |
| :--- | :--- | :--- |
| `token`, `token_id` | `index`, `position`, `row` | `scatterRowsByMatchedId` |
| `vocab_size` | `out_features`, `dim_size` | `n_cols` |
| `prompt` | `sequence`, `matrix` | `input_matrix` |
| `repetition_penalty` | `index_penalty` | `applyIndexPenalty` |
| `logit_bias` | `index_bias` | `applyIndexBias` |
| `Llama3`, `YaRN` | (None, precompute upstream) | `initFromInvFreq` |
| `MoE`, `SSM`, `Mamba` | `sparse_gating`, `state_scan` | `stateScanF32` |

*(Note: Minor exceptions exist inside legacy C++ interop boundaries or specific inference-pipeline adapters, but all new/pure compute code must adhere to the math vocabulary).*

### The Metal / GPU Boundary
For hardware backends like Apple Silicon (Metal/MLX), `compute` **does not** define fused model topologies. 
*   `compute/metal/graph.zig` exposes a generic, lazy computation graph API (e.g., "Add", "Matmul", "Reshape").
*   The actual construction of a "Fused Transformer" lives entirely in `inference/backend/metal/`. 

### Tensor Metadata And Layout Contract
`compute/tensor_desc.zig` is the canonical compute-local path for tensor metadata validation:

*   rank must be `1...8`
*   active dimensions and generic strides must be positive
*   inactive shape and stride slots must be zero
*   logical element counts, dense byte counts, and strided physical byte spans use checked arithmetic
*   `Tensor.data_size` is physical storage bytes, not an implicit dense byte count

The compute layout vocabulary is intentionally small:

| Layout | Meaning |
| :--- | :--- |
| `row_major_contiguous` | Dense C-order storage with canonical strides |
| `strided` | Inspectable storage with explicit positive element strides |
| `opaque_backend` | Backend-native storage that generic byte/span validators cannot inspect |

Generic byte validators reject `opaque_backend`. Backend-native paths must opt into explicit backend capability checks instead of claiming dense byte compatibility.

Block-quantized dtypes such as grouped-affine and MXFP4 do not have a generic dense element byte count. Callers must validate declared physical storage bytes explicitly.

### Backend Capability Contract
`compute/capability.zig` defines static query types for backend, primitive name, primitive input dtype, primitive output dtype, layout, raw byte-copy devices, typed copy direction, cast pair, rank limits, and alignment requirements. CPU, CUDA, and Metal capability modules publish table-driven facts for their current primitive, raw copy, typed copy, and cast surfaces.

Unknown support defaults to unsupported. Capability declarations must stay compute-generic: no model ids, layer ranges, stages, request state, scheduler slots, placement, transport, or orchestration concepts.

Primitive queries must provide both input and output dtype. Same-dtype primitives declare matching input/output sets; conversion, dequantization, matvec, and attention descriptors declare the source/storage dtype separately from the produced dtype so output mismatches fail closed.

CUDA primitive facts are derived from `compute/cuda/descriptors.zig`. Descriptors reference module-owned `op_name*` constants, and root tests audit both directions: every exported CUDA op name has exactly one capability entry, and every capability entry resolves through one descriptor-backed exported implementation.

Raw copy facts describe byte movement only. They do not imply dtype-specific copy kernels. Raw-copy validators derive the direction from source and destination `Device` values, then enforce backend/device constraints for CPU host copies, CUDA host/device and device/device copies, CUDA peer copies, and Metal host/device copies. Typed copy and cast helpers use `compute/copy_cast.zig` validators before destination mutation or kernel argument packing wherever signatures expose dtype, layout, element count, and buffer sizes. Unsupported copy directions, cast pairs, dtypes, layouts, devices, buffer sizes, and alignment mismatches fail with typed errors.

---

## 3. Testing & Micro-Benchmarking Strategy

Because `compute` is completely isolated from tokenizers, chat templates, and model file formats, it can and **must** be tested in complete isolation.

### Unit Testing
*   Tests in `compute` must instantiate raw arrays (`[]f32`, `
