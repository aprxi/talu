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

*(Note: Minor exceptions exist inside legacy C++ interop boundaries or specific inference-bridge adapters, but all new/pure compute code must adhere to the math vocabulary).*

### The Metal / GPU Boundary
For hardware backends like Apple Silicon (Metal/MLX), `compute` **does not** define fused model topologies. 
*   `compute/metal/graph.zig` exposes a generic, lazy computation graph API (e.g., "Add", "Matmul", "Reshape").
*   The actual construction of a "Fused Transformer" lives entirely in `inference/backend/metal/`. 

---

## 3. Testing & Micro-Benchmarking Strategy

Because `compute` is completely isolated from tokenizers, chat templates, and model file formats, it can and **must** be tested in complete isolation.

### Unit Testing
*   Tests in `compute` must instantiate raw arrays (`[]f32`, `
