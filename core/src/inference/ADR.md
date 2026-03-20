# Inference ADR

Status: Accepted  
Owners: Core Inference + Models + Compute  
Decision Type: Architectural

## Normative Language
This ADR uses only normative terms:
- `MUST`
- `MUST NOT`
- `REQUIRED`

Any statement using these terms is part of the architectural contract.

## Scope
This ADR applies to:
- `core/src/models/**`
- `core/src/inference/**`
- `core/src/compute/**`

This ADR governs text and multimodal routes that execute `ExecutionPlan` instructions.

## Ownership Boundaries
`models/` MUST own:
- architecture metadata
- plan compilation
- instruction-local binding metadata for weights, params, and state

`inference/` MUST own:
- scheduler lifecycle
- register/state routing and validation
- adapter-table orchestration

`compute/` MUST own:
- math kernels
- fused kernel implementations

`compute/` MUST NOT own scheduler policy or model-family orchestration.

## Normative Contract Types
The following types are REQUIRED runtime contracts. Their semantics MUST match `core/src/inference/runtime_contract/types.zig`.

```zig
pub const Opcode = enum(u8) { ... };
pub const RegisterRef = enum(u16) { _ };

pub const WeightRef = struct { index: u32 };
pub const WeightBinding = struct { index: u32, name: []const u8 };

pub const StateLifecycle = enum(u8) {
    slot_persistent,
    request_scoped,
    step_scoped,
};

pub const StateDescriptor = struct {
    id: u8,
    size_bytes: u64,
    align_bytes: u16,
    zero_init: bool,
    lifecycle: StateLifecycle,
    runtime_kind: u8,
};

pub const Instruction = struct {
    opcode: Opcode,
    inputs: []const RegisterRef,
    outputs: []const RegisterRef,
    weights: []const WeightRef,
    param_block_id: ?u16,
    state_block_id: ?u8,
};

pub const ExecutionPlan = struct {
    instructions: []const Instruction,
    register_count: u16,
    state_descs: []const StateDescriptor,
};

pub const LivenessMap = struct {
    register_last_read: []const u32,
    kill_after_instruction: []const []const u64,
};

pub const PhysicalBufferSpec = struct {
    size: usize,
    @"align": u16,
    dtype: DType,
    layout: RegisterLayout,
};

pub const CompiledPlan = struct {
    plan: ExecutionPlan,
    param_blocks: []const ParamBlock,
    weight_bindings: []const WeightBinding,
    register_buffer_specs: []const PhysicalBufferSpec,
    liveness: LivenessMap,
    peak_registers: u16,
    diagnostics: []const PlanDiagnostic,
};

pub const ModelPlans = struct {
    vision_encode: ?CompiledPlan,
    scatter: ?CompiledPlan,
    decoder_prefill: ?CompiledPlan,
    decoder_decode: ?CompiledPlan,
};

pub const PhysicalMapping = struct {
    register_to_physical: []const u16,
    physical_count: u16,
    physical_specs: []const PhysicalBufferSpec,
};

pub const TensorHandle = struct {
    register: RegisterRef,
    ptr: *anyopaque,
};

pub const TensorViewDesc = struct {
    dtype: DType,
    rank: u8,
    shape: [4]u32,
    stride_elems: [4]u32,
    layout: TensorLayout,
};

pub const StateBlockHandle = struct {
    id: u8,
    ptr: [*]align(64) u8,
    size: u64,
    align_bytes: u16,
};

pub const ParamBlock = struct {
    version: u8,
    opcode: Opcode,
    data: []align(8) const u8,
};

pub const ExecutionContext = struct {
    mode: ExecutionMode,
    active_slots: []const usize,
    sequence_lengths: []const u32,
    batch_size: usize,
    stream_or_queue: ?*anyopaque,
    dispatch_counters: ?*DispatchCounters,
    workspace: Workspace,
};

pub const KernelAdapterFn = *const fn (
    ctx: *ExecutionContext,
    insn: *const Instruction,
    registers: []TensorHandle,
    register_views: []const TensorViewDesc,
    state_blocks: []StateBlockHandle,
    params: []const ParamBlock,
) anyerror!void;

pub const AdapterTable = [256]?KernelAdapterFn;

pub const AdapterCapability = struct {
    supports_batch: bool,
    supports_graph_emit: bool,
    max_batch_size: ?usize,
};
pub const AdapterCapabilities = [256]AdapterCapability;
```

## Opcode Policy
1. Macro-op opcode values in use are `0..31`.
2. Opcode range `32..63` is RESERVED for future macro-ops.
3. Primitive compatibility opcode values in use are `64..84`.
4. Opcode range `85..127` is RESERVED for future primitives/extensions.
5. Opcode range `128..255` is RESERVED for backend-local or experimental opcodes.
6. The adapter table index MUST be `@intFromEnum(Opcode)` and MUST use a `[256]?KernelAdapterFn`.
7. The runtime MUST reject unsupported opcodes at load/validation time.

## ParamBlock ABI Contract
1. `ParamBlock.data` MUST be little-endian payload bytes.
2. `ParamBlock.data` MUST be naturally aligned to 8 bytes.
3. `ParamBlock.data.len` MUST be `<= 256` for v1 ABI.
4. ParamBlock ABI version MUST be validated during plan load before execution.
5. Execution hot paths MUST NOT parse strings, allocate, or branch on ABI version.
6. Adapter-side decode MUST use comptime-known typed casts.

## Batching and ExecutionContext Contract
1. Adapters MUST execute batched work for all active slots in `ExecutionContext`.
2. Hot execution loops MUST NOT perform per-slot host orchestration loops that bypass adapter batching semantics.
3. `batch_size` MUST equal `active_slots.len`.
4. `sequence_lengths.len` MUST equal `active_slots.len`.
5. `batch_size == 0` MUST fail with typed invalid-batch error.
6. Adapter capability checks MUST enforce `supports_batch` and `max_batch_size` before execute.

## Backend Execution Semantics
CPU backend:
1. CPU execution semantics are synchronous.
2. CPU adapters MUST treat `TensorHandle.ptr` as CPU tensor payloads for CPU kernels.

Metal backend:
1. Metal execution MUST use graph-emit adapter semantics for layer program execution.
2. Graph evaluation boundaries MUST be explicit at route/stage boundaries.
3. Vision staged routes MUST honor explicit sync boundaries defined by the runtime.

CUDA backend:
1. CUDA execution MUST use stream-based semantics through `ExecutionContext.stream_or_queue`.
2. A single-slot request MUST execute as batch size `1`.
3. Batch capability validation MUST enforce any CUDA batch-size restrictions.

Mixed-backend staged topologies (`cpu+gpu`, `gpu+gpu`, `cpu+gpu+gpu` roadmap):
1. Topology capability/split validation MUST complete successfully before backend runtime initialization starts.
2. Stage-boundary activation dtype/layout MUST be negotiated from stage-advertised support sets.
3. Any boundary conversion MUST be explicit and accounted for in the selected topology plan; implicit widening/narrowing in hot paths is forbidden.
4. KV/state blocks MUST remain backend-local to the stage that owns the executed layer range.
5. Inter-stage KV/state migration over PCIe MUST NOT occur in v1 staged routes.
6. Slot lifecycle operations (alloc, bind, reset, unbind, free) MUST fan out deterministically to every stage participating in the topology.
7. Cross-stage runtime-state pointer aliasing MUST NOT occur; stages requiring runtime payload rebinding MUST synthesize stage-local wrapper blocks.

## Vision Staged Plan Contract
Multimodal-capable models MUST represent staged plans through `ModelPlans`:
1. `vision_encode` stage
2. `scatter` stage
3. `decoder_prefill` stage
4. `decoder_decode` stage

Lifecycle rules:
1. Vision stages MUST NOT implicitly reuse decoder persistent state unless explicitly declared by `StateDescriptor`.
2. `scatter` MUST use only declared instruction/state/param bindings.
3. Decoder stages MUST apply the same state binding contract as text-only routes.

## Register Allocation Contract
Register allocation is REQUIRED to be two-stage:
1. Compile-time stage (`models/`):
- compiler assigns logical registers
- compiler produces liveness map (`register_last_read`, `kill_after_instruction`)
2. Runtime/backend-init stage (`inference/` + backend):
- backend builds physical mapping from plan + liveness
- allocation MUST size from plan and mapping, not hardcoded caps

## Rule Catalog (Normative)
Every rule entry uses this REQUIRED schema:
1. Rule ID
2. Normative statement
3. Enforcement point
4. Typed failure mode
5. Verification

### R1 Typed Plan Contract
1. Rule ID: `R1`
2. Normative statement: `models/` MUST emit typed `CompiledPlan` contracts and runtime MUST reject malformed plans before execution.
3. Enforcement point: `load`
4. Typed failure mode: `error.UnsupportedModel`, `error.InvalidInstructionBinding`, `error.InvalidParamBlockABI`
5. Verification: `zig build test-models -Drelease`, `zig build test-inference -Drelease`

### R2 Adapter-Table Orchestration
1. Rule ID: `R2`
2. Normative statement: backend orchestration loops MUST dispatch by `AdapterTable` and MUST NOT use giant opcode-switch orchestration in hot loops.
3. Enforcement point: `execute`, `build`
4. Typed failure mode: `error.UnsupportedModel` at validation when adapter slot is null; compile-time coverage failure via `@compileError` for required-opcode table mismatch
5. Verification: `zig build test-inference -Drelease`

### R3 Load-Time Capability Rejection
1. Rule ID: `R3`
2. Normative statement: unsupported opcode/backend envelopes MUST fail at load/validation and MUST NOT defer primary unsupported detection to token execution.
3. Enforcement point: `load`
4. Typed failure mode: `error.UnsupportedModel`
5. Verification: `zig build test-inference -Drelease`, backend unsupported-model regression tests

### R4 Instruction-Bound Execution Truth
1. Rule ID: `R4`
2. Normative statement: adapters MUST resolve weights, params, and state exclusively from instruction-bound handles and MUST NOT read execute-path template/reference backdoors.
3. Enforcement point: `execute`
4. Typed failure mode: `error.InvalidInstructionBinding`, `error.InvalidWeightRefCount`, `error.InvalidStateDescriptorBinding`
5. Verification: `zig build test-inference -Drelease`

### R5 State Descriptor Lifecycle
1. Rule ID: `R5`
2. Normative statement: scheduler MUST own descriptor allocation and deterministic lifecycle actions (`alloc`, `reset`, `reuse`, `evict`) and direct/scheduler routes MUST share the same explicit bind contract.
3. Enforcement point: `bind`, `execute`, `cleanup`
4. Typed failure mode: `error.InvalidStateDescriptorBinding`, `error.InvalidStateLifecycleAction`
5. Verification: `zig build test-inference -Drelease`, scheduler lifecycle tests

### R6 Single Runtime Path
1. Rule ID: `R6`
2. Normative statement: each backend route MUST have one orchestration path and MUST NOT contain hidden fallback or compatibility execution loops.
3. Enforcement point: `execute`
4. Typed failure mode: `error.UnsupportedModel` at load/validation for unsupported routes
5. Verification: `zig build test-inference -Drelease`, backend route dispatch tests

### R7 Typed Failures
1. Rule ID: `R7`
2. Normative statement: contract violations MUST return typed errors and MUST NOT silently coerce, mask, or recover.
3. Enforcement point: `load`, `bind`, `execute`
4. Typed failure mode: `error.UnsupportedModel`, `error.InvalidInstructionBinding`, `error.InvalidWeightRefCount`, `error.InvalidStateDescriptorBinding`, `error.InvalidParamBlockABI`, `error.InvalidBatchSize`, `error.UnsupportedBatchSize`
5. Verification: `zig build test-inference -Drelease`, `zig build test-integration -Drelease`

### A1 Hot-Path Hygiene
1. Rule ID: `A1`
2. Normative statement: hot loops MUST NOT perform dynamic string parsing, per-token allocations, or default-on dispatch-counter atomics.
3. Enforcement point: `execute`
4. Typed failure mode: `error.InvalidParamBlockABI` for illegal dynamic param parsing paths; gate failure for forbidden hot-path pattern checks
5. Verification: `zig build test-inference -Drelease`

### A2 Register Semantics
1. Rule ID: `A2`
2. Normative statement: logical registers MUST be compiler-assigned and runtime physical mapping MUST be liveness-driven without hardcoded legacy execution caps.
3. Enforcement point: `load`, `backend init`
4. Typed failure mode: `error.InvalidInstructionBinding`, `error.InvalidRegisterSpecSize`
5. Verification: `zig build test-models -Drelease`, `zig build test-inference -Drelease`

### A3 Weight Arity Contract
1. Rule ID: `A3`
2. Normative statement: opcode weight-slot arity MUST be single-source and compiler, loader, and adapters MUST agree on exact slot counts.
3. Enforcement point: `load`, `execute`
4. Typed failure mode: `error.InvalidWeightRefCount`
5. Verification: `zig build test-inference -Drelease`

### A4 State Binding Contract
1. Rule ID: `A4`
2. Normative statement: direct routes and scheduler routes MUST use one explicit state binding contract and implicit bind helpers MUST NOT exist.
3. Enforcement point: `bind`, `execute`
4. Typed failure mode: `error.InvalidStateDescriptorBinding`
5. Verification: `zig build test-inference -Drelease`

### A5 Backend Support Envelope Truthfulness
1. Rule ID: `A5`
2. Normative statement: declared backend capability MUST match real adapter/kernel support and dummy-support declarations MUST NOT exist.
3. Enforcement point: `load`, `build`
4. Typed failure mode: `error.UnsupportedModel`; compile-time table coverage failure via `@compileError` where applicable
5. Verification: `zig build test-inference -Drelease`

### A6 Boundary Integrity
1. Rule ID: `A6`
2. Normative statement: model-family routing MUST remain in `models/`, orchestration MUST remain in `inference/`, and math kernels MUST remain in `compute/`.
3. Enforcement point: `build`, `review`
4. Typed failure mode: gate failure for boundary-policy violation
5. Verification: `zig build release -Drelease`, `zig build test-inference -Drelease`

### A7 Lifecycle Safety
1. Rule ID: `A7`
2. Normative statement: runtime ownership and cleanup MUST be deterministic and idempotent for state, buffers, and backend resources.
3. Enforcement point: `bind`, `cleanup`
4. Typed failure mode: `error.InvalidStateLifecycleAction`, `error.InvalidStateDescriptorBinding`
5. Verification: `zig build test-inference -Drelease`, `zig build test-integration -Drelease`

### A8 Deterministic Tests
1. Rule ID: `A8`
2. Normative statement: tests MUST be deterministic and bug fixes MUST include regression tests that fail before the fix.
3. Enforcement point: `test`
4. Typed failure mode: test target failure
5. Verification: `zig build test-models -Drelease`, `zig build test-inference -Drelease`, `zig build test-integration -Drelease`

### A9 Change Coherence
1. Rule ID: `A9`
2. Normative statement: contract changes MUST land atomically with code, tests, and docs and partial rollouts MUST NOT merge.
3. Enforcement point: `build`, `test`, `review`
4. Typed failure mode: gate failure for missing atomic updates
5. Verification: `zig build release -Drelease`, `make`

## Validation and Change Discipline
1. Validation gates MUST follow repository policy in `AGENTS.md` and `core/POLICY.md`.
2. `zig build test` monolithic target MUST NOT be used.
