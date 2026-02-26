//! Generic inference runtime contract root.

pub const types = @import("types.zig");
pub const allocator = @import("allocator.zig");

pub const Opcode = types.Opcode;
pub const RegisterRef = types.RegisterRef;
pub const WeightRef = types.WeightRef;
pub const StateLifecycle = types.StateLifecycle;
pub const StateDescriptor = types.StateDescriptor;
pub const Instruction = types.Instruction;
pub const ExecutionPlan = types.ExecutionPlan;
pub const LivenessMap = types.LivenessMap;
pub const PlanDiagnosticLevel = types.PlanDiagnosticLevel;
pub const PlanDiagnostic = types.PlanDiagnostic;
pub const CompiledPlan = types.CompiledPlan;
pub const PhysicalBufferSpec = types.PhysicalBufferSpec;
pub const PhysicalMapping = types.PhysicalMapping;
pub const TensorHandle = types.TensorHandle;
pub const TensorLayout = types.TensorLayout;
pub const TensorViewDesc = types.TensorViewDesc;
pub const StateBlockHandle = types.StateBlockHandle;
pub const ParamBlock = types.ParamBlock;
pub const ExecutionMode = types.ExecutionMode;
pub const Workspace = types.Workspace;
pub const ExecutionContext = types.ExecutionContext;
pub const KernelAdapterFn = types.KernelAdapterFn;
pub const AdapterTable = types.AdapterTable;
pub const AdapterCapability = types.AdapterCapability;
pub const AdapterCapabilities = types.AdapterCapabilities;

pub const registerFromIndex = types.registerFromIndex;
pub const registerToIndex = types.registerToIndex;
pub const validateTensorViewDesc = types.validateTensorViewDesc;
pub const validateExecutionContext = types.validateExecutionContext;
pub const validateExecutionPlan = types.validateExecutionPlan;
pub const validateCompiledPlan = types.validateCompiledPlan;
pub const RegisterBufferSpec = allocator.RegisterBufferSpec;
pub const buildPhysicalMappingLinearScan = allocator.buildPhysicalMappingLinearScan;
pub const deinitPhysicalMapping = allocator.deinitPhysicalMapping;
