//! MoE weight upload helpers for the CUDA inference backend.

const upload_dispatch = @import("upload_dispatch.zig");
const uploadLinearWeightWithContext = upload_dispatch.uploadLinearWeightWithContext;
const uploadTensor = upload_dispatch.uploadTensor;

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");
const load_transforms = @import("models_pkg").load.transforms;
const models = @import("models_pkg");
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;

/// Convert UE8M0 block scale exponent to f32 scale factor.
inline fn ue8m0ToScale(e8m0: u8) f32 {
    const exp_bits = @as(u32, e8m0) << 23;
    return @bitCast(exp_bits);
}

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/root.zig");
const KvCacheDtype = engine_types.KvCacheDtype;
const gaffine_scales_dtype_f16 = engine_types.gaffine_scales_dtype_f16;
const gaffine_scales_dtype_bf16 = engine_types.gaffine_scales_dtype_bf16;
const DenseU16Dtype = engine_types.DenseU16Dtype;
const EmbeddingLookupKind = engine_types.EmbeddingLookupKind;
const EmbeddingLookup = engine_types.EmbeddingLookup;
const LinearWeight = engine_types.LinearWeight;
const DeviceTensor = engine_types.DeviceTensor;
const MoEWeightRefs = engine_types.MoEWeightRefs;
const MoEWeights = models.runtime_blocks.MoEWeights;

pub fn uploadMoEWeights(
    device: *compute.cuda.Device,
    allocator: std.mem.Allocator,
    moe: *const MoEWeights,
    d_model: usize,
    layer_idx: usize,
    use_gelu: bool,
) !MoEWeightRefs {
    const num_experts: u32 = @intCast(moe.num_experts);
    const experts_per_token: u32 = @intCast(moe.experts_per_token);

    // Determine expert dimensions from first expert's gate_up_proj
    if (moe.experts.len == 0) return error.MissingWeight;
    const first_gate_up = moe.experts[0].gate_up_proj orelse return error.MissingWeight;
    if (first_gate_up.n_dims != 2) return error.UnsupportedModel;
    const gate_up_out_dim: usize = @intCast(first_gate_up.shape[0]);
    if (gate_up_out_dim == 0 or (gate_up_out_dim % 2) != 0) return error.UnsupportedModel;
    const expert_d_ff: u32 = @intCast(gate_up_out_dim / 2);

    // Shared MLP dimensions
    const shared_w1_src = moe.shared_w1 orelse return error.MissingWeight;
    const shared_w2_src = moe.shared_w2 orelse return error.MissingWeight;
    const shared_w3_src = moe.shared_w3 orelse return error.MissingWeight;
    if (shared_w1_src.n_dims != 2) return error.UnsupportedModel;
    const shared_d_ff: u32 = @intCast(shared_w1_src.shape[0]);

    // Upload expert weights
    var expert_gate_up = try allocator.alloc(LinearWeight, num_experts);
    errdefer allocator.free(expert_gate_up);
    var uploaded_gate_up: usize = 0;
    errdefer for (expert_gate_up[0..uploaded_gate_up]) |*w| w.deinit(device);

    var expert_down = try allocator.alloc(LinearWeight, num_experts);
    errdefer allocator.free(expert_down);
    var uploaded_down: usize = 0;
    errdefer for (expert_down[0..uploaded_down]) |*w| w.deinit(device);

    for (0..num_experts) |e| {
        const gu = moe.experts[e].gate_up_proj orelse return error.MissingWeight;
        expert_gate_up[e] = try uploadLinearWeightWithContext(device, allocator, &gu, d_model, layer_idx, "experts.gate_up_proj");
        uploaded_gate_up += 1;

        expert_down[e] = try uploadLinearWeightWithContext(device, allocator, &moe.experts[e].down_proj, @as(usize, expert_d_ff), layer_idx, "experts.down_proj");
        uploaded_down += 1;
    }

    // Upload shared MLP
    var shared_gate = try uploadLinearWeightWithContext(device, allocator, &shared_w1_src, d_model, layer_idx, "mlp.gate_proj.weight");
    errdefer shared_gate.deinit(device);
    var shared_up = try uploadLinearWeightWithContext(device, allocator, &shared_w3_src, d_model, layer_idx, "mlp.up_proj.weight");
    errdefer shared_up.deinit(device);
    var shared_down = try uploadLinearWeightWithContext(device, allocator, &shared_w2_src, @as(usize, shared_d_ff), layer_idx, "mlp.down_proj.weight");
    errdefer shared_down.deinit(device);

    // Upload router
    var router_proj = try uploadLinearWeightWithContext(device, allocator, &moe.router_weight, d_model, layer_idx, "router.proj.weight");
    errdefer router_proj.deinit(device);

    // Upload optional router scaling tensors.
    var router_input_scale: ?DeviceTensor = null;
    if (moe.router_input_scale) |src| {
        router_input_scale = try uploadTensor(device, allocator, &src);
    }
    errdefer if (router_input_scale) |*t| t.deinit(device);
    var router_per_expert_scale: ?DeviceTensor = null;
    if (moe.router_per_expert_scale) |src| {
        router_per_expert_scale = try uploadTensor(device, allocator, &src);
    }
    errdefer if (router_per_expert_scale) |*t| t.deinit(device);

    // Upload optional internal norm tensors.
    var pre_ffn_norm: ?DeviceTensor = null;
    if (moe.pre_ffn_norm) |src| {
        pre_ffn_norm = try uploadTensor(device, allocator, &src);
    }
    errdefer if (pre_ffn_norm) |*t| t.deinit(device);
    var post_shared_norm: ?DeviceTensor = null;
    if (moe.post_shared_norm) |src| {
        post_shared_norm = try uploadTensor(device, allocator, &src);
    }
    errdefer if (post_shared_norm) |*t| t.deinit(device);
    var pre_expert_norm: ?DeviceTensor = null;
    if (moe.pre_expert_norm) |src| {
        pre_expert_norm = try uploadTensor(device, allocator, &src);
    }
    errdefer if (pre_expert_norm) |*t| t.deinit(device);
    var post_expert_norm: ?DeviceTensor = null;
    if (moe.post_expert_norm) |src| {
        post_expert_norm = try uploadTensor(device, allocator, &src);
    }
    errdefer if (post_expert_norm) |*t| t.deinit(device);
    var post_combine_norm: ?DeviceTensor = null;
    if (moe.post_combine_norm) |src| {
        post_combine_norm = try uploadTensor(device, allocator, &src);
    }
    errdefer if (post_combine_norm) |*t| t.deinit(device);

    // Upload optional shared-expert gate (sigmoid scaling).
    var shared_expert_gate: ?LinearWeight = null;
    if (moe.shared_expert_gate) |gate_src| {
        shared_expert_gate = try uploadLinearWeightWithContext(device, allocator, &gate_src, d_model, layer_idx, "shared_expert_gate");
    }
    errdefer if (shared_expert_gate) |*w| w.deinit(device);

    // Router scalar is only active when router input scaling is present.
    const router_scalar: f32 = if (router_input_scale != null)
        1.0 / @sqrt(@as(f32, @floatFromInt(d_model)))
    else
        1.0;

    log.info("inference", "CUDA MoE weights uploaded", .{
        .layer = layer_idx,
        .num_experts = num_experts,
        .experts_per_token = experts_per_token,
        .expert_d_ff = expert_d_ff,
        .shared_d_ff = shared_d_ff,
    });

    return .{
        .expert_gate_up = expert_gate_up,
        .expert_down = expert_down,
        .shared_gate = shared_gate,
        .shared_up = shared_up,
        .shared_down = shared_down,
        .router_proj = router_proj,
        .router_input_scale = router_input_scale,
        .router_per_expert_scale = router_per_expert_scale,
        .pre_ffn_norm = pre_ffn_norm,
        .post_shared_norm = post_shared_norm,
        .pre_expert_norm = pre_expert_norm,
        .post_expert_norm = post_expert_norm,
        .post_combine_norm = post_combine_norm,
        .shared_expert_gate = shared_expert_gate,
        .num_experts = num_experts,
        .experts_per_token = experts_per_token,
        .expert_d_ff = expert_d_ff,
        .shared_d_ff = shared_d_ff,
        .router_scalar = router_scalar,
        .use_gelu = use_gelu,
    };
}
