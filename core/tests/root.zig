//! Integration Test Runner
//!
//! This file imports all integration tests for the public API.
//! Integration tests are for STRUCTS exported from root.zig files:
//!
//!   core/src/<module>/root.zig : pub const StructName = impl.StructName;
//!     -> tests/<module>/struct_name_test.zig
//!
//! Functions (pub fn) in root.zig should have unit tests in the same file.
//!
//! Run with: zig build test-integration

const std = @import("std");

// =============================================================================
// Struct Integration Tests
// =============================================================================

// messages (TODO: several test files are outdated)
// pub const messages_storage_backend = @import("messages/storage_backend_test.zig");
// pub const messages_memory_backend = @import("messages/memory_backend_test.zig");
// pub const messages_message_record = @import("messages/message_record_test.zig");
// pub const messages_chat = @import("messages/chat_test.zig");
// pub const messages_message = @import("messages/message_test.zig");
// pub const messages_messages = @import("messages/messages_test.zig");
// pub const messages_role = @import("messages/role_test.zig");
// pub const messages_content_type = @import("messages/content_type_test.zig");

// router
pub const router_local_engine = @import("router/local_engine_test.zig");
pub const router_generation_result = @import("router/generation_result_test.zig");
pub const router_canonical_spec = @import("router/canonical_spec_test.zig");
pub const router_inference_backend = @import("router/inference_backend_test.zig");
pub const router_parsed_tool_call = @import("router/parsed_tool_call_test.zig");
pub const router_token_iterator = @import("router/token_iterator_test.zig");
// pub const router_scheduler = @import("router/scheduler_test.zig");
// pub const router_scheduler_request = @import("router/scheduler_request_test.zig");

// tokenizer
pub const tokenizer = @import("tokenizer/root.zig");

// validate (schema validation for structured output)
pub const validate = @import("validate/root.zig");

// template
pub const template = @import("template/root.zig");

// inference (TODO: outdated tests)
// pub const inference = @import("inference/root.zig");

// xray (tensor inspection and tracing) (TODO: outdated - xray.TracePoint enum changed)
// pub const xray_trace_point = @import("xray/trace_point_test.zig");
// pub const xray_traced_tensor = @import("xray/traced_tensor_test.zig");
// pub const xray_tensor_stats = @import("xray/tensor_stats_test.zig");
// pub const xray_trace_capture = @import("xray/trace_capture_test.zig");
// pub const xray_trace_point_set = @import("xray/trace_point_set_test.zig");
// pub const xray_capture_query = @import("xray/capture_query_test.zig");
// Remaining xray tests still work:
pub const xray_kernel_info = @import("xray/kernel_info_test.zig");
pub const xray_kernel_op = @import("xray/kernel_op_test.zig");
pub const xray_shape_dim = @import("xray/shape_dim_test.zig");
pub const xray_layer_geometry = @import("xray/layer_geometry_test.zig");
pub const xray_execution_plan = @import("xray/execution_plan_test.zig");
pub const xray_matmul_kernel = @import("xray/matmul_kernel_test.zig");
pub const xray_attention_type = @import("xray/attention_type_test.zig");
pub const xray_ffn_type = @import("xray/ffn_type_test.zig");
pub const xray_dump = @import("xray/dump/root.zig");

// compute
pub const compute_device = @import("compute/device_test.zig");
pub const compute_thread_pool = @import("compute/thread_pool_test.zig");
// Metal integration tests are currently excluded from this aggregate runner.
// Metal behavior remains covered by unit tests under `core/tests/compute/metal/`.
pub const compute_cpu_module_surface = @import("compute/cpu/module_surface_test.zig");
pub const compute_dl_data_type = @import("compute/d_l_data_type_test.zig");
pub const compute_dl_device = @import("compute/d_l_device_test.zig");
pub const compute_cuda = @import("compute/cuda/root.zig");
pub const compute_ops_math_primitives_ro_p_e = @import("compute/ops/math_primitives/ro_p_e_test.zig");

// io
pub const io_safetensors = @import("io/safetensors/root.zig");
pub const io_repository_bundle = @import("io/repository/bundle_test.zig");
pub const io_repository_cached_model_list_c = @import("io/repository/cached_model_list_c_test.zig");
pub const io_json = @import("io/json_test.zig");

// inference
pub const inference_config_generation = @import("inference/config/generation_config_test.zig");

// converter (top-level, not in io/)
pub const converter = @import("converter/root.zig");

// models (static architecture/types/loader/config metadata)
pub const models = @import("models/root.zig");
pub const models_loaded_model = @import("models/loaded_model_test.zig");
pub const models_loader = @import("models/loader/root.zig");
pub const models_config_model_description = @import("models/config/model_description_test.zig");

// db (TaluDB append-only columnar storage)
pub const db = @import("db/root.zig");

// responses (conversation / item / reasoning types)
pub const responses = @import("responses/root.zig");

// policy (IAM-style tool call firewall)
pub const policy = @import("policy/root.zig");

// agent (tool registry + agent loop orchestration)
pub const agent = @import("agent/root.zig");

// agent filesystem (workspace sandbox + fs operations)
pub const fs = @import("fs/root.zig");

// image (decode/convert/model_input/encode/capi)
pub const image = @import("image/root.zig");

// root exports
pub const device = @import("device_test.zig");

test {
    std.testing.refAllDecls(@This());
}
