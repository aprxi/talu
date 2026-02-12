//! Integration tests for the inference/backend/cpu/kernels module.

test {
    _ = @import("transformer_block_test.zig");
    _ = @import("scratch_buffer_test.zig");
    _ = @import("attn_temp_test.zig");
    _ = @import("attn_cache_test.zig");
    _ = @import("ffn_scratch_test.zig");
    _ = @import("multi_head_attention_test.zig");
    _ = @import("swi_g_l_u_test.zig");
    _ = @import("mo_e_f_f_n_test.zig");
    _ = @import("mo_e_scratch_test.zig");
    _ = @import("batched_k_v_cache_test.zig");
    _ = @import("layered_batched_k_v_cache_test.zig");
    _ = @import("batched_attn_temp_test.zig");
    _ = @import("r_m_s_norm_test.zig");
    _ = @import("mamba_kernel_test.zig");
    _ = @import("mamba_state_test.zig");
    _ = @import("mamba_scratch_test.zig");
    _ = @import("short_conv_kernel_test.zig");
    _ = @import("short_conv_state_test.zig");
    _ = @import("short_conv_scratch_test.zig");
}
