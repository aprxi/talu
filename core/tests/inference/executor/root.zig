//! Integration tests for the inference/executor module.

test {
    _ = @import("attention_test.zig");
    _ = @import("f_f_n_layer_test.zig");
    _ = @import("attn_temp_test.zig");
    _ = @import("attn_cache_test.zig");
    _ = @import("scratch_buffer_test.zig");
    _ = @import("transformer_block_test.zig");
    _ = @import("linear_test.zig");
    _ = @import("embedding_test.zig");
    _ = @import("block_test.zig");
    _ = @import("layers_test.zig");
    _ = @import("transformer_test.zig");
    _ = @import("r_m_s_norm_test.zig");
}
