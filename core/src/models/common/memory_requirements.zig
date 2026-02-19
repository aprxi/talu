//! Runtime memory requirements for static model execution.

pub const KVCacheRequirements = struct {
    n_layers: usize = 0,
    n_kv_heads: usize = 0,
    head_dim: usize = 0,
    max_seq_len: usize = 0,
};

pub const ScratchRequirements = struct {
    d_model: usize = 0,
    d_ff: usize = 0,
    batch_size: usize = 0,
};

pub const MemoryRequirements = struct {
    kv_cache: KVCacheRequirements = .{},
    scratch: ScratchRequirements = .{},
    needs_shortconv_state: bool = false,
    needs_mamba_state: bool = false,
    needs_mla_state: bool = false,
};
