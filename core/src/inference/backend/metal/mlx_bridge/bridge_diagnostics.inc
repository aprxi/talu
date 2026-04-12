// bridge_diagnostics.inc — XRay tracing, quantization metadata detection, env var helpers,
// prefill configuration, memory budget estimation, KV cache management.

constexpr uint8_t XRAY_POINT_EMBED = 0;
constexpr uint8_t XRAY_POINT_LAYER_INPUT = 2;
constexpr uint8_t XRAY_POINT_ATTN_OUT = 15;
constexpr uint8_t XRAY_POINT_BLOCK_OUT = 21;
constexpr uint8_t XRAY_POINT_FINAL_NORM = 26;
constexpr uint8_t XRAY_POINT_LM_HEAD = 27;
constexpr uint16_t XRAY_GLOBAL_LAYER = 0xFFFF;

struct TraceFrame {
    uint32_t token = 0;
    uint32_t layer_position = 0;
    bool emit_embed = false;
};

thread_local std::optional<mlx_ctx> g_fused_decode_ctx_cache;
thread_local std::vector<mlx_ctx*> g_fused_decode_ctx_cache_rows;
thread_local std::string g_fused_decode_ctx_cache_key;
thread_local bool g_fused_decode_ctx_cache_dirty = false;

bool xray_should_emit(uint8_t point_id, uint16_t layer, uint32_t position) {
    return talu_metal_xray_should_emit(point_id, layer, position) != 0;
}

void xray_emit_array_f32(
    uint8_t point_id,
    uint16_t layer,
    uint32_t token,
    uint32_t position,
    const array& value,
    const char* kernel_name
) {
    if (!xray_should_emit(point_id, layer, position)) return;
    const int ndim = value.ndim();
    if (ndim <= 0 || ndim > 4) return;

    const array as_f32 = astype(value, float32);
    const int flat_size = static_cast<int>(as_f32.size());
    const array flat = reshape(as_f32, {flat_size});
    eval(flat);
    synchronize();

    uint32_t dims[4] = {0, 0, 0, 0};
    for (int i = 0; i < ndim; ++i) {
        dims[i] = static_cast<uint32_t>(as_f32.shape(i));
    }
    talu_metal_xray_emit_f32(
        point_id,
        layer,
        token,
        position,
        flat.data<float>(),
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        static_cast<uint8_t>(ndim),
        kernel_name
    );
}

bool token_is_eos(int32_t token, const int32_t* eos_ids, int32_t eos_len) {
    if (!eos_ids || eos_len <= 0) return false;
    for (int32_t i = 0; i < eos_len; ++i) {
        if (token == eos_ids[i]) return true;
    }
    return false;
}

bool has_fp8_quant_metadata(const std::unordered_map<std::string, array>& tensors) {
    for (const auto& kv : tensors) {
        if (ends_with(kv.first, ".weight_scale_inv") ||
            ends_with(kv.first, ".weight_block_scale") ||
            ends_with(kv.first, ".weight_scale")) {
            return true;
        }
    }
    return false;
}

bool has_mxfp8_quant_metadata(const std::unordered_map<std::string, array>& tensors) {
    for (const auto& kv : tensors) {
        if (ends_with(kv.first, ".weight_block_scale")) {
            return true;
        }
    }
    return false;
}

bool has_grouped_affine_quant_metadata(const std::unordered_map<std::string, array>& tensors) {
    for (const auto& kv : tensors) {
        if (!ends_with(kv.first, ".weight")) continue;
        const std::string base = kv.first.substr(0, kv.first.size() - std::string(".weight").size());
        if (tensors.find(base + ".scales") != tensors.end() &&
            tensors.find(base + ".biases") != tensors.end()) {
            return true;
        }
    }
    return false;
}

bool has_nvfp4_quant_metadata(const std::unordered_map<std::string, array>& tensors) {
    for (const auto& kv : tensors) {
        if (!ends_with(kv.first, ".weight_packed")) continue;
        const std::string base = kv.first.substr(0, kv.first.size() - std::string(".weight_packed").size());
        if (tensors.find(base + ".weight_scale") != tensors.end()) {
            return true;
        }
    }
    return false;
}

bool env_truthy(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    return std::string(raw) == "1";
}

bool logits_sanity_enabled() {
    return env_truthy("TALU_METAL_LOGITS_SANITY", false);
}

void emit_logits_sanity(const char* stage, int row_idx, const float* logits, int logits_len) {
    if (!logits || logits_len <= 0) return;

    int top_idx = 0;
    float top_val = logits[0];
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t neg_inf_count = 0;

    for (int i = 0; i < logits_len; ++i) {
        const float v = logits[i];
        if (std::isnan(v)) {
            nan_count += 1;
            continue;
        }
        if (!std::isfinite(v)) {
            inf_count += 1;
            if (v < 0.0f) neg_inf_count += 1;
            continue;
        }
        if (v > top_val) {
            top_val = v;
            top_idx = i;
        }
    }

    std::fprintf(
        stderr,
        "[mlx][logits_sanity] stage=%s row=%d top0_id=%d top0=%.6f nan=%zu inf=%zu neg_inf=%zu len=%d\n",
        stage ? stage : "(null)",
        row_idx,
        top_idx,
        static_cast<double>(top_val),
        nan_count,
        inf_count,
        neg_inf_count,
        logits_len
    );
    std::fflush(stderr);
}

void emit_logits_sanity_rows(
    const char* stage,
    const array& logits_last,
    const int32_t* logical_rows,
    int32_t logical_row_count
) {
    if (!logits_sanity_enabled()) return;
    if (logits_last.ndim() != 3) return;
    const int rows = logits_last.shape(0);
    const int vocab = logits_last.shape(2);
    if (rows <= 0 || vocab <= 0) return;

    const array logits_rows = reshape(astype(logits_last, float32), {rows, vocab});
    eval(logits_rows);
    synchronize();

    const float* ptr = logits_rows.data<float>();
    if (!ptr) return;

    const int limit = std::min(rows, std::max(0, logical_row_count));
    for (int r = 0; r < limit; ++r) {
        const int row_id = logical_rows ? logical_rows[r] : r;
        emit_logits_sanity(stage, row_id, ptr + static_cast<size_t>(r) * static_cast<size_t>(vocab), vocab);
    }
}

int env_positive_int(const char* name, int fallback) {
    const char* raw = std::getenv(name);
    if (!raw) return fallback;
    try {
        const int parsed = std::stoi(raw);
        if (parsed > 0) return parsed;
    } catch (...) {
    }
    return fallback;
}

int resolve_prefill_chunk_size(const mlx_ctx* ctx = nullptr) {
    // Env var always wins when set.
    if (const char* raw = std::getenv("TALU_METAL_PREFILL_CHUNK_SIZE")) {
        try {
            const int parsed = std::stoi(raw);
            if (parsed > 0) return parsed;
        } catch (...) {}
    }
    // Larger chunks improve matmul efficiency during prefill.
    return 256;
}

int resolve_prefill_sync_interval_chunks() {
    // Chunked prefill only needs periodic sync to bound queued graph work.
    // Synchronizing every chunk leaves long prompts unnecessarily serial.
    return env_positive_int("TALU_METAL_PREFILL_SYNC_INTERVAL_CHUNKS", 4);
}

int resolve_prefill_full_prompt_threshold(const mlx_ctx* ctx = nullptr) {
    if (const char* raw = std::getenv("TALU_METAL_PREFILL_FULL_PROMPT_THRESHOLD")) {
        try {
            const int parsed = std::stoi(raw);
            if (parsed > 0) return parsed;
        } catch (...) {}
    }
    // Full-prompt prefill avoids chunking overhead for short prompts.
    return 512;
}

bool should_use_full_prompt_prefill(const mlx_ctx* ctx, int32_t prompt_len, int full_prompt_threshold) {
    if (!ctx) return false;
    if (prompt_len <= 0) return false;
    if (prompt_len > full_prompt_threshold) return false;
    if (ctx->cfg.hidden_size_per_layer_input > 0) return false;
    return true;
}

int resolve_prefill_batch_full_prompt_threshold() {
    // Batched eval traffic commonly uses ~120-200 token prompts. A higher
    // batch-only threshold keeps these requests on fused prefill paths instead
    // of falling back to per-row serial prefill.
    return env_positive_int(
        "TALU_METAL_PREFILL_BATCH_FULL_PROMPT_THRESHOLD",
        env_positive_int("TALU_METAL_PREFILL_FULL_PROMPT_THRESHOLD", 512)
    );
}

int resolve_prefill_batch_len_bucket() {
    // Variable-length fused prefill pads rows to the group's max prompt
    // length. Bucketing avoids oversized padding when one long row would
    // otherwise stretch the whole batch.
    return env_positive_int("TALU_METAL_PREFILL_BATCH_LEN_BUCKET", 64);
}

struct MemoryBudgetEstimate {
    size_t memory_size_bytes = 0;
    size_t max_recommended_bytes = 0;
    size_t effective_budget_bytes = 0;
    size_t weight_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t linear_state_bytes = 0;
    size_t overhead_bytes = 0;
    size_t total_required_bytes = 0;
    int context_tokens = 0;
    int active_slots = 1;
};

size_t checked_add_size_t(size_t a, size_t b, const char* context) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error(std::string("size overflow while computing ") + context);
    }
    return a + b;
}

std::optional<size_t> recommended_working_set_bytes() {
    if (!metal::is_available()) return std::nullopt;
    const auto& info = device_info(Device::gpu);
    auto it = info.find("max_recommended_working_set_size");
    if (it == info.end() || !std::holds_alternative<size_t>(it->second)) {
        return std::nullopt;
    }
    const size_t value = std::get<size_t>(it->second);
    if (value == 0) return std::nullopt;
    return value;
}

std::optional<size_t> device_memory_size_bytes() {
    if (!metal::is_available()) return std::nullopt;
    const auto& info = device_info(Device::gpu);
    auto it = info.find("memory_size");
    if (it != info.end() && std::holds_alternative<size_t>(it->second)) {
        const size_t value = std::get<size_t>(it->second);
        if (value > 0) return value;
    }
    it = info.find("total_memory");
    if (it != info.end() && std::holds_alternative<size_t>(it->second)) {
        const size_t value = std::get<size_t>(it->second);
        if (value > 0) return value;
    }
    return std::nullopt;
}

size_t model_weight_bytes_on_disk(const std::string& model_path) {
    size_t total = 0;
    for (const std::string& rel : discover_weight_files(model_path)) {
        const std::filesystem::path abs = std::filesystem::path(model_path) / rel;
        const auto raw_size = std::filesystem::file_size(abs);
        if (raw_size > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("weight shard too large for host size_t: " + abs.string());
        }
        total = checked_add_size_t(total, static_cast<size_t>(raw_size), "weight bytes");
    }
    return total;
}

int estimate_full_attention_layer_count(const Qwen35Config& cfg) {
    const bool hybrid =
        cfg.linear_num_value_heads > 0 &&
        cfg.linear_num_key_heads > 0 &&
        cfg.linear_key_head_dim > 0 &&
        cfg.linear_value_head_dim > 0 &&
        cfg.linear_conv_kernel_dim > 0 &&
        cfg.full_attention_interval > 1;
    if (!hybrid) return cfg.num_hidden_layers;
    const int interval = std::max(1, cfg.full_attention_interval);
    return (cfg.num_hidden_layers + interval - 1) / interval;
}

size_t estimate_kv_cache_bytes(const Qwen35Config& cfg, int context_tokens, int active_slots) {
    if (context_tokens <= 0 || active_slots <= 0) return 0;
    const int full_layers = estimate_full_attention_layer_count(cfg);
    const size_t elems =
        static_cast<size_t>(full_layers) *
        static_cast<size_t>(active_slots) *
        2 *
        static_cast<size_t>(std::max(0, cfg.num_key_value_heads)) *
        static_cast<size_t>(context_tokens) *
        static_cast<size_t>(std::max(0, cfg.head_dim));
    return elems * sizeof(uint16_t); // bf16
}

size_t estimate_linear_state_bytes(const Qwen35Config& cfg) {
    const bool has_linear =
        cfg.linear_num_value_heads > 0 &&
        cfg.linear_num_key_heads > 0 &&
        cfg.linear_key_head_dim > 0 &&
        cfg.linear_value_head_dim > 0 &&
        cfg.linear_conv_kernel_dim > 0;
    if (!has_linear) return 0;

    const int full_layers = estimate_full_attention_layer_count(cfg);
    const int linear_layers = std::max(0, cfg.num_hidden_layers - full_layers);
    if (linear_layers == 0) return 0;

    const size_t lin_key_dim = static_cast<size_t>(cfg.linear_num_key_heads) * static_cast<size_t>(cfg.linear_key_head_dim);
    const size_t lin_value_dim = static_cast<size_t>(cfg.linear_num_value_heads) * static_cast<size_t>(cfg.linear_value_head_dim);
    const size_t lin_conv_dim = (lin_key_dim * 2) + lin_value_dim;

    const size_t conv_elems =
        static_cast<size_t>(std::max(0, cfg.linear_conv_kernel_dim - 1)) *
        lin_conv_dim;
    const size_t ssm_elems =
        static_cast<size_t>(cfg.linear_num_value_heads) *
        static_cast<size_t>(cfg.linear_value_head_dim) *
        static_cast<size_t>(cfg.linear_key_head_dim);
    const size_t per_layer_elems = checked_add_size_t(conv_elems, ssm_elems, "linear state bytes");
    const size_t total_elems = per_layer_elems * static_cast<size_t>(linear_layers);
    return total_elems * sizeof(uint16_t); // bf16
}

std::optional<MemoryBudgetEstimate> estimate_memory_budget(
    const std::string& model_path,
    const Qwen35Config& cfg
) {
    const std::optional<size_t> memory_size = device_memory_size_bytes();
    if (!memory_size.has_value()) return std::nullopt;
    const std::optional<size_t> recommended = recommended_working_set_bytes();
    const int context_tokens = env_positive_int("TALU_METAL_MEMORY_CONTEXT_TOKENS", env_positive_int("TOKENS", 200));
    const int active_slots = env_positive_int("TALU_MAX_BATCH_SIZE", 8);

    MemoryBudgetEstimate estimate{};
    estimate.memory_size_bytes = *memory_size;
    estimate.max_recommended_bytes = recommended.has_value() ? *recommended : 0;
    // On Apple Silicon unified memory, Metal and CPU share physical RAM.
    // recommendedMaxWorkingSetSize is Apple's conservative perf guideline (~65-70% of RAM),
    // not a hard allocation limit — Metal can succeed well beyond it as long as physical
    // pages are available. Use physical RAM minus OS overhead as the effective budget,
    // taking the max with recommendedMaxWorkingSetSize so we never go below Apple's guidance.
    static constexpr size_t os_overhead_bytes = static_cast<size_t>(4ULL * 1024 * 1024 * 1024); // 4 GiB for OS
    const size_t physical_budget = estimate.memory_size_bytes > os_overhead_bytes
        ? estimate.memory_size_bytes - os_overhead_bytes
        : 0;
    const size_t recommended_val = recommended.has_value() ? *recommended : 0;
    estimate.effective_budget_bytes = std::max(recommended_val, physical_budget);
    estimate.context_tokens = std::max(1, context_tokens);
    estimate.active_slots = std::max(1, active_slots);
    estimate.weight_bytes = model_weight_bytes_on_disk(model_path);
    estimate.kv_cache_bytes = estimate_kv_cache_bytes(cfg, estimate.context_tokens, estimate.active_slots);
    estimate.linear_state_bytes = estimate_linear_state_bytes(cfg);
    estimate.overhead_bytes = estimate.weight_bytes / 10; // 10% weight overhead
    estimate.total_required_bytes = estimate.weight_bytes;
    estimate.total_required_bytes = checked_add_size_t(estimate.total_required_bytes, estimate.kv_cache_bytes, "total memory budget");
    estimate.total_required_bytes = checked_add_size_t(estimate.total_required_bytes, estimate.linear_state_bytes, "total memory budget");
    estimate.total_required_bytes = checked_add_size_t(estimate.total_required_bytes, estimate.overhead_bytes, "total memory budget");
    return estimate;
}

bool validate_memory_budget_or_set_error(const std::string& model_path, const Qwen35Config& cfg) {
    const std::optional<MemoryBudgetEstimate> estimate_opt = estimate_memory_budget(model_path, cfg);
    if (!estimate_opt.has_value()) return true;

    const MemoryBudgetEstimate& estimate = *estimate_opt;
    if (estimate.total_required_bytes <= estimate.effective_budget_bytes) return true;

    auto to_gib = [](size_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    };
    char buf[1024];
    std::snprintf(
        buf,
        sizeof(buf),
        "metal memory budget exceeded: required=%.2f GiB (weights=%.2f, kv=%.2f, linear=%.2f, overhead=%.2f, context_tokens=%d, slots=%d) > budget=%.2f GiB (memory_size=%.2f GiB, max_recommended=%.2f GiB)",
        to_gib(estimate.total_required_bytes),
        to_gib(estimate.weight_bytes),
        to_gib(estimate.kv_cache_bytes),
        to_gib(estimate.linear_state_bytes),
        to_gib(estimate.overhead_bytes),
        estimate.context_tokens,
        estimate.active_slots,
        to_gib(estimate.effective_budget_bytes),
        to_gib(estimate.memory_size_bytes),
        to_gib(estimate.max_recommended_bytes)
    );
    g_last_error = buf;
    return false;
}

int max_kv_cache_step(const mlx_ctx* ctx) {
    if (!ctx) return 0;
    int max_step = 0;
    for (const KVCacheState& kv : ctx->kv_cache) {
        max_step = std::max(max_step, kv.step);
    }
    return max_step;
}

int resolve_kv_reserve_target_tokens(const mlx_ctx* ctx, int prompt_len, int decode_tokens_hint) {
    if (!ctx) return 0;
    const int base_prompt = std::max(prompt_len, 0);
    const int base_decode = std::max(decode_tokens_hint, 0);
    const int headroom = base_decode > 0 ? base_decode : max_kv_cache_step(ctx);
    return base_prompt + headroom;
}

void ensure_kv_cache_capacity(
    const mlx_ctx* ctx,
    KVCacheState* cache,
    int batch,
    int num_kv_heads,
    int head_dim,
    int required_tokens,
    Dtype key_dtype,
    Dtype value_dtype
) {
    if (!cache) return;
    if (required_tokens <= 0) return;

    const int effective_step = std::max(cache->step, 1);
    const int reserve_tokens = ctx ? std::max(ctx->kv_reserve_tokens, 0) : 0;
    const int target_tokens = std::max(required_tokens, reserve_tokens);
    const bool has_cache = cache->keys.has_value() && cache->values.has_value();
    const int current_capacity = has_cache ? cache->keys->shape(2) : 0;
    if (current_capacity >= target_tokens) return;

    int target_capacity = target_tokens;
    if (has_cache) {
        target_capacity = ((target_tokens + effective_step - 1) / effective_step) * effective_step;
    }

    if (!has_cache || cache->offset <= 0) {
        cache->keys = zeros({batch, num_kv_heads, target_capacity, head_dim}, key_dtype);
        cache->values = zeros({batch, num_kv_heads, target_capacity, head_dim}, value_dtype);
        return;
    }

    const int preserved_tokens = std::min(cache->offset, current_capacity);
    array preserved_keys = preserved_tokens == current_capacity
        ? *cache->keys
        : slice(*cache->keys, {0, 0, 0, 0}, {batch, num_kv_heads, preserved_tokens, head_dim});
    array preserved_values = preserved_tokens == current_capacity
        ? *cache->values
        : slice(*cache->values, {0, 0, 0, 0}, {batch, num_kv_heads, preserved_tokens, head_dim});

    const int tail_tokens = std::max(0, target_capacity - preserved_tokens);
    if (tail_tokens > 0) {
        preserved_keys = concatenate(
            {preserved_keys, zeros({batch, num_kv_heads, tail_tokens, head_dim}, key_dtype)},
            2
        );
        preserved_values = concatenate(
            {preserved_values, zeros({batch, num_kv_heads, tail_tokens, head_dim}, value_dtype)},
            2
        );
    }
    cache->keys = preserved_keys;
    cache->values = preserved_values;
}
