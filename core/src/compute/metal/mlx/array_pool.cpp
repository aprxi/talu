// MLX Bridge - Array Pool and Memory Management
//
// Provides array object pooling to eliminate per-step heap allocations.
// Also handles MLX initialization and memory cache management.

#include "compute_common.h"
#include "../device.h"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

extern "C" void* mlx_persistent_cast_f16(const void* input);

// ============================================================================
// Global state
// ============================================================================

static constexpr size_t kArrayPoolMaxRetained = 4096;

// Heap-owned handles created via make_owned_array()/array_from_* APIs.
// Freed by mlx_array_free() and guarded for cross-thread lifecycle safety.
static std::mutex g_owned_arrays_mu;
static std::unordered_set<void*> g_owned_arrays;
static std::atomic_size_t g_ingest_zero_copy_count{0};
static std::atomic_size_t g_ingest_copy_count{0};
static constexpr size_t kIngestZeroCopyAlignment = 16 * 1024;

// Pre-computed constants
const std::vector<int> g_transpose_perm = {0, 2, 1, 3};
const Shape g_slice_start = {0, 0, 0, 0};

enum class IngestLifetime {
    copy_owned,
    caller_persistent,
};

static inline std::deque<std::optional<array>>& array_pool_store() {
    return tls_never_destroyed<std::deque<std::optional<array>>>();
}

static inline size_t& pool_index_store() {
    return tls_never_destroyed<size_t>();
}

template <typename T>
static void* create_ingested_array(
    const void* data,
    const size_t* shape,
    size_t ndim,
    Dtype dtype,
    IngestLifetime lifetime
) {
    Shape shape_dims;
    size_t element_count = 1;
    for (size_t i = 0; i < ndim; i++) {
        shape_dims.push_back(static_cast<int>(shape[i]));
        element_count *= shape[i];
    }

    const auto* typed_data = static_cast<const T*>(data);
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(data);
    const bool can_zero_copy = lifetime == IngestLifetime::caller_persistent and
        (ptr_value % kIngestZeroCopyAlignment) == 0;

    if (can_zero_copy) {
        // Persistent zero-copy is only valid for host buffers whose lifetime is
        // explicitly guaranteed to outlive the resulting MLX array handle.
        // Generic runtime ingestion must not take this path.
        g_ingest_zero_copy_count.fetch_add(1, std::memory_order_relaxed);
        auto no_op_deleter = [](void*) {};
        return make_owned_array(array(const_cast<T*>(typed_data), shape_dims, dtype, no_op_deleter));
    }

    // Default host ingestion copies into compute-owned aligned storage.
    // This makes the lifetime contract explicit: generic array creation is safe
    // for request-scoped and temporary host buffers, independent of allocator
    // alignment. Only explicit persistent APIs may borrow caller-owned memory.
    g_ingest_copy_count.fetch_add(1, std::memory_order_relaxed);
    const size_t byte_count = element_count * sizeof(T);
    const size_t alloc_bytes = std::max(byte_count, sizeof(T));
    void* aligned_ptr = nullptr;
    if (posix_memalign(&aligned_ptr, kIngestZeroCopyAlignment, alloc_bytes) != 0 || aligned_ptr == nullptr) {
        throw std::bad_alloc();
    }
    if (byte_count > 0) {
        std::memcpy(aligned_ptr, data, byte_count);
    }
    // Keep aligned storage ownership in a ref-counted holder. This avoids
    // double-free hazards if MLX copies/moves array handles internally.
    auto aligned_owner = std::shared_ptr<void>(aligned_ptr, [](void* ptr) {
        std::free(ptr);
    });
    auto shared_deleter = [aligned_owner](void*) {
        // ownership retained by closure capture
    };
    return make_owned_array(array(static_cast<T*>(aligned_ptr), shape_dims, dtype, shared_deleter));
}

// ============================================================================
// MLX Initialization
// ============================================================================
// Runs once at library load to configure MLX for optimal performance.

static struct Init {
    Init() {
        // Do not touch MLX Metal device APIs at static-init time.
        // Some host setups can raise Objective-C exceptions before main(),
        // which bypass C++ catch blocks and abort the process.
        (void)metal_is_available;

        // Enable compilation mode for better operation fusion
        enable_compile();
    }
} g_init;

// ============================================================================
// Array Pool Implementation
// ============================================================================

void* pool_array(array&& result) {
    auto& g_array_pool = array_pool_store();
    auto& g_pool_index = pool_index_store();
    if (g_pool_index < g_array_pool.size()) {
        g_array_pool[g_pool_index] = std::move(result);
        return &g_array_pool[g_pool_index++].value();
    }
    // Grow pool if needed - deque doesn't invalidate existing pointers
    g_array_pool.push_back(std::move(result));
    g_pool_index++;
    return &g_array_pool.back().value();
}

void* pool_array(const array& result) {
    return pool_array(array(result));  // Copy then move
}

void* make_owned_array(array&& result) {
    auto* handle = new array(std::move(result));
    std::lock_guard<std::mutex> lock(g_owned_arrays_mu);
    g_owned_arrays.insert(handle);
    return handle;
}

// ============================================================================
// C API - Array Pool Management
// ============================================================================

extern "C" {

void mlx_array_free(void* arr);

void mlx_pool_reset() {
    // reset is the explicit eval-complete barrier for pooled temporaries
    ScopedAutoreleasePool pool;
    auto& g_pool_index = pool_index_store();
    auto& g_array_pool = array_pool_store();
    g_pool_index = 0;
    if (g_array_pool.size() > kArrayPoolMaxRetained) {
        g_array_pool.resize(kArrayPoolMaxRetained);
    }
}

size_t mlx_pool_max_retained() {
    return kArrayPoolMaxRetained;
}

bool mlx_pool_clear_if_idle() {
    ScopedAutoreleasePool pool;
    auto& g_pool_index = pool_index_store();
    auto& g_array_pool = array_pool_store();
    if (g_pool_index != 0) {
        return false;
    }
    g_array_pool.clear();
    return true;
}

bool mlx_pool_compact_if_idle(size_t max_retained) {
    ScopedAutoreleasePool pool;
    auto& g_pool_index = pool_index_store();
    auto& g_array_pool = array_pool_store();
    if (g_pool_index != 0) {
        return false;
    }
    if (g_array_pool.size() > max_retained) {
        g_array_pool.resize(max_retained);
    }
    return true;
}

void mlx_clear_memory_cache() {
    ScopedAutoreleasePool pool;
    clear_cache();
    mlx_weight_transform_cache_clear();
    mlx_quant_param_cast_cache_clear();
}

void mlx_clear_thread_local_run_state() {
    // Independent runs must not inherit pooled transient arrays or op-count
    // instrumentation from prior work on the same thread.
    ScopedAutoreleasePool pool;
    auto& g_pool_index = pool_index_store();
    auto& g_array_pool = array_pool_store();
    g_pool_index = 0;
    std::deque<std::optional<array>>().swap(g_array_pool);
    g_count_ops_enabled = false;
    g_count_ops_value = 0;
}

void mlx_clear_thread_local_state() {
    // Backend shutdown must clear both pooled transients and transform caches
    // on the same execution thread. This avoids late thread-local destructor
    // teardown and keeps cache-key lifetimes explicit.
    mlx_clear_thread_local_run_state();
    mlx_weight_transform_cache_clear();
    mlx_quant_param_cast_cache_clear();
    mlx_gqa_index_cache_clear();
}

void mlx_synchronize_default_stream() {
    synchronize();
}

void mlx_pool_stats(size_t* pool_size, size_t* used) {
    auto& g_pool_index = pool_index_store();
    auto& g_array_pool = array_pool_store();
    *pool_size = g_array_pool.size();
    *used = g_pool_index;
}

void mlx_array_ingest_stats(size_t* zero_copy_count, size_t* copy_count) {
    *zero_copy_count = g_ingest_zero_copy_count.load(std::memory_order_relaxed);
    *copy_count = g_ingest_copy_count.load(std::memory_order_relaxed);
}

void mlx_array_ingest_stats_reset() {
    g_ingest_zero_copy_count.store(0, std::memory_order_relaxed);
    g_ingest_copy_count.store(0, std::memory_order_relaxed);
}

// ============================================================================
// C API - Array from existing pointer
// ============================================================================

void* mlx_array_from_ptr(void* mlx_array_ptr) {
    return mlx_array_ptr;  // Just return as-is, it's already an array*
}

// ============================================================================
// C API - Array Creation
// ============================================================================

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_float32(const void* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<float>(data, shape, ndim, float32, IngestLifetime::copy_owned);
}

void* mlx_array_from_float32_persistent(const void* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<float>(data, shape, ndim, float32, IngestLifetime::caller_persistent);
}

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_uint32(const void* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<uint32_t>(data, shape, ndim, uint32, IngestLifetime::copy_owned);
}

void* mlx_array_from_uint32_persistent(const void* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<uint32_t>(data, shape, ndim, uint32, IngestLifetime::caller_persistent);
}

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_bfloat16(const void* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<uint16_t>(data, shape, ndim, bfloat16, IngestLifetime::copy_owned);
}

void* mlx_array_from_bfloat16_dense_weight(const void* data, const size_t* shape, size_t ndim) {
    void* handle = create_ingested_array<uint16_t>(data, shape, ndim, bfloat16, IngestLifetime::caller_persistent);
    // Dense matrix weights are decode-hot; keep them persisted as float16.
    if (ndim >= 2) {
        void* casted = mlx_persistent_cast_f16(handle);
        mlx_array_free(handle);
        return casted;
    }
    return handle;
}

void* mlx_array_from_bfloat16_norm(const void* data, const size_t* shape, size_t ndim) {
    (void)ndim;
    void* handle = create_ingested_array<uint16_t>(data, shape, ndim, bfloat16, IngestLifetime::caller_persistent);
    // Norm vectors are used on every layer step; keep on the same f16 path.
    void* casted = mlx_persistent_cast_f16(handle);
    mlx_array_free(handle);
    return casted;
}

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_float16(const void* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<uint16_t>(data, shape, ndim, float16, IngestLifetime::copy_owned);
}

void* mlx_array_from_uint8(const uint8_t* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<uint8_t>(data, shape, ndim, uint8, IngestLifetime::copy_owned);
}

void mlx_array_free(void* arr) {
    if (arr == nullptr) return;
    array* owned = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_owned_arrays_mu);
        auto it = g_owned_arrays.find(arr);
        if (it == g_owned_arrays.end()) return;
        owned = static_cast<array*>(*it);
        g_owned_arrays.erase(it);
    }
    // Transform caches are keyed by persistent weight handles and must be
    // cleared only at explicit backend/run lifecycle barriers. Clearing them on
    // every owned-array delete destabilizes teardown and defeats caching.
    //
    // MLX Metal arrays ultimately release ARC-managed Objective-C resources.
    // Draining an autorelease pool at this exact destruction boundary keeps
    // their lifetime deterministic across the Zig/Rust/FFI teardown path.
    ScopedAutoreleasePool pool;
    delete owned;
}

// ============================================================================
// C API - Array Evaluation
// ============================================================================

void mlx_eval(void** handles, size_t n) {
    ScopedAutoreleasePool pool;
    if (n == 1) {
        eval(*static_cast<array*>(handles[0]));
        return;
    }
    std::vector<array> arrays;
    arrays.reserve(n);
    for (size_t i = 0; i < n; i++) {
        arrays.push_back(*static_cast<array*>(handles[i]));
    }
    eval(arrays);
}

void mlx_async_eval(void** handles, size_t n) {
    ScopedAutoreleasePool pool;
    if (n == 1) {
        async_eval(*static_cast<array*>(handles[0]));
        return;
    }
    std::vector<array> arrays;
    arrays.reserve(n);
    for (size_t i = 0; i < n; i++) {
        arrays.push_back(*static_cast<array*>(handles[i]));
    }
    async_eval(arrays);
}

// ============================================================================
// C API - Array Data Access
// ============================================================================

void mlx_array_to_float32(const void* handle, float* out, size_t size) {
    ScopedAutoreleasePool pool;
    const auto& array_ref = *static_cast<const array*>(handle);
    auto converted = (array_ref.dtype() != float32) ? astype(array_ref, float32) : array_ref;
    eval(converted);
    memcpy(out, converted.data<float>(), size * sizeof(float));
}

void mlx_array_to_uint32(const void* handle, uint32_t* out, size_t size) {
    ScopedAutoreleasePool pool;
    const auto& array_ref = *static_cast<const array*>(handle);
    auto converted = (array_ref.dtype() != uint32) ? astype(array_ref, uint32) : array_ref;
    eval(converted);
    memcpy(out, converted.data<uint32_t>(), size * sizeof(uint32_t));
}

uint32_t mlx_array_item_u32(const void* handle) {
    ScopedAutoreleasePool pool;
    const auto& array_ref = *static_cast<const array*>(handle);
    return static_cast<uint32_t>(array_ref.item<int32_t>());
}

void mlx_array_shape(const void* handle, size_t* shape_out, size_t* ndim_out) {
    const auto& array_ref = *static_cast<const array*>(handle);
    *ndim_out = array_ref.ndim();
    for (size_t i = 0; i < array_ref.ndim(); i++) {
        shape_out[i] = array_ref.shape(i);
    }
}

} // extern "C"
