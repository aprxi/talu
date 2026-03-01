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

// ============================================================================
// Global state
// ============================================================================

// Array pool - thread-local for safety
thread_local std::deque<std::optional<array>> g_array_pool;
thread_local size_t g_pool_index = 0;
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

template <typename T>
static void* create_ingested_array(
    const void* data,
    const size_t* shape,
    size_t ndim,
    Dtype dtype
) {
    Shape shape_dims;
    size_t element_count = 1;
    for (size_t i = 0; i < ndim; i++) {
        shape_dims.push_back(static_cast<int>(shape[i]));
        element_count *= shape[i];
    }

    const auto* typed_data = static_cast<const T*>(data);
    const auto ptr_value = reinterpret_cast<std::uintptr_t>(data);
    const bool can_zero_copy = (ptr_value % kIngestZeroCopyAlignment) == 0;

    if (can_zero_copy) {
        // Zero-copy path: caller-owned backing store must outlive resulting array.
        g_ingest_zero_copy_count.fetch_add(1, std::memory_order_relaxed);
        auto no_op_deleter = [](void*) {};
        return make_owned_array(array(const_cast<T*>(typed_data), shape_dims, dtype, no_op_deleter));
    }

    // Fallback copy path for unaligned pointers.
    // Copy into page-aligned storage so MLX/Metal can consume it without an
    // additional internal realignment copy.
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

void mlx_pool_reset() {
    // reset is the explicit eval-complete barrier for pooled temporaries
    g_pool_index = 0;
    if (g_array_pool.size() > kArrayPoolMaxRetained) {
        g_array_pool.resize(kArrayPoolMaxRetained);
    }
}

size_t mlx_pool_max_retained() {
    return kArrayPoolMaxRetained;
}

bool mlx_pool_clear_if_idle() {
    if (g_pool_index != 0) return false;
    g_array_pool.clear();
    return true;
}

bool mlx_pool_compact_if_idle(size_t max_retained) {
    if (g_pool_index != 0) return false;
    if (g_array_pool.size() > max_retained) {
        g_array_pool.resize(max_retained);
    }
    return true;
}

void mlx_clear_memory_cache() {
    clear_cache();
}

void mlx_pool_stats(size_t* pool_size, size_t* used) {
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
    return create_ingested_array<float>(data, shape, ndim, float32);
}

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_uint32(const void* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<uint32_t>(data, shape, ndim, uint32);
}

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_bfloat16(const void* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<uint16_t>(data, shape, ndim, bfloat16);
}

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_float16(const void* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<uint16_t>(data, shape, ndim, float16);
}

void* mlx_array_from_uint8(const uint8_t* data, const size_t* shape, size_t ndim) {
    return create_ingested_array<uint8_t>(data, shape, ndim, uint8);
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
    delete owned;
}

// ============================================================================
// C API - Array Evaluation
// ============================================================================

void mlx_eval(void** handles, size_t n) {
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
    const auto& array_ref = *static_cast<const array*>(handle);
    auto converted = (array_ref.dtype() != float32) ? astype(array_ref, float32) : array_ref;
    eval(converted);
    memcpy(out, converted.data<float>(), size * sizeof(float));
}

uint32_t mlx_array_item_u32(const void* handle) {
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
