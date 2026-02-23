// MLX Bridge - Array Pool and Memory Management
//
// Provides array object pooling to eliminate per-step heap allocations.
// Also handles MLX initialization and memory cache management.

#include "compute_common.h"
#include "../device.h"

// ============================================================================
// Global state
// ============================================================================

// Array pool - thread-local for safety
thread_local std::deque<std::optional<array>> g_array_pool;
thread_local size_t g_pool_index = 0;

// Heap-owned handles created via make_owned_array()/array_from_* APIs.
// Freed by mlx_array_free() and guarded for cross-thread lifecycle safety.
static std::mutex g_owned_arrays_mu;
static std::unordered_set<void*> g_owned_arrays;

// Pre-computed constants
const std::vector<int> g_transpose_perm = {0, 2, 1, 3};
const Shape g_slice_start = {0, 0, 0, 0};

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
    g_pool_index = 0;
}

void mlx_clear_memory_cache() {
    clear_cache();
}

void mlx_pool_stats(size_t* pool_size, size_t* used) {
    *pool_size = g_array_pool.size();
    *used = g_pool_index;
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
    Shape shape_dims;
    size_t element_count = 1;
    for (size_t i = 0; i < ndim; i++) {
        shape_dims.push_back(static_cast<int>(shape[i]));
        element_count *= shape[i];
    }
    // Use memcpy to handle potentially unaligned mmap'd data from safetensors
    auto vec_data = std::make_shared<std::vector<float>>(element_count);
    std::memcpy(vec_data->data(), data, element_count * sizeof(float));
    auto deleter = [vec_data](void*) { /* ref-counts vec_data */ };
    return make_owned_array(array(vec_data->data(), shape_dims, float32, deleter));
}

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_uint32(const void* data, const size_t* shape, size_t ndim) {
    Shape shape_dims;
    size_t element_count = 1;
    for (size_t i = 0; i < ndim; i++) {
        shape_dims.push_back(static_cast<int>(shape[i]));
        element_count *= shape[i];
    }
    // Use memcpy to handle potentially unaligned mmap'd data from safetensors
    auto vec_data = std::make_shared<std::vector<uint32_t>>(element_count);
    std::memcpy(vec_data->data(), data, element_count * sizeof(uint32_t));
    auto deleter = [vec_data](void*) { /* ref-counts vec_data */ };
    return make_owned_array(array(vec_data->data(), shape_dims, uint32, deleter));
}

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_bfloat16(const void* data, const size_t* shape, size_t ndim) {
    Shape shape_dims;
    size_t element_count = 1;
    for (size_t i = 0; i < ndim; i++) {
        shape_dims.push_back(static_cast<int>(shape[i]));
        element_count *= shape[i];
    }
    // Use memcpy to handle potentially unaligned mmap'd data from safetensors
    auto vec_data = std::make_shared<std::vector<uint16_t>>(element_count);
    std::memcpy(vec_data->data(), data, element_count * sizeof(uint16_t));
    auto deleter = [vec_data](void*) { /* ref-counts vec_data */ };
    return make_owned_array(array(vec_data->data(), shape_dims, bfloat16, deleter));
}

// Note: data is const void* to handle potentially unaligned mmap'd safetensor data
void* mlx_array_from_float16(const void* data, const size_t* shape, size_t ndim) {
    Shape shape_dims;
    size_t element_count = 1;
    for (size_t i = 0; i < ndim; i++) {
        shape_dims.push_back(static_cast<int>(shape[i]));
        element_count *= shape[i];
    }
    // Use memcpy to handle potentially unaligned mmap'd data from safetensors
    auto vec_data = std::make_shared<std::vector<uint16_t>>(element_count);
    std::memcpy(vec_data->data(), data, element_count * sizeof(uint16_t));
    auto deleter = [vec_data](void*) { /* ref-counts vec_data */ };
    return make_owned_array(array(vec_data->data(), shape_dims, float16, deleter));
}

void* mlx_array_from_uint8(const uint8_t* data, const size_t* shape, size_t ndim) {
    Shape shape_dims;
    size_t element_count = 1;
    for (size_t i = 0; i < ndim; i++) {
        shape_dims.push_back(static_cast<int>(shape[i]));
        element_count *= shape[i];
    }
    auto vec_data = std::make_shared<std::vector<uint8_t>>(data, data + element_count);
    auto deleter = [vec_data](void*) { /* ref-counts vec_data */ };
    return make_owned_array(array(vec_data->data(), shape_dims, uint8, deleter));
}

void mlx_array_free(void* arr) {
    if (arr == nullptr) return;
    std::lock_guard<std::mutex> lock(g_owned_arrays_mu);
    auto it = g_owned_arrays.find(arr);
    if (it == g_owned_arrays.end()) return;
    delete static_cast<array*>(arr);
    g_owned_arrays.erase(it);
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
