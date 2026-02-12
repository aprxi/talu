// =============================================================================
// Metal Device Management - C API
// =============================================================================
// This header provides a C interface to Metal device management for use from Zig.

#ifndef TALU_METAL_DEVICE_H
#define TALU_METAL_DEVICE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to a Metal device context.
typedef struct MetalDevice MetalDevice;

/// Opaque handle to a Metal buffer.
typedef struct MetalBuffer MetalBuffer;

/// Initialize Metal and create a device context.
/// Returns NULL if Metal is not available or initialization fails.
MetalDevice* metal_device_create(void);

/// Destroy Metal device context and release resources.
void metal_device_destroy(MetalDevice* device);

/// Check if Metal is available on this system.
bool metal_is_available(void);

/// Get device name (e.g., "Apple M2").
const char* metal_device_name(MetalDevice* device);

/// Allocate a Metal buffer with the given size.
/// Returns NULL on failure.
MetalBuffer* metal_buffer_create(MetalDevice* device, size_t size);

/// Copy data from CPU to Metal buffer.
void metal_buffer_upload(MetalBuffer* buffer, const void* data, size_t size);

/// Copy data from Metal buffer to CPU.
void metal_buffer_download(MetalBuffer* buffer, void* data, size_t size);

/// Get raw buffer pointer (for Metal-side operations).
void* metal_buffer_contents(MetalBuffer* buffer);

/// Destroy Metal buffer.
void metal_buffer_destroy(MetalBuffer* buffer);

/// Wait for all GPU operations to complete.
void metal_device_synchronize(MetalDevice* device);

#ifdef __cplusplus
}
#endif

#endif // TALU_METAL_DEVICE_H
