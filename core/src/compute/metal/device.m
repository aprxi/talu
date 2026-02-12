// =============================================================================
// Metal Device Management - Objective-C Implementation
// =============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "device.h"

struct MetalDevice {
    void* device;
    void* commandQueue;
};

struct MetalBuffer {
    void* buffer;
    size_t size;
};

bool metal_is_available(void) {
    return MTLCopyAllDevices() != nil && [MTLCopyAllDevices() count] > 0;
}

MetalDevice* metal_device_create(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return NULL;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            return NULL;
        }

        MetalDevice* context = (MetalDevice*)malloc(sizeof(MetalDevice));
        context->device = (__bridge_retained void*)device;
        context->commandQueue = (__bridge_retained void*)commandQueue;

        return context;
    }
}

void metal_device_destroy(MetalDevice* device) {
    if (!device) return;

    @autoreleasepool {
        if (device->commandQueue) {
            CFRelease(device->commandQueue);
        }
        if (device->device) {
            CFRelease(device->device);
        }
        free(device);
    }
}

const char* metal_device_name(MetalDevice* device) {
    if (!device || !device->device) return "Unknown";

    // Static buffer to hold device name (survives autoreleasepool)
    static char device_name_buffer[256] = {0};

    @autoreleasepool {
        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)(device->device);
        NSString* name = [mtl_device name];
        if (name) {
            const char* utf8_name = [name UTF8String];
            if (utf8_name) {
                strncpy(device_name_buffer, utf8_name, sizeof(device_name_buffer) - 1);
                device_name_buffer[sizeof(device_name_buffer) - 1] = '\0';
                return device_name_buffer;
            }
        }
    }
    return "Unknown";
}

MetalBuffer* metal_buffer_create(MetalDevice* device, size_t size) {
    if (!device || !device->device || size == 0) return NULL;

    @autoreleasepool {
        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)(device->device);
        id<MTLBuffer> buffer = [mtl_device newBufferWithLength:size
                                                       options:MTLResourceStorageModeShared];
        if (!buffer) return NULL;

        MetalBuffer* metal_buffer = (MetalBuffer*)malloc(sizeof(MetalBuffer));
        metal_buffer->buffer = (__bridge_retained void*)buffer;
        metal_buffer->size = size;

        return metal_buffer;
    }
}

void metal_buffer_upload(MetalBuffer* buffer, const void* data, size_t size) {
    if (!buffer || !buffer->buffer || !data) return;

    @autoreleasepool {
        id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)(buffer->buffer);
        size_t copy_size = size < buffer->size ? size : buffer->size;
        memcpy([mtl_buffer contents], data, copy_size);
    }
}

void metal_buffer_download(MetalBuffer* buffer, void* data, size_t size) {
    if (!buffer || !buffer->buffer || !data) return;

    @autoreleasepool {
        id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)(buffer->buffer);
        size_t copy_size = size < buffer->size ? size : buffer->size;
        memcpy(data, [mtl_buffer contents], copy_size);
    }
}

void* metal_buffer_contents(MetalBuffer* buffer) {
    if (!buffer || !buffer->buffer) return NULL;

    @autoreleasepool {
        id<MTLBuffer> mtl_buffer = (__bridge id<MTLBuffer>)(buffer->buffer);
        return [mtl_buffer contents];
    }
}

void metal_buffer_destroy(MetalBuffer* buffer) {
    if (!buffer) return;

    @autoreleasepool {
        if (buffer->buffer) {
            CFRelease(buffer->buffer);
        }
        free(buffer);
    }
}

void metal_device_synchronize(MetalDevice* device) {
    if (!device || !device->commandQueue) return;

    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)(device->commandQueue);
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}
