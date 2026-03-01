// =============================================================================
// Metal Matrix Multiplication - Objective-C Implementation
// =============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "matmul.h"
#include "device.h"

// Import internal device structure
struct MetalDevice {
    void* device;
    void* commandQueue;
};

bool metal_matmul_f32(
    MetalDevice* context,
    const float* a, size_t m, size_t k,
    const float* b, size_t n,
    float* c
) {
    if (!context || !a || !b || !c || m == 0 || k == 0 || n == 0) return false;

    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)(context->device);
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)(context->commandQueue);

        // For simplicity, transpose inputs to column-major, use MPS directly, then transpose output
        // Calculate aligned rowBytes and buffer sizes
        size_t rowBytes_a = ((m * sizeof(float) + 15) / 16) * 16;
        size_t rowBytes_b = ((k * sizeof(float) + 15) / 16) * 16;
        size_t rowBytes_c = ((m * sizeof(float) + 15) / 16) * 16;

        // Buffer sizes: rows * rowBytes (with column-major layout)
        size_t buffer_a_size = k * rowBytes_a;
        size_t buffer_b_size = n * rowBytes_b;
        size_t buffer_c_size = n * rowBytes_c;

        id<MTLBuffer> buffer_a = [device newBufferWithLength:buffer_a_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_b = [device newBufferWithLength:buffer_b_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_c = [device newBufferWithLength:buffer_c_size options:MTLResourceStorageModeShared];

        if (!buffer_a || !buffer_b || !buffer_c) return false;

        // Create matrix descriptors with explicit rowBytes
        // For column-major storage: A is [k rows, m cols], B is [n rows, k cols], C is [n rows, m cols]
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:k
                             columns:m
                            rowBytes:rowBytes_a
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:n
                             columns:k
                            rowBytes:rowBytes_b
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:n
                             columns:m
                            rowBytes:rowBytes_c
                            dataType:MPSDataTypeFloat32];

        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:buffer_a descriptor:descA];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:buffer_b descriptor:descB];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:buffer_c descriptor:descC];

        // Copy data (transpose row-major to column-major)
        float* a_data = (float*)[buffer_a contents];
        size_t a_stride = [descA rowBytes] / sizeof(float);
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < k; j++) {
                a_data[j * a_stride + i] = a[i * k + j];
            }
        }

        float* b_data = (float*)[buffer_b contents];
        size_t b_stride = [descB rowBytes] / sizeof(float);
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < n; j++) {
                b_data[j * b_stride + i] = b[i * n + j];
            }
        }

        // Column-major matmul: C^T = B^T @ A^T
        // MPS computes: result[resultRows, resultColumns] = left[resultRows, interior] @ right[interior, resultColumns]
        // We need: C^T[n, m] = B^T[n, k] @ A^T[k, m]
        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:NO
            transposeRight:NO
            resultRows:n
            resultColumns:m
            interiorColumns:k
            alpha:1.0
            beta:0.0];

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [matmul encodeToCommandBuffer:commandBuffer
                           leftMatrix:matrixB   // B^T
                          rightMatrix:matrixA   // A^T
                         resultMatrix:matrixC];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back (transpose column-major to row-major)
        float* c_data = (float*)[buffer_c contents];
        size_t c_stride = [descC rowBytes] / sizeof(float);
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < n; j++) {
                c[i * n + j] = c_data[j * c_stride + i];
            }
        }

        return true;
    }
}
