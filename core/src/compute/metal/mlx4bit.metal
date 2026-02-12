#include <metal_stdlib>
using namespace metal;

// MLX 4-bit quantization format:
// - Nibbles packed 2 per byte (lower nibble first)
// - Nibbles represent values 0-15, stored as unsigned
// - Actual value = (nibble - 8) * scale + bias
// - Scales and biases are BF16 (16-bit brain float)
// - Group size = 32 (32 elements share same scale/bias)

inline float bf16_to_f32(uint16_t bf16) {
    // BF16 is upper 16 bits of IEEE 754 float32
    uint32_t bits = (uint32_t)bf16 << 16;
    return as_type<float>(bits);
}

inline float dequantize_mlx4bit(uint8_t nibble, uint16_t scale, uint16_t bias) {
    // Nibbles are 0-15, interpret as signed by subtracting 8
    float val = (float)nibble - 8.0f;
    return val * bf16_to_f32(scale) + bf16_to_f32(bias);
}

// Matrix multiply: C = A @ B
// A: [m x k] f32, row-major
// B: [k x n] mlx_4bit (packed nibbles + bf16 scales/biases), row-major
// C: [m x n] f32, row-major
//
// B storage:
// - b_data: packed nibbles (k * n / 2 bytes)
// - b_scales: BF16 scales ((k * n / 32) uint16_t)
// - b_biases: BF16 biases ((k * n / 32) uint16_t)
kernel void mlx4bit_matmul(
    constant float* a [[buffer(0)]],           // [m x k]
    constant uint8_t* b_data [[buffer(1)]],    // packed nibbles
    constant uint16_t* b_scales [[buffer(2)]], // bf16 scales
    constant uint16_t* b_biases [[buffer(3)]], // bf16 biases
    device float* c [[buffer(4)]],             // [m x n] output
    constant uint32_t& m [[buffer(5)]],
    constant uint32_t& k [[buffer(6)]],
    constant uint32_t& n [[buffer(7)]],
    constant uint32_t& group_size [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;  // output row (0..m-1)
    uint col = gid.x;  // output column (0..n-1)

    if (row >= m || col >= n) return;

    float sum = 0.0f;

    // Iterate over k dimension
    for (uint k_idx = 0; k_idx < k; k_idx++) {
        float a_val = a[row * k + k_idx];

        // B is stored row-major: b[i][col]
        // Position in B: (i * n + col)
        uint b_index = k_idx * n + col;

        // Get packed nibble
        uint byte_index = b_index / 2;
        uint nibble;
        if (b_index % 2 == 0) {
            nibble = b_data[byte_index] & 0x0F;  // lower nibble
        } else {
            nibble = (b_data[byte_index] >> 4) & 0x0F;  // upper nibble
        }

        // Get scale/bias for this element's group
        uint group_idx = b_index / group_size;
        uint16_t scale = b_scales[group_idx];
        uint16_t bias = b_biases[group_idx];

        // Dequantize and accumulate
        float b_val = dequantize_mlx4bit(nibble, scale, bias);
        sum += a_val * b_val;
    }

    c[row * n + col] = sum;
}
