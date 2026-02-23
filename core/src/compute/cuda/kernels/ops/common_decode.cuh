static __device__ __forceinline__ float talu_decode_f16_u16(unsigned short raw) {
    return __half2float(*reinterpret_cast<const __half*>(&raw));
}

static __device__ __forceinline__ float talu_decode_bf16_u16(unsigned short raw) {
    const unsigned int bits = static_cast<unsigned int>(raw) << 16;
    return __uint_as_float(bits);
}

static __device__ __forceinline__ float talu_decode_scale_bias_u16(unsigned short raw, unsigned int dtype_tag) {
    // dtype_tag: 0 => f16, 1 => bf16
    return (dtype_tag == 0) ? talu_decode_f16_u16(raw) : talu_decode_bf16_u16(raw);
}
