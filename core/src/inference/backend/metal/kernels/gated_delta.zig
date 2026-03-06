//! Metal gated-delta kernel surface.

pub const supported = false;

pub const UnsupportedError = error{UnsupportedKernel};

pub fn unsupported() UnsupportedError {
    return error.UnsupportedKernel;
}

pub fn requireImplemented() UnsupportedError {
    return error.UnsupportedKernel;
}

pub const GatedDeltaState = struct {};
pub const GatedDeltaScratch = struct {};
pub const MatmulScratch = struct {};

pub const GatedDeltaKernel = struct {
    pub const ForwardParams = struct {
        input_tensor: ?*anyopaque,
        output_tensor: ?*anyopaque,
        state: *GatedDeltaState,
        scratch: *GatedDeltaScratch,
        matmul_scratch: *MatmulScratch,
    };

    pub fn forward(
        self: *const GatedDeltaKernel,
        input_tensor: ?*anyopaque,
        output_tensor: ?*anyopaque,
        state: *GatedDeltaState,
        scratch: *GatedDeltaScratch,
        matmul_scratch: *MatmulScratch,
    ) !void {
        _ = self;
        _ = input_tensor;
        _ = output_tensor;
        _ = state;
        _ = scratch;
        _ = matmul_scratch;
        return error.UnsupportedModel;
    }
};
