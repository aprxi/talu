//! Inference transport executors.
//!
//! Transport owns process-local byte movement for staged activation handoff.
//! Bridge modules provide the contracts and ordering; backend adapters provide
//! concrete CPU/CUDA/Metal payload operations.

pub const local_stage = @import("local_stage.zig");

pub const LocalStageTransportValidationError = local_stage.LocalStageTransportValidationError;
pub const LocalStageTransportRequest = local_stage.LocalStageTransportRequest;
pub const LocalStageTransportEntryFailure = local_stage.LocalStageTransportEntryFailure;
pub const LocalStageTransportFailureReport = local_stage.LocalStageTransportFailureReport;
pub const LocalStageTransportFailureCapture = local_stage.LocalStageTransportFailureCapture;

pub const executeLocalStageTransport = local_stage.executeLocalStageTransport;
pub const executeLocalStageTransportWithFailureCapture = local_stage.executeLocalStageTransportWithFailureCapture;

test "inference transport root exports local_stage contract" {
    _ = LocalStageTransportRequest;
    _ = LocalStageTransportValidationError;
    _ = executeLocalStageTransport;
}
