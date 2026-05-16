//! Local staged activation transport executor.
//!
//! This module validates bridge-owned handoff contracts and executes one
//! adjacent process-local activation transfer through transport endpoint
//! adapters. It does not allocate transport buffers or choose backend routes.

const std = @import("std");

const boundary_byte_image = @import("../bridge/boundary_byte_image.zig");
const host_capability = @import("../bridge/host_capability.zig");
const staged_error = @import("../bridge/staged_error.zig");
const stage_transport = @import("../bridge/stage_transport.zig");
const stage_transfer_mode = @import("../bridge/stage_transfer_mode.zig");
const state_ownership = @import("../bridge/state_ownership.zig");
const tensor_frame = @import("../bridge/tensor_frame.zig");

pub const LocalStageTransportValidationError =
    stage_transfer_mode.StageTransferModeError ||
    stage_transport.StageTransportError ||
    boundary_byte_image.BoundaryByteImageError ||
    tensor_frame.TensorFrameValidationError ||
    error{
        LocalStageTransportMetadataMismatch,
        LocalStageTransportDecisionMismatch,
        LocalStageTransportEnvelopeMismatch,
        LocalStageTransportPayloadByteCountMismatch,
        LocalStageTransportMissingStaging,
        LocalStageTransportBufferTooSmall,
        LocalStageTransportBorrowUnsupported,
        LocalStageTransportSegmentedUploadUnsupported,
        LocalStageTransportPeerCopyUnsupported,
        LocalStageTransportRemoteModeUnsupported,
    };

pub const StageExecutionFence = union(enum) {
    none,
};

pub const StageExecutionReceipt = struct {
    stage_id: usize,
    fence: StageExecutionFence = .none,

    pub fn completed(stage_id: usize) @This() {
        return .{ .stage_id = stage_id };
    }
};

pub const LocalStageTransportRequest = struct {
    placement_plan: *const host_capability.PlacementPlan,
    metadata: *const tensor_frame.TensorFrameMetadata,
    image: *const boundary_byte_image.BoundaryByteImageRef,
    decision: stage_transfer_mode.StageTransferModeDecision,
    envelope: *const stage_transport.StageTransportEnvelope,
    source_receipt: ?StageExecutionReceipt = null,
    staging: ?[]align(64) u8 = null,
    allow_borrow: bool = true,
    local_device_peer_copy_available: bool = false,
    state_ownership_plan: ?*const state_ownership.StageStateOwnershipPlan = null,
    cleanup_obligations: []const state_ownership.StateCleanupObligation = &.{},
};

pub const LocalStageTransportEndpointVTable = struct {
    synchronize: *const fn (*anyopaque, StageExecutionReceipt) anyerror!void,
    prepare_boundary_transfer_to: ?*const fn (*anyopaque, *anyopaque, *const tensor_frame.TensorFrameMetadata) anyerror!void = null,
    download_activation: *const fn (*anyopaque, []u8, usize) anyerror!void,
    upload_activation: *const fn (*anyopaque, []const u8, usize) anyerror!void,
    upload_activation_segments: ?*const fn (*anyopaque, []const []const u8, usize) anyerror!void = null,
    consume_borrowed_activation: ?*const fn (*anyopaque, []const u8, usize) anyerror!void = null,
    peer_copy_activation_to: ?*const fn (*anyopaque, *anyopaque, usize) anyerror!void = null,
    peer_copy_handles_stage_sync: ?*const fn (*anyopaque) bool = null,
};

pub const LocalStageTransportEndpoint = struct {
    stage_id: usize,
    ptr: *anyopaque,
    vtable: *const LocalStageTransportEndpointVTable,

    pub fn synchronize(self: *@This(), receipt: StageExecutionReceipt) anyerror!void {
        if (receipt.stage_id != self.stage_id) return error.StageTransferBoundaryMismatch;
        return self.vtable.synchronize(self.ptr, receipt);
    }

    pub fn prepareBoundaryTransferTo(
        self: *@This(),
        target: *@This(),
        metadata: *const tensor_frame.TensorFrameMetadata,
    ) anyerror!void {
        const prepare = self.vtable.prepare_boundary_transfer_to orelse return;
        return prepare(self.ptr, target.ptr, metadata);
    }

    pub fn downloadActivation(self: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
        return self.vtable.download_activation(self.ptr, host_buf, byte_count);
    }

    pub fn uploadActivation(self: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
        return self.vtable.upload_activation(self.ptr, host_buf, byte_count);
    }

    pub fn uploadActivationSegments(self: *@This(), host_segments: []const []const u8, byte_count: usize) anyerror!void {
        const upload_segments = self.vtable.upload_activation_segments orelse return error.LocalStageTransportSegmentedUploadUnsupported;
        return upload_segments(self.ptr, host_segments, byte_count);
    }

    pub fn consumeBorrowedActivation(self: *@This(), host_bytes: []const u8, byte_count: usize) anyerror!void {
        const consume = self.vtable.consume_borrowed_activation orelse return error.LocalStageTransportBorrowUnsupported;
        return consume(self.ptr, host_bytes, byte_count);
    }

    pub fn peerCopyActivationTo(self: *@This(), target: anytype, byte_count: usize) anyerror!void {
        const peer_copy = self.vtable.peer_copy_activation_to orelse return error.LocalStageTransportPeerCopyUnsupported;
        return peer_copy(self.ptr, target.ptr, byte_count);
    }

    pub fn peerCopyHandlesStageSync(self: *const @This()) bool {
        const handles_sync = self.vtable.peer_copy_handles_stage_sync orelse return false;
        return handles_sync(self.ptr);
    }
};

pub const LocalStageTransportEndpointRegistry = struct {
    endpoints: []LocalStageTransportEndpoint,

    pub fn endpointForStageId(self: *@This(), stage_id: usize) error{ DuplicateStageRef, MissingStageRef }!*LocalStageTransportEndpoint {
        var found: ?*LocalStageTransportEndpoint = null;
        for (self.endpoints) |*endpoint| {
            if (endpoint.stage_id != stage_id) continue;
            if (found != null) return error.DuplicateStageRef;
            found = endpoint;
        }
        return found orelse error.MissingStageRef;
    }
};

pub fn localStageTransportAdapter(comptime Stage: type, stage_id: usize, stage: *Stage) LocalStageTransportEndpoint {
    const Adapter = struct {
        fn stagePtr(ptr: *anyopaque) *Stage {
            return @ptrCast(@alignCast(ptr));
        }

        fn synchronize(ptr: *anyopaque, receipt: StageExecutionReceipt) anyerror!void {
            _ = receipt;
            if (comptime @hasDecl(Stage, "synchronize")) {
                return stagePtr(ptr).synchronize();
            }
        }

        fn prepareBoundaryTransferTo(
            ptr: *anyopaque,
            target_ptr: *anyopaque,
            metadata: *const tensor_frame.TensorFrameMetadata,
        ) anyerror!void {
            if (comptime @hasDecl(Stage, "prepareBoundaryTransferToErased")) {
                return stagePtr(ptr).prepareBoundaryTransferToErased(target_ptr, metadata);
            }
        }

        fn downloadActivation(ptr: *anyopaque, host_buf: []u8, byte_count: usize) anyerror!void {
            if (comptime @hasDecl(Stage, "downloadActivation")) {
                return stagePtr(ptr).downloadActivation(host_buf, byte_count);
            }
            return error.InvalidTopologyConfig;
        }

        fn uploadActivation(ptr: *anyopaque, host_buf: []const u8, byte_count: usize) anyerror!void {
            if (comptime @hasDecl(Stage, "uploadActivation")) {
                return stagePtr(ptr).uploadActivation(host_buf, byte_count);
            }
            return error.InvalidTopologyConfig;
        }

        fn uploadActivationSegments(ptr: *anyopaque, host_segments: []const []const u8, byte_count: usize) anyerror!void {
            if (comptime @hasDecl(Stage, "uploadActivationSegments")) {
                return stagePtr(ptr).uploadActivationSegments(host_segments, byte_count);
            }
            return error.LocalStageTransportSegmentedUploadUnsupported;
        }

        fn consumeBorrowedActivation(ptr: *anyopaque, host_bytes: []const u8, byte_count: usize) anyerror!void {
            if (comptime @hasDecl(Stage, "consumeBorrowedActivation")) {
                return stagePtr(ptr).consumeBorrowedActivation(host_bytes, byte_count);
            }
            return error.LocalStageTransportBorrowUnsupported;
        }

        fn peerCopyActivationTo(ptr: *anyopaque, target_ptr: *anyopaque, byte_count: usize) anyerror!void {
            if (comptime @hasDecl(Stage, "peerCopyActivationToErased")) {
                return stagePtr(ptr).peerCopyActivationToErased(target_ptr, byte_count);
            }
            return error.LocalStageTransportPeerCopyUnsupported;
        }

        fn peerCopyHandlesStageSync(ptr: *anyopaque) bool {
            if (comptime @hasDecl(Stage, "peerCopyHandlesStageSync")) {
                return stagePtr(ptr).peerCopyHandlesStageSync();
            }
            return false;
        }

        const vtable = LocalStageTransportEndpointVTable{
            .synchronize = synchronize,
            .prepare_boundary_transfer_to = if (@hasDecl(Stage, "prepareBoundaryTransferToErased")) prepareBoundaryTransferTo else null,
            .download_activation = downloadActivation,
            .upload_activation = uploadActivation,
            .upload_activation_segments = if (@hasDecl(Stage, "uploadActivationSegments")) uploadActivationSegments else null,
            .consume_borrowed_activation = if (@hasDecl(Stage, "consumeBorrowedActivation")) consumeBorrowedActivation else null,
            .peer_copy_activation_to = if (@hasDecl(Stage, "peerCopyActivationToErased")) peerCopyActivationTo else null,
            .peer_copy_handles_stage_sync = if (@hasDecl(Stage, "peerCopyHandlesStageSync")) peerCopyHandlesStageSync else null,
        };
    };
    return .{
        .stage_id = stage_id,
        .ptr = stage,
        .vtable = &Adapter.vtable,
    };
}

pub const LocalStageTransportEntryFailure = struct {
    primary_failure: staged_error.StagedFailure,
    touched_stages: []const staged_error.TouchedStageCleanupRef,
    cleanup_plan: ?staged_error.StagedCleanupPlan = null,
    cleanup_report: ?staged_error.StagedCleanupReport = null,
    error_report: staged_error.StagedErrorReport,
};

pub const LocalStageTransportFailureReport = struct {
    allocator: std.mem.Allocator,
    source_error: anyerror,
    entries: []LocalStageTransportEntryFailure,

    pub fn deinit(self: *@This()) void {
        for (self.entries) |*entry| {
            entry.error_report.deinit();
            if (entry.cleanup_report) |*report| report.deinit();
            if (entry.cleanup_plan) |*plan| plan.deinit();
            self.allocator.free(entry.touched_stages);
        }
        self.allocator.free(self.entries);
        self.* = undefined;
    }
};

pub const LocalStageTransportFailureCapture = struct {
    allocator: std.mem.Allocator,
    report: ?LocalStageTransportFailureReport = null,

    pub fn init(allocator: std.mem.Allocator) @This() {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *@This()) void {
        if (self.report) |*report| report.deinit();
        self.report = null;
    }
};

pub fn executeLocalStageTransport(
    comptime SourceStage: type,
    comptime TargetStage: type,
    source: *SourceStage,
    target: *TargetStage,
    request: LocalStageTransportRequest,
) anyerror!void {
    return executeLocalStageTransportWithFailureCapture(
        SourceStage,
        TargetStage,
        source,
        target,
        request,
        null,
    );
}

pub fn executeLocalStageTransportWithFailureCapture(
    comptime SourceStage: type,
    comptime TargetStage: type,
    source: *SourceStage,
    target: *TargetStage,
    request: LocalStageTransportRequest,
    failure_capture: ?*LocalStageTransportFailureCapture,
) anyerror!void {
    comptime {
        requireStageAdapter(SourceStage);
        requireStageAdapter(TargetStage);
    }

    host_capability.validatePlacementPlan(request.placement_plan) catch |err| {
        return failWithCapture(failure_capture, request, err, .validation_before_mutation, .validation, .none);
    };
    request.metadata.validate() catch |err| {
        return failWithCapture(failure_capture, request, err, .validation_before_mutation, .validation, .none);
    };
    boundary_byte_image.validateBoundaryByteImage(request.image, .{}) catch |err| {
        return failWithCapture(failure_capture, request, err, .validation_before_mutation, .validation, .none);
    };
    stage_transport.validateStageTransportEnvelope(request.envelope) catch |err| {
        return failWithCapture(failure_capture, request, err, .validation_before_mutation, .validation, .none);
    };

    if (request.image.metadata != request.metadata) {
        return failWithCapture(failure_capture, request, error.LocalStageTransportMetadataMismatch, .validation_before_mutation, .validation, .none);
    }
    if (request.envelope.kind != .activation_payload) {
        return failWithCapture(failure_capture, request, error.LocalStageTransportEnvelopeMismatch, .validation_before_mutation, .validation, .none);
    }
    if (request.envelope.transfer_mode.? != request.decision.mode) {
        return failWithCapture(failure_capture, request, error.LocalStageTransportDecisionMismatch, .validation_before_mutation, .validation, .none);
    }
    if (request.envelope.payload_byte_count != request.image.byte_count) {
        return failWithCapture(failure_capture, request, error.LocalStageTransportPayloadByteCountMismatch, .validation_before_mutation, .validation, .none);
    }

    const batch_entry_count = std.math.cast(u64, request.metadata.batch.entries.len) orelse {
        return failWithCapture(failure_capture, request, error.LocalStageTransportEnvelopeMismatch, .validation_before_mutation, .validation, .none);
    };
    if (request.envelope.batch_entry_count != batch_entry_count) {
        return failWithCapture(failure_capture, request, error.LocalStageTransportEnvelopeMismatch, .validation_before_mutation, .validation, .none);
    }
    if (request.metadata.batch.entries.len == 1) {
        if (request.envelope.activation_scope.? != .single_entry_header) {
            return failWithCapture(failure_capture, request, error.LocalStageTransportEnvelopeMismatch, .validation_before_mutation, .validation, .none);
        }
    } else {
        if (request.envelope.activation_scope.? != .multi_entry_local) {
            return failWithCapture(failure_capture, request, error.LocalStageTransportEnvelopeMismatch, .validation_before_mutation, .validation, .none);
        }
    }

    const recomputed = stage_transfer_mode.chooseStageTransferMode(.{
        .placement_plan = request.placement_plan,
        .metadata = request.metadata,
        .image = request.image,
        .allow_borrow = request.allow_borrow,
        .local_device_peer_copy_available = request.local_device_peer_copy_available,
    }) catch |err| {
        return failWithCapture(failure_capture, request, err, .validation_before_mutation, .validation, .none);
    };
    if (!transferDecisionsEql(recomputed, request.decision)) {
        return failWithCapture(failure_capture, request, error.LocalStageTransportDecisionMismatch, .validation_before_mutation, .validation, .none);
    }

    const byte_count = std.math.cast(usize, request.image.byte_count) orelse {
        return failWithCapture(failure_capture, request, error.LocalStageTransportPayloadByteCountMismatch, .validation_before_mutation, .validation, .none);
    };

    switch (request.decision.mode) {
        .borrow_in_process => {
            const host_bytes = request.image.host_bytes orelse {
                return failWithCapture(failure_capture, request, error.LocalStageTransportBorrowUnsupported, .validation_before_mutation, .validation, .none);
            };
            if (comptime !@hasDecl(TargetStage, "consumeBorrowedActivation")) {
                return failWithCapture(failure_capture, request, error.LocalStageTransportBorrowUnsupported, .validation_before_mutation, .validation, .none);
            }
            try prepareSourceBoundaryTransfer(SourceStage, TargetStage, source, target, request, failure_capture, true);
            target.consumeBorrowedActivation(host_bytes, byte_count) catch |err| {
                return failWithCapture(failure_capture, request, err, .frame_handoff, .transport, .source_target);
            };
        },
        .copy_in_process => {
            if (request.image.host_bytes) |host_bytes| {
                try prepareSourceBoundaryTransfer(SourceStage, TargetStage, source, target, request, failure_capture, true);
                target.uploadActivation(host_bytes, byte_count) catch |err| {
                    return failWithCapture(failure_capture, request, err, .frame_handoff, .transport, .source_target);
                };
            } else if (request.image.host_segments) |host_segments| {
                if (comptime !@hasDecl(TargetStage, "uploadActivationSegments")) {
                    return failWithCapture(failure_capture, request, error.LocalStageTransportSegmentedUploadUnsupported, .validation_before_mutation, .validation, .none);
                }
                try prepareSourceBoundaryTransfer(SourceStage, TargetStage, source, target, request, failure_capture, true);
                target.uploadActivationSegments(host_segments, byte_count) catch |err| {
                    return failWithCapture(failure_capture, request, err, .frame_handoff, .transport, .source_target);
                };
            } else {
                return failWithCapture(failure_capture, request, error.LocalStageTransportDecisionMismatch, .validation_before_mutation, .validation, .none);
            }
        },
        .device_download_then_copy => {
            const staging = request.staging orelse {
                return failWithCapture(failure_capture, request, error.LocalStageTransportMissingStaging, .validation_before_mutation, .validation, .none);
            };
            if (byte_count > staging.len) {
                return failWithCapture(failure_capture, request, error.LocalStageTransportBufferTooSmall, .validation_before_mutation, .validation, .none);
            }
            const transfer_buf = staging[0..byte_count];
            try prepareSourceBoundaryTransfer(SourceStage, TargetStage, source, target, request, failure_capture, true);
            source.downloadActivation(transfer_buf, byte_count) catch |err| {
                return failWithCapture(failure_capture, request, err, .frame_handoff, .transport, .source);
            };
            target.uploadActivation(transfer_buf, byte_count) catch |err| {
                return failWithCapture(failure_capture, request, err, .frame_handoff, .transport, .source_target);
            };
        },
        .device_peer_copy_in_process => {
            if (comptime !@hasDecl(SourceStage, "peerCopyActivationTo")) {
                return failWithCapture(failure_capture, request, error.LocalStageTransportPeerCopyUnsupported, .validation_before_mutation, .validation, .none);
            }
            const peer_copy_handles_sync = if (comptime @hasDecl(SourceStage, "peerCopyHandlesStageSync"))
                source.peerCopyHandlesStageSync()
            else
                false;
            try prepareSourceBoundaryTransfer(SourceStage, TargetStage, source, target, request, failure_capture, !peer_copy_handles_sync);
            source.peerCopyActivationTo(target, byte_count) catch |err| {
                return failWithCapture(failure_capture, request, err, .frame_handoff, .transport, .source_target);
            };
        },
        .remote_stream, .device_download_then_remote_stream => {
            return failWithCapture(failure_capture, request, error.LocalStageTransportRemoteModeUnsupported, .validation_before_mutation, .validation, .none);
        },
    }
}

fn prepareSourceBoundaryTransfer(
    comptime SourceStage: type,
    comptime TargetStage: type,
    source: *SourceStage,
    target: *TargetStage,
    request: LocalStageTransportRequest,
    failure_capture: ?*LocalStageTransportFailureCapture,
    synchronize_source: bool,
) anyerror!void {
    if (synchronize_source) {
        const receipt = request.source_receipt orelse StageExecutionReceipt.completed(request.metadata.boundary.source_stage_id);
        synchronizeSource(SourceStage, source, receipt) catch |err| {
            return failWithCapture(failure_capture, request, err, .frame_handoff, .transport, .source);
        };
    }
    if (comptime @hasDecl(SourceStage, "prepareBoundaryTransferTo")) {
        source.prepareBoundaryTransferTo(target, request.metadata) catch |err| {
            return failWithCapture(failure_capture, request, err, .frame_handoff, .transport, .source_target);
        };
    }
}

fn synchronizeSource(
    comptime SourceStage: type,
    source: *SourceStage,
    receipt: StageExecutionReceipt,
) anyerror!void {
    const synchronize_info = @typeInfo(@TypeOf(SourceStage.synchronize)).@"fn";
    if (comptime synchronize_info.params.len == 2) {
        return source.synchronize(receipt);
    }
    return source.synchronize();
}

fn requireStageAdapter(comptime Stage: type) void {
    inline for (.{ "synchronize", "downloadActivation", "uploadActivation" }) |name| {
        if (!@hasDecl(Stage, name)) {
            @compileError("local stage transport adapter missing required method: " ++ name);
        }
    }
}

fn transferDecisionsEql(
    lhs: stage_transfer_mode.StageTransferModeDecision,
    rhs: stage_transfer_mode.StageTransferModeDecision,
) bool {
    return lhs.mode == rhs.mode and
        boundaryFrameProfilesEql(lhs.boundary_profile, rhs.boundary_profile) and
        lhs.source_host_id.value == rhs.source_host_id.value and
        lhs.target_host_id.value == rhs.target_host_id.value;
}

fn boundaryFrameProfilesEql(
    lhs: host_capability.BoundaryFrameProfile,
    rhs: host_capability.BoundaryFrameProfile,
) bool {
    return lhs.boundary_index == rhs.boundary_index and
        lhs.source_stage_id == rhs.source_stage_id and
        lhs.target_stage_id == rhs.target_stage_id and
        lhs.tensor_frame_contract_version == rhs.tensor_frame_contract_version and
        lhs.step_kind == rhs.step_kind and
        lhs.dtype == rhs.dtype and
        lhs.layout == rhs.layout and
        lhs.max_batch_entries == rhs.max_batch_entries and
        lhs.max_token_count_per_frame == rhs.max_token_count_per_frame and
        lhs.max_activation_payload_bytes == rhs.max_activation_payload_bytes and
        lhs.handoff_mode == rhs.handoff_mode;
}

const FailureSource = enum {
    validation,
    transport,
};

const TouchedSide = enum {
    none,
    source,
    source_target,
};

fn failWithCapture(
    failure_capture: ?*LocalStageTransportFailureCapture,
    request: LocalStageTransportRequest,
    source_error: anyerror,
    phase: staged_error.StagedFailurePhase,
    failure_source: FailureSource,
    touched_side: TouchedSide,
) anyerror {
    if (failure_capture) |capture| {
        captureLocalStageTransportFailure(
            capture,
            request,
            source_error,
            phase,
            failure_source,
            touched_side,
        ) catch {};
    }
    return source_error;
}

fn captureLocalStageTransportFailure(
    capture: *LocalStageTransportFailureCapture,
    request: LocalStageTransportRequest,
    source_error: anyerror,
    phase: staged_error.StagedFailurePhase,
    failure_source: FailureSource,
    touched_side: TouchedSide,
) !void {
    capture.deinit();
    const entries = try capture.allocator.alloc(LocalStageTransportEntryFailure, request.metadata.batch.entries.len);
    errdefer capture.allocator.free(entries);

    for (request.metadata.batch.entries, 0..) |batch_entry, index| {
        entries[index] = try buildEntryFailure(
            capture.allocator,
            request,
            batch_entry,
            source_error,
            phase,
            failure_source,
            touched_side,
        );
        errdefer deinitEntryFailure(capture.allocator, &entries[index]);
    }

    capture.report = .{
        .allocator = capture.allocator,
        .source_error = source_error,
        .entries = entries,
    };
}

fn buildEntryFailure(
    allocator: std.mem.Allocator,
    request: LocalStageTransportRequest,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
    source_error: anyerror,
    phase: staged_error.StagedFailurePhase,
    failure_source: FailureSource,
    touched_side: TouchedSide,
) !LocalStageTransportEntryFailure {
    const primary_failure = try staged_error.buildStagedFailure(.{
        .kind = failureKind(source_error, failure_source),
        .phase = phase,
        .scope = failureScope(source_error, failure_source),
        .context = try failureContext(request, batch_entry),
        .source = failureSourceError(source_error, failure_source),
    }, .{
        .placement_plan = request.placement_plan,
        .state_ownership_plan = request.state_ownership_plan,
    });

    const touched = try buildTouchedStageRefs(allocator, request, batch_entry, touched_side);
    errdefer allocator.free(touched);

    var cleanup_plan_opt: ?staged_error.StagedCleanupPlan = null;
    var cleanup_report_opt: ?staged_error.StagedCleanupReport = null;
    errdefer if (cleanup_report_opt) |*report| report.deinit();
    errdefer if (cleanup_plan_opt) |*plan| plan.deinit();

    var cleanup_obligation_buffer = try buildCleanupObligationsForEntry(allocator, request, batch_entry, touched);
    defer cleanup_obligation_buffer.deinit();
    const cleanup_obligations = cleanup_obligation_buffer.obligations;

    if (staged_error.stagedCleanupRequired(primary_failure, touched, cleanup_obligations)) {
        cleanup_plan_opt = try staged_error.buildStagedCleanupPlan(allocator, .{
            .primary_failure = primary_failure,
            .request_id = batch_entry.request_id,
            .placement_plan = request.placement_plan,
            .state_ownership_plan = request.state_ownership_plan,
            .touched_stages = touched,
            .cleanup_obligations = cleanup_obligations,
        });
        if (cleanup_plan_opt.?.steps.len == 0) {
            cleanup_report_opt = try staged_error.buildStagedCleanupReport(allocator, &.{}, .{
                .cleanup_plan = &cleanup_plan_opt.?,
                .primary_failure = &primary_failure,
            });
        }
    }

    const cleanup_plan_id = if (cleanup_plan_opt) |plan| plan.plan_id else null;
    var error_report = try staged_error.buildStagedErrorReport(allocator, primary_failure, cleanup_plan_id, &.{}, .{
        .placement_plan = request.placement_plan,
        .state_ownership_plan = request.state_ownership_plan,
        .cleanup_plan = if (cleanup_plan_opt) |*plan| plan else null,
    });
    errdefer error_report.deinit();

    return .{
        .primary_failure = primary_failure,
        .touched_stages = touched,
        .cleanup_plan = cleanup_plan_opt,
        .cleanup_report = cleanup_report_opt,
        .error_report = error_report,
    };
}

fn deinitEntryFailure(allocator: std.mem.Allocator, entry: *LocalStageTransportEntryFailure) void {
    entry.error_report.deinit();
    if (entry.cleanup_report) |*report| report.deinit();
    if (entry.cleanup_plan) |*plan| plan.deinit();
    allocator.free(entry.touched_stages);
}

fn failureKind(source_error: anyerror, failure_source: FailureSource) staged_error.StagedFailureKind {
    if (source_error == error.OutOfMemory) return .resource_exhausted;
    if (source_error == error.RequestCancelled) return .request_cancelled;
    return switch (failure_source) {
        .validation => .internal_contract_violation,
        .transport => .transfer_failed,
    };
}

fn failureScope(source_error: anyerror, failure_source: FailureSource) staged_error.StagedFailureScope {
    if (source_error == error.RequestCancelled) return .request;
    return switch (failure_source) {
        .validation, .transport => .transport,
    };
}

fn failureSourceError(source_error: anyerror, failure_source: FailureSource) staged_error.StagedSourceError {
    return .{
        .domain = switch (failure_source) {
            .validation => .internal,
            .transport => .transport,
        },
        .source_error_name = @errorName(source_error),
    };
}

fn failureContext(
    request: LocalStageTransportRequest,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
) !staged_error.StagedErrorContext {
    const boundary = request.metadata.boundary;
    const source_binding = try host_capability.bindingForStage(request.placement_plan, boundary.source_stage_id);
    const target_binding = try host_capability.bindingForStage(request.placement_plan, boundary.target_stage_id);
    return .{
        .graph_digest = request.placement_plan.graph_digest,
        .graph_contract_version = request.placement_plan.graph_contract_version,
        .stage_plan_contract_version = request.placement_plan.stage_plan_contract_version,
        .stage_plan_id = request.placement_plan.stage_plan_id,
        .placement_plan_id = request.placement_plan.plan_id,
        .state_ownership_plan_id = if (request.state_ownership_plan) |plan| plan.plan_id else request.placement_plan.state_ownership_plan_id,
        .tensor_frame_id = request.metadata.frame_id,
        .boundary_index = boundary.boundary_index,
        .source_stage_id = boundary.source_stage_id,
        .target_stage_id = boundary.target_stage_id,
        .source_host_id = source_binding.host_id,
        .target_host_id = target_binding.host_id,
        .request_id = batch_entry.request_id,
        .slot_id = batch_entry.slot_id,
        .state_epoch = batch_entry.state_epoch,
    };
}

fn buildTouchedStageRefs(
    allocator: std.mem.Allocator,
    request: LocalStageTransportRequest,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
    touched_side: TouchedSide,
) ![]staged_error.TouchedStageCleanupRef {
    const count: usize = switch (touched_side) {
        .none => 0,
        .source => 1,
        .source_target => 2,
    };
    const touched = try allocator.alloc(staged_error.TouchedStageCleanupRef, count);
    errdefer allocator.free(touched);
    if (count >= 1) {
        touched[0] = .{
            .stage_id = request.metadata.boundary.source_stage_id,
            .request_id = batch_entry.request_id,
            .slot_id = batch_entry.slot_id,
            .state_epoch = batch_entry.state_epoch,
        };
    }
    if (count >= 2) {
        touched[1] = .{
            .stage_id = request.metadata.boundary.target_stage_id,
            .request_id = batch_entry.request_id,
            .slot_id = batch_entry.slot_id,
            .state_epoch = batch_entry.state_epoch,
        };
    }
    return touched;
}

const CleanupObligationBuffer = struct {
    allocator: std.mem.Allocator,
    obligations: []const state_ownership.StateCleanupObligation = &.{},

    fn deinit(self: *@This()) void {
        if (self.obligations.len != 0) self.allocator.free(self.obligations);
        self.* = undefined;
    }
};

fn buildCleanupObligationsForEntry(
    allocator: std.mem.Allocator,
    request: LocalStageTransportRequest,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
    touched_stages: []const staged_error.TouchedStageCleanupRef,
) !CleanupObligationBuffer {
    const obligations = if (request.cleanup_obligations.len != 0)
        try copyCleanupObligationsForEntry(allocator, request.cleanup_obligations, batch_entry)
    else if (request.state_ownership_plan) |plan|
        try deriveCleanupObligationsForTouched(allocator, plan, touched_stages)
    else
        &.{};
    return .{
        .allocator = allocator,
        .obligations = obligations,
    };
}

fn copyCleanupObligationsForEntry(
    allocator: std.mem.Allocator,
    obligations: []const state_ownership.StateCleanupObligation,
    batch_entry: tensor_frame.TensorFrameBatchEntry,
) ![]state_ownership.StateCleanupObligation {
    var count: usize = 0;
    for (obligations) |obligation| {
        if (obligation.request_id == batch_entry.request_id and obligation.slot_id == batch_entry.slot_id) {
            count += 1;
        }
    }
    if (count == 0) return &.{};

    const filtered = try allocator.alloc(state_ownership.StateCleanupObligation, count);
    var index: usize = 0;
    for (obligations) |obligation| {
        if (obligation.request_id == batch_entry.request_id and obligation.slot_id == batch_entry.slot_id) {
            filtered[index] = obligation;
            index += 1;
        }
    }
    return filtered;
}

fn deriveCleanupObligationsForTouched(
    allocator: std.mem.Allocator,
    plan: *const state_ownership.StageStateOwnershipPlan,
    touched_stages: []const staged_error.TouchedStageCleanupRef,
) ![]state_ownership.StateCleanupObligation {
    if (touched_stages.len == 0) return &.{};

    const targets = try allocator.alloc(state_ownership.StageStateCleanupTarget, touched_stages.len);
    defer allocator.free(targets);
    var obligation_capacity: usize = 0;
    for (touched_stages, 0..) |touched, index| {
        const descriptors = try state_ownership.descriptorSetForStage(plan, touched.stage_id);
        obligation_capacity += descriptors.descriptors.len;
        targets[index] = .{
            .stage_id = touched.stage_id,
            .request_id = touched.request_id,
            .slot_id = touched.slot_id,
        };
    }
    if (obligation_capacity == 0) return &.{};

    const scratch = try allocator.alloc(state_ownership.StateCleanupObligation, obligation_capacity);
    defer allocator.free(scratch);
    const obligations = try state_ownership.buildStateCleanupObligations(plan, targets, scratch);
    if (obligations.len == 0) return &.{};
    return allocator.dupe(state_ownership.StateCleanupObligation, obligations);
}

const TraceEvent = enum {
    source_synchronize,
    source_prepare,
    source_download,
    source_peer_copy,
    target_upload,
    target_borrow,
    target_upload_segments,
};

const Trace = struct {
    events: [16]TraceEvent = undefined,
    count: usize = 0,

    fn record(self: *Trace, event: TraceEvent) void {
        self.events[self.count] = event;
        self.count += 1;
    }

    fn expect(self: *const Trace, expected: []const TraceEvent) !void {
        try std.testing.expectEqual(expected.len, self.count);
        for (expected, 0..) |event, index| {
            try std.testing.expectEqual(event, self.events[index]);
        }
    }
};

const TestSourceStage = struct {
    trace: *Trace,
    fail_synchronize: ?anyerror = null,
    fail_download: ?anyerror = null,
    peer_handles_sync: bool = false,
    peer_copy_byte_count: usize = 0,

    pub fn synchronize(self: *@This()) anyerror!void {
        self.trace.record(.source_synchronize);
        if (self.fail_synchronize) |err| return err;
    }

    pub fn downloadActivation(self: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
        self.trace.record(.source_download);
        if (self.fail_download) |err| return err;
        @memset(host_buf[0..byte_count], 0x5a);
    }

    pub fn uploadActivation(self: *@This(), _: []const u8, _: usize) anyerror!void {
        _ = self;
        unreachable;
    }

    pub fn peerCopyActivationTo(self: *@This(), target: anytype, byte_count: usize) anyerror!void {
        _ = target;
        self.trace.record(.source_peer_copy);
        self.peer_copy_byte_count = byte_count;
    }

    pub fn peerCopyHandlesStageSync(self: *const @This()) bool {
        return self.peer_handles_sync;
    }
};

const TestSourceStageNoPeer = struct {
    trace: *Trace,

    pub fn synchronize(self: *@This()) anyerror!void {
        self.trace.record(.source_synchronize);
    }

    pub fn downloadActivation(self: *@This(), _: []u8, _: usize) anyerror!void {
        self.trace.record(.source_download);
    }

    pub fn uploadActivation(self: *@This(), _: []const u8, _: usize) anyerror!void {
        _ = self;
        unreachable;
    }
};

const TestPreparingSourceStage = struct {
    trace: *Trace,
    fail_prepare: ?anyerror = null,
    prepared_slot_id: u64 = 0,

    pub fn synchronize(self: *@This()) anyerror!void {
        self.trace.record(.source_synchronize);
    }

    pub fn prepareBoundaryTransferTo(
        self: *@This(),
        target: anytype,
        metadata: *const tensor_frame.TensorFrameMetadata,
    ) anyerror!void {
        _ = target;
        self.trace.record(.source_prepare);
        self.prepared_slot_id = metadata.batch.entries[0].slot_id;
        if (self.fail_prepare) |err| return err;
    }

    pub fn downloadActivation(self: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
        self.trace.record(.source_download);
        @memset(host_buf[0..byte_count], 0x5a);
    }

    pub fn uploadActivation(self: *@This(), _: []const u8, _: usize) anyerror!void {
        _ = self;
        unreachable;
    }
};

const TestTargetStage = struct {
    trace: *Trace,
    upload_byte_count: usize = 0,
    upload_first_byte: u8 = 0,
    borrowed_byte_count: usize = 0,
    segment_byte_count: usize = 0,
    fail_upload: ?anyerror = null,
    fail_upload_segments: ?anyerror = null,

    pub fn synchronize(self: *@This()) anyerror!void {
        _ = self;
        unreachable;
    }

    pub fn downloadActivation(self: *@This(), _: []u8, _: usize) anyerror!void {
        _ = self;
        unreachable;
    }

    pub fn uploadActivation(self: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
        self.trace.record(.target_upload);
        if (self.fail_upload) |err| return err;
        self.upload_byte_count = byte_count;
        self.upload_first_byte = host_buf[0];
    }

    pub fn consumeBorrowedActivation(self: *@This(), host_bytes: []const u8, byte_count: usize) anyerror!void {
        self.trace.record(.target_borrow);
        self.borrowed_byte_count = byte_count;
        self.upload_first_byte = host_bytes[0];
    }

    pub fn uploadActivationSegments(self: *@This(), host_segments: []const []const u8, byte_count: usize) anyerror!void {
        self.trace.record(.target_upload_segments);
        if (self.fail_upload_segments) |err| return err;
        self.segment_byte_count = byte_count;
        self.upload_first_byte = host_segments[0][0];
    }
};

const TestTargetStageNoBorrow = struct {
    trace: *Trace,

    pub fn synchronize(self: *@This()) anyerror!void {
        _ = self;
        unreachable;
    }

    pub fn downloadActivation(self: *@This(), _: []u8, _: usize) anyerror!void {
        _ = self;
        unreachable;
    }

    pub fn uploadActivation(self: *@This(), _: []const u8, _: usize) anyerror!void {
        self.trace.record(.target_upload);
    }

    pub fn uploadActivationSegments(self: *@This(), _: []const []const u8, _: usize) anyerror!void {
        self.trace.record(.target_upload_segments);
    }
};

const TestTargetStageNoSegments = struct {
    trace: *Trace,

    pub fn synchronize(self: *@This()) anyerror!void {
        _ = self;
        unreachable;
    }

    pub fn downloadActivation(self: *@This(), _: []u8, _: usize) anyerror!void {
        _ = self;
        unreachable;
    }

    pub fn uploadActivation(self: *@This(), _: []const u8, _: usize) anyerror!void {
        self.trace.record(.target_upload);
    }

    pub fn consumeBorrowedActivation(self: *@This(), _: []const u8, _: usize) anyerror!void {
        self.trace.record(.target_borrow);
    }
};

const TestBundle = struct {
    placement: host_capability.PlacementPlan,
    metadata: *tensor_frame.TensorFrameMetadata,
    image: boundary_byte_image.BoundaryByteImageRef,
    decision: stage_transfer_mode.StageTransferModeDecision,
    envelope: stage_transport.StageTransportEnvelope,

    fn deinit(self: *TestBundle) void {
        self.placement.deinit();
        self.* = undefined;
    }

    fn request(self: *const TestBundle) LocalStageTransportRequest {
        return .{
            .placement_plan = &self.placement,
            .metadata = self.metadata,
            .image = &self.image,
            .decision = self.decision,
            .envelope = &self.envelope,
        };
    }
};

const test_decode_entry = [_]tensor_frame.TensorFrameBatchEntry{.{
    .batch_index = 0,
    .request_id = 101,
    .slot_id = 88,
    .sequence_start = 12,
    .token_count = 1,
}};

const test_decode_entries_two = [_]tensor_frame.TensorFrameBatchEntry{
    .{ .batch_index = 0, .request_id = 101, .slot_id = 88, .sequence_start = 12, .token_count = 1 },
    .{ .batch_index = 1, .request_id = 102, .slot_id = 89, .sequence_start = 12, .token_count = 1 },
};

const TestImageKind = enum { host, segments, device };

fn buildTestBundle(
    allocator: std.mem.Allocator,
    handoff_mode: host_capability.BoundaryHandoffMode,
    image_kind: TestImageKind,
    allow_borrow: bool,
    peer_copy_available: bool,
) !TestBundle {
    return buildTestBundleWithBackends(
        allocator,
        handoff_mode,
        image_kind,
        allow_borrow,
        peer_copy_available,
        .cuda,
        .cuda,
        .{ .cuda = 0 },
    );
}

fn buildTestBundleWithBackends(
    allocator: std.mem.Allocator,
    handoff_mode: host_capability.BoundaryHandoffMode,
    image_kind: TestImageKind,
    allow_borrow: bool,
    peer_copy_available: bool,
    source_backend_kind: host_capability.HostBackendKind,
    target_backend_kind: host_capability.HostBackendKind,
    device_location_hint: tensor_frame.TensorFramePayloadLocationHint,
) !TestBundle {
    var placement = try buildTestPlacementWithBackends(allocator, handoff_mode, source_backend_kind, target_backend_kind);
    errdefer placement.deinit();
    const entries = if (image_kind == .segments) &test_decode_entries_two else &test_decode_entry;
    const arena_allocator = placement.arena.allocator();
    const metadata = try arena_allocator.create(tensor_frame.TensorFrameMetadata);
    metadata.* = try testMetadata(&placement, entries, if (image_kind == .segments) .{ 2, 1, 4, 0 } else .{ 1, 1, 4, 0 });
    const host_payload = try arena_allocator.alloc(u8, @intCast(metadata.payload.byte_count));
    @memset(host_payload, 0x31);
    const first = try arena_allocator.alloc(u8, 8);
    @memset(first, 0x41);
    const second_len: usize = @intCast(metadata.payload.byte_count - @as(u64, @intCast(first.len)));
    const second = try arena_allocator.alloc(u8, second_len);
    @memset(second, 0x42);
    const segments = try arena_allocator.alloc([]const u8, 2);
    segments[0] = first;
    segments[1] = second;
    const image = switch (image_kind) {
        .host => testImage(metadata, .host_readable_now, host_payload),
        .segments => testSegmentedImage(metadata, segments),
        .device => blk: {
            metadata.payload.location_hint = device_location_hint;
            break :blk testImage(metadata, .device_download_required, null);
        },
    };
    const decision = try stage_transfer_mode.chooseStageTransferMode(.{
        .placement_plan = &placement,
        .metadata = metadata,
        .image = &image,
        .allow_borrow = allow_borrow,
        .local_device_peer_copy_available = peer_copy_available,
    });
    const envelope = try stage_transport.buildStageTransportActivationEnvelope(.{
        .metadata = metadata,
        .image = &image,
        .decision = decision,
    });
    return .{
        .placement = placement,
        .metadata = metadata,
        .image = image,
        .decision = decision,
        .envelope = envelope,
    };
}

fn testMetadata(
    placement: *const host_capability.PlacementPlan,
    entries: []const tensor_frame.TensorFrameBatchEntry,
    shape: [4]u64,
) !tensor_frame.TensorFrameMetadata {
    const boundary = tensor_frame.TensorFrameBoundaryRef{
        .boundary_index = placement.boundary_summaries[0].boundary_index,
        .source_stage_id = placement.boundary_summaries[0].source_stage_id,
        .target_stage_id = placement.boundary_summaries[0].target_stage_id,
        .producer_layer_start = placement.boundary_summaries[0].producer_layer_start,
        .producer_layer_end = placement.boundary_summaries[0].producer_layer_end,
        .consumer_layer_start = placement.boundary_summaries[0].consumer_layer_start,
        .consumer_layer_end = placement.boundary_summaries[0].consumer_layer_end,
    };
    const tensor = try tensor_frame.TensorFrameTensorDesc.contiguousActivation(.f32, shape);
    return .{
        .frame_id = try tensor_frame.TensorFrameInstanceId.init(55),
        .plan = .{
            .graph_digest = placement.graph_digest,
            .graph_contract_version = placement.graph_contract_version,
            .stage_plan_contract_version = placement.stage_plan_contract_version,
            .stage_plan_id = placement.stage_plan_id,
        },
        .boundary = boundary,
        .selected_contract = .{
            .boundary = boundary,
            .dtype = .f32,
            .layout = .row_major,
            .source = .explicit,
        },
        .role = .activation,
        .step_kind = .decode,
        .shape_context = .{
            .expected_hidden_size = shape[2],
            .expected_step_kind = .decode,
        },
        .tensor = tensor,
        .batch = .{ .entries = entries },
        .payload = .{
            .byte_count = tensor.payload_byte_count,
            .location_hint = .cpu,
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
}

fn testImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    readiness: boundary_byte_image.BoundaryByteImageReadiness,
    host_bytes: ?[]const u8,
) boundary_byte_image.BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_bytes = host_bytes,
        .location_hint = metadata.payload.location_hint,
        .readiness = readiness,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

fn testSegmentedImage(
    metadata: *const tensor_frame.TensorFrameMetadata,
    host_segments: []const []const u8,
) boundary_byte_image.BoundaryByteImageRef {
    return .{
        .metadata = metadata,
        .byte_count = metadata.payload.byte_count,
        .host_segments = host_segments,
        .location_hint = metadata.payload.location_hint,
        .readiness = .host_readable_now,
        .ownership = metadata.payload.ownership,
        .lifetime = metadata.payload.lifetime,
    };
}

fn buildTestPlacement(
    allocator: std.mem.Allocator,
    handoff_mode: host_capability.BoundaryHandoffMode,
) !host_capability.PlacementPlan {
    return buildTestPlacementWithBackends(allocator, handoff_mode, .cuda, .cuda);
}

fn buildTestPlacementWithBackends(
    allocator: std.mem.Allocator,
    handoff_mode: host_capability.BoundaryHandoffMode,
    source_backend_kind: host_capability.HostBackendKind,
    target_backend_kind: host_capability.HostBackendKind,
) !host_capability.PlacementPlan {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();

    const stages = try arena_allocator.alloc(host_capability.PlacementStageSummary, 2);
    stages[0] = testStageSummary(0, 0, 2);
    stages[1] = testStageSummary(1, 2, 4);

    const boundaries = try arena_allocator.alloc(host_capability.PlacementBoundarySummary, 1);
    boundaries[0] = .{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
    };

    const step_kinds = try arena_allocator.alloc(tensor_frame.TensorFrameStepKind, 1);
    step_kinds[0] = .decode;

    const bindings = try arena_allocator.alloc(host_capability.StageHostBinding, 2);
    bindings[0] = .{ .stage_id = 0, .host_id = .{ .value = 1 } };
    bindings[1] = .{ .stage_id = 1, .host_id = .{ .value = if (handoff_mode == .same_host_direct) 1 else 2 } };

    const host_count: usize = if (handoff_mode == .same_host_direct) 1 else 2;
    const hosts = try arena_allocator.alloc(host_capability.PlacementHostSummary, host_count);
    hosts[0] = .{
        .host_id = .{ .value = 1 },
        .backend_kind = source_backend_kind,
        .capability_id = .{ .digest = testDigest(0x10) },
        .residency_snapshot_id = .{ .digest = testDigest(0x20) },
    };
    if (host_count == 2) {
        hosts[1] = .{
            .host_id = .{ .value = 2 },
            .backend_kind = target_backend_kind,
            .capability_id = .{ .digest = testDigest(0x30) },
            .residency_snapshot_id = .{ .digest = testDigest(0x40) },
        };
    }

    const profiles = try arena_allocator.alloc(host_capability.BoundaryFrameProfile, 1);
    profiles[0] = .{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .step_kind = .decode,
        .dtype = .f32,
        .max_batch_entries = 4,
        .max_token_count_per_frame = 1,
        .max_activation_payload_bytes = 1024,
        .handoff_mode = handoff_mode,
    };

    var placement = host_capability.PlacementPlan{
        .arena = arena,
        .version = host_capability.placement_contract_version,
        .graph_digest = testDigest(0x50),
        .graph_contract_version = 1,
        .stage_plan_contract_version = 1,
        .stage_plan_id = .{ .digest = testDigest(0x60) },
        .plan_id = undefined,
        .stage_summaries = stages,
        .boundary_summaries = boundaries,
        .required_step_kinds = step_kinds,
        .state_placement_mode = .stateless_only,
        .stage_host_bindings = bindings,
        .host_summaries = hosts,
        .boundary_frame_profiles = profiles,
    };
    placement.plan_id = computeTestPlacementPlanId(&placement);
    try host_capability.validatePlacementPlan(&placement);
    return placement;
}

fn testStageSummary(stage_id: usize, layer_start: usize, layer_end: usize) host_capability.PlacementStageSummary {
    return .{
        .stage_id = stage_id,
        .layer_start = layer_start,
        .layer_end = layer_end,
        .owned_roles = emptyOwnedRoles(),
        .residency = .{
            .layer_start = layer_start,
            .layer_end = layer_end,
        },
    };
}

fn emptyOwnedRoles() @FieldType(host_capability.PlacementStageSummary, "owned_roles") {
    const OwnedRoles = @FieldType(host_capability.PlacementStageSummary, "owned_roles");
    return [_]bool{false} ** @typeInfo(OwnedRoles).array.len;
}

fn testDigest(seed: u8) [32]u8 {
    var digest: [32]u8 = undefined;
    for (&digest, 0..) |*byte, index| {
        byte.* = seed +% @as(u8, @intCast(index));
    }
    return digest;
}

fn computeTestPlacementPlanId(plan: *const host_capability.PlacementPlan) host_capability.PlacementPlanId {
    var encoder = TestHashEncoder.init();
    encoder.writeString("talu.placement_plan");
    encoder.writeU32(plan.version);
    encoder.writeBytes(&plan.graph_digest);
    encoder.writeU32(plan.graph_contract_version);
    encoder.writeU32(plan.stage_plan_contract_version);
    encoder.writeBytes(&plan.stage_plan_id.digest);
    encoder.writeUsize(plan.stage_summaries.len);
    for (plan.stage_summaries) |stage| writeTestPlacementStageSummary(&encoder, stage);
    encoder.writeUsize(plan.boundary_summaries.len);
    for (plan.boundary_summaries) |boundary| writeTestPlacementBoundarySummary(&encoder, boundary);
    encoder.writeUsize(plan.required_step_kinds.len);
    for (plan.required_step_kinds) |step| encoder.writeU8(@intFromEnum(step));
    encoder.writeU8(@intFromEnum(plan.state_placement_mode));
    encoder.writeOptionalU32(plan.state_ownership_contract_version);
    encoder.writeBool(plan.state_ownership_plan_id != null);
    if (plan.state_ownership_plan_id) |id| encoder.writeBytes(&id.digest);
    encoder.writeUsize(plan.state_stage_summaries.len);
    for (plan.state_stage_summaries) |summary| writeTestStageStatePlacementStageSummary(&encoder, summary);
    encoder.writeUsize(plan.stage_host_bindings.len);
    for (plan.stage_host_bindings) |binding| writeTestStageHostBinding(&encoder, binding);
    encoder.writeUsize(plan.host_summaries.len);
    for (plan.host_summaries) |host| writeTestPlacementHostSummary(&encoder, host);
    encoder.writeUsize(plan.boundary_frame_profiles.len);
    for (plan.boundary_frame_profiles) |profile| writeTestBoundaryFrameProfile(&encoder, profile);
    return .{ .digest = encoder.finish() };
}

fn writeTestPlacementStageSummary(encoder: *TestHashEncoder, stage: host_capability.PlacementStageSummary) void {
    encoder.writeUsize(stage.stage_id);
    encoder.writeUsize(stage.layer_start);
    encoder.writeUsize(stage.layer_end);
    for (stage.owned_roles) |owned| encoder.writeBool(owned);
    writeTestStageResidencyReport(encoder, stage.residency);
}

fn writeTestPlacementBoundarySummary(encoder: *TestHashEncoder, boundary: host_capability.PlacementBoundarySummary) void {
    encoder.writeUsize(boundary.boundary_index);
    encoder.writeUsize(boundary.source_stage_id);
    encoder.writeUsize(boundary.target_stage_id);
    encoder.writeUsize(boundary.producer_layer_start);
    encoder.writeUsize(boundary.producer_layer_end);
    encoder.writeUsize(boundary.consumer_layer_start);
    encoder.writeUsize(boundary.consumer_layer_end);
}

fn writeTestStageStatePlacementStageSummary(encoder: *TestHashEncoder, summary: host_capability.StageStatePlacementStageSummary) void {
    encoder.writeUsize(summary.stage_id);
    encoder.writeUsize(summary.descriptor_count);
    encoder.writeBool(summary.owns_runtime_state);
    encoder.writeUsize(summary.descriptors.len);
    for (summary.descriptors) |descriptor| {
        encoder.writeU8(descriptor.descriptor_id);
        encoder.writeU64(descriptor.size_bytes);
        encoder.writeU16(descriptor.align_bytes);
        encoder.writeBool(descriptor.zero_init);
        encoder.writeU8(@intFromEnum(descriptor.lifecycle));
        encoder.writeU8(descriptor.runtime_kind);
    }
}

fn writeTestStageHostBinding(encoder: *TestHashEncoder, binding: host_capability.StageHostBinding) void {
    encoder.writeUsize(binding.stage_id);
    encoder.writeU64(binding.host_id.value);
    encoder.writeBool(binding.expected_capability_id != null);
    if (binding.expected_capability_id) |id| encoder.writeBytes(&id.digest);
    encoder.writeBool(binding.expected_residency_snapshot_id != null);
    if (binding.expected_residency_snapshot_id) |id| encoder.writeBytes(&id.digest);
}

fn writeTestPlacementHostSummary(encoder: *TestHashEncoder, summary: host_capability.PlacementHostSummary) void {
    encoder.writeU64(summary.host_id.value);
    encoder.writeU8(@intFromEnum(summary.backend_kind));
    encoder.writeBytes(&summary.capability_id.digest);
    encoder.writeBytes(&summary.residency_snapshot_id.digest);
}

fn writeTestBoundaryFrameProfile(encoder: *TestHashEncoder, profile: host_capability.BoundaryFrameProfile) void {
    encoder.writeUsize(profile.boundary_index);
    encoder.writeUsize(profile.source_stage_id);
    encoder.writeUsize(profile.target_stage_id);
    encoder.writeU32(profile.tensor_frame_contract_version);
    encoder.writeU8(@intFromEnum(profile.step_kind));
    encoder.writeU8(@intFromEnum(profile.dtype));
    encoder.writeU8(@intFromEnum(profile.layout));
    encoder.writeU64(profile.max_batch_entries);
    encoder.writeU64(profile.max_token_count_per_frame);
    encoder.writeU64(profile.max_activation_payload_bytes);
    encoder.writeU8(@intFromEnum(profile.handoff_mode));
}

fn writeTestStageResidencyReport(
    encoder: *TestHashEncoder,
    residency: @FieldType(host_capability.PlacementStageSummary, "residency"),
) void {
    encoder.writeUsize(residency.layer_start);
    encoder.writeUsize(residency.layer_end);
    encoder.writeUsize(residency.total_checkpoint_bytes);
    for (residency.role_bytes) |bytes| encoder.writeUsize(bytes);
}

const TestHashEncoder = struct {
    hasher: std.crypto.hash.sha2.Sha256,

    fn init() TestHashEncoder {
        return .{ .hasher = std.crypto.hash.sha2.Sha256.init(.{}) };
    }

    fn finish(self: *TestHashEncoder) [32]u8 {
        var digest: [32]u8 = undefined;
        self.hasher.final(&digest);
        return digest;
    }

    fn writeBytes(self: *TestHashEncoder, bytes: []const u8) void {
        self.hasher.update(bytes);
    }

    fn writeString(self: *TestHashEncoder, value: []const u8) void {
        self.writeU64(value.len);
        self.writeBytes(value);
    }

    fn writeBool(self: *TestHashEncoder, value: bool) void {
        self.writeU8(@intFromBool(value));
    }

    fn writeU8(self: *TestHashEncoder, value: u8) void {
        self.writeBytes(&.{value});
    }

    fn writeU16(self: *TestHashEncoder, value: u16) void {
        var buf: [2]u8 = undefined;
        std.mem.writeInt(u16, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeU32(self: *TestHashEncoder, value: u32) void {
        var buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeU64(self: *TestHashEncoder, value: u64) void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf, value, .little);
        self.writeBytes(&buf);
    }

    fn writeUsize(self: *TestHashEncoder, value: usize) void {
        self.writeU64(@intCast(value));
    }

    fn writeOptionalU32(self: *TestHashEncoder, value: ?u32) void {
        self.writeBool(value != null);
        if (value) |payload| self.writeU32(payload);
    }
};

test "inference bridge local_stage_transport executeLocalStageTransport borrows host bytes with target support" {
    var bundle = try buildTestBundle(std.testing.allocator, .same_host_direct, .host, true, false);
    defer bundle.deinit();
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.borrow_in_process, bundle.decision.mode);
    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace };

    try executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, bundle.request());

    try trace.expect(&.{ .source_synchronize, .target_borrow });
    try std.testing.expectEqual(@as(usize, 16), target.borrowed_byte_count);
    try std.testing.expectEqual(@as(u8, 0x31), target.upload_first_byte);
}

test "inference bridge local_stage_transport endpoint registry waits on matching source receipt" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .device, false, false);
    defer bundle.deinit();
    var staging: [16]u8 align(64) = undefined;
    var request = bundle.request();
    request.staging = staging[0..];
    request.source_receipt = StageExecutionReceipt.completed(bundle.metadata.boundary.source_stage_id);

    var trace = Trace{};
    var source_stage = TestSourceStage{ .trace = &trace };
    var target_stage = TestTargetStage{ .trace = &trace };
    var endpoints = [_]LocalStageTransportEndpoint{
        localStageTransportAdapter(TestSourceStage, bundle.metadata.boundary.source_stage_id, &source_stage),
        localStageTransportAdapter(TestTargetStage, bundle.metadata.boundary.target_stage_id, &target_stage),
    };
    var registry = LocalStageTransportEndpointRegistry{ .endpoints = endpoints[0..] };
    const source = try registry.endpointForStageId(bundle.metadata.boundary.source_stage_id);
    const target = try registry.endpointForStageId(bundle.metadata.boundary.target_stage_id);

    try executeLocalStageTransport(LocalStageTransportEndpoint, LocalStageTransportEndpoint, source, target, request);
    try trace.expect(&.{ .source_synchronize, .source_download, .target_upload });

    trace = .{};
    request.source_receipt = StageExecutionReceipt.completed(bundle.metadata.boundary.target_stage_id);
    try std.testing.expectError(
        error.StageTransferBoundaryMismatch,
        executeLocalStageTransport(LocalStageTransportEndpoint, LocalStageTransportEndpoint, source, target, request),
    );
    try std.testing.expectEqual(@as(usize, 0), trace.count);
}

test "inference bridge local_stage_transport executeLocalStageTransportWithFailureCapture reports target mutation failure and preserves source error" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .host, true, false);
    defer bundle.deinit();
    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace, .fail_upload = error.AccessDenied };
    var capture = LocalStageTransportFailureCapture.init(std.testing.allocator);
    defer capture.deinit();

    try std.testing.expectError(
        error.AccessDenied,
        executeLocalStageTransportWithFailureCapture(
            TestSourceStage,
            TestTargetStage,
            &source,
            &target,
            bundle.request(),
            &capture,
        ),
    );

    const report = capture.report orelse return error.MissingFailureReport;
    try std.testing.expectEqual(error.AccessDenied, report.source_error);
    try std.testing.expectEqual(@as(usize, 1), report.entries.len);
    const entry = report.entries[0];
    try std.testing.expectEqual(staged_error.StagedFailureKind.transfer_failed, entry.primary_failure.kind);
    try std.testing.expectEqual(staged_error.StagedFailurePhase.frame_handoff, entry.primary_failure.phase);
    try std.testing.expectEqual(staged_error.StagedFailureScope.transport, entry.primary_failure.scope);
    try std.testing.expectEqual(staged_error.StagedSourceDomain.transport, entry.primary_failure.source.domain);
    try std.testing.expectEqualStrings("AccessDenied", entry.primary_failure.source.source_error_name.?);
    try std.testing.expectEqual(@as(usize, 2), entry.touched_stages.len);
    try std.testing.expectEqual(@as(usize, 0), entry.touched_stages[0].stage_id);
    try std.testing.expectEqual(@as(usize, 1), entry.touched_stages[1].stage_id);
    try std.testing.expect(entry.cleanup_plan != null);
    try std.testing.expect(entry.cleanup_report != null);
    try std.testing.expectEqual(staged_error.StagedFailureKind.transfer_failed, entry.error_report.primary_failure.kind);
    try trace.expect(&.{ .source_synchronize, .target_upload });
}

test "inference bridge local_stage_transport executeLocalStageTransportWithFailureCapture records batched source cancellation before target mutation" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .segments, true, false);
    defer bundle.deinit();
    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace, .fail_synchronize = error.RequestCancelled };
    var target = TestTargetStage{ .trace = &trace };
    var capture = LocalStageTransportFailureCapture.init(std.testing.allocator);
    defer capture.deinit();

    try std.testing.expectError(
        error.RequestCancelled,
        executeLocalStageTransportWithFailureCapture(
            TestSourceStage,
            TestTargetStage,
            &source,
            &target,
            bundle.request(),
            &capture,
        ),
    );

    const report = capture.report orelse return error.MissingFailureReport;
    try std.testing.expectEqual(error.RequestCancelled, report.source_error);
    try std.testing.expectEqual(@as(usize, 2), report.entries.len);
    for (report.entries, test_decode_entries_two) |entry, expected_batch_entry| {
        try std.testing.expectEqual(staged_error.StagedFailureKind.request_cancelled, entry.primary_failure.kind);
        try std.testing.expectEqual(staged_error.StagedFailureScope.request, entry.primary_failure.scope);
        try std.testing.expectEqual(expected_batch_entry.request_id, entry.primary_failure.context.request_id.?);
        try std.testing.expectEqual(expected_batch_entry.slot_id, entry.primary_failure.context.slot_id.?);
        try std.testing.expectEqual(@as(usize, 1), entry.touched_stages.len);
        try std.testing.expectEqual(@as(usize, 0), entry.touched_stages[0].stage_id);
        try std.testing.expect(entry.cleanup_plan != null);
        try std.testing.expect(entry.cleanup_report != null);
    }
    try trace.expect(&.{.source_synchronize});
}

test "inference bridge local_stage_transport executeLocalStageTransport copies host bytes and preserves source sync errors" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .host, true, false);
    defer bundle.deinit();
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.copy_in_process, bundle.decision.mode);
    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace };

    try executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, bundle.request());

    try trace.expect(&.{ .source_synchronize, .target_upload });
    try std.testing.expectEqual(@as(usize, 16), target.upload_byte_count);
    try std.testing.expectEqual(@as(u8, 0x31), target.upload_first_byte);

    trace = .{};
    source = .{ .trace = &trace, .fail_synchronize = error.InjectedSynchronizeFailure };
    target = .{ .trace = &trace };
    try std.testing.expectError(error.InjectedSynchronizeFailure, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, bundle.request()));
    try trace.expect(&.{.source_synchronize});
}

test "inference bridge local_stage_transport executeLocalStageTransport prepares source before target mutation" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .host, true, false);
    defer bundle.deinit();
    var trace = Trace{};
    var source = TestPreparingSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace };

    try executeLocalStageTransport(TestPreparingSourceStage, TestTargetStage, &source, &target, bundle.request());

    try trace.expect(&.{ .source_synchronize, .source_prepare, .target_upload });
    try std.testing.expectEqual(test_decode_entry[0].slot_id, source.prepared_slot_id);
    try std.testing.expectEqual(@as(usize, 16), target.upload_byte_count);
}

test "inference bridge local_stage_transport executeLocalStageTransportWithFailureCapture preserves source prepare errors" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .host, true, false);
    defer bundle.deinit();
    var trace = Trace{};
    var source = TestPreparingSourceStage{ .trace = &trace, .fail_prepare = error.InjectedPrepareFailure };
    var target = TestTargetStage{ .trace = &trace };
    var capture = LocalStageTransportFailureCapture.init(std.testing.allocator);
    defer capture.deinit();

    try std.testing.expectError(
        error.InjectedPrepareFailure,
        executeLocalStageTransportWithFailureCapture(
            TestPreparingSourceStage,
            TestTargetStage,
            &source,
            &target,
            bundle.request(),
            &capture,
        ),
    );

    const report = capture.report orelse return error.MissingFailureReport;
    try std.testing.expectEqual(error.InjectedPrepareFailure, report.source_error);
    try std.testing.expectEqual(@as(usize, 1), report.entries.len);
    try std.testing.expectEqual(staged_error.StagedFailureKind.transfer_failed, report.entries[0].primary_failure.kind);
    try std.testing.expectEqual(staged_error.StagedFailurePhase.frame_handoff, report.entries[0].primary_failure.phase);
    try std.testing.expectEqual(@as(usize, 2), report.entries[0].touched_stages.len);
    try std.testing.expectEqual(@as(usize, 0), target.upload_byte_count);
    try trace.expect(&.{ .source_synchronize, .source_prepare });
}

test "inference bridge local_stage_transport executeLocalStageTransport uploads host segments only with target support" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .segments, true, false);
    defer bundle.deinit();
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.copy_in_process, bundle.decision.mode);
    try std.testing.expectEqual(stage_transport.StageTransportActivationScope.multi_entry_local, bundle.envelope.activation_scope.?);

    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace };
    try executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, bundle.request());
    try trace.expect(&.{ .source_synchronize, .target_upload_segments });
    try std.testing.expectEqual(@as(usize, 32), target.segment_byte_count);
    try std.testing.expectEqual(@as(u8, 0x41), target.upload_first_byte);

    trace = .{};
    source = .{ .trace = &trace };
    var no_segments_target = TestTargetStageNoSegments{ .trace = &trace };
    try std.testing.expectError(error.LocalStageTransportSegmentedUploadUnsupported, executeLocalStageTransport(TestSourceStage, TestTargetStageNoSegments, &source, &no_segments_target, bundle.request()));
    try trace.expect(&.{});
}

test "inference bridge local_stage_transport executeLocalStageTransport downloads device bytes through caller staging" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .device, true, false);
    defer bundle.deinit();
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.device_download_then_copy, bundle.decision.mode);
    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace };
    var staging_storage: [64]u8 align(64) = [_]u8{0} ** 64;
    var request = bundle.request();
    request.staging = staging_storage[0..];

    try executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request);

    try trace.expect(&.{ .source_synchronize, .source_download, .target_upload });
    try std.testing.expectEqual(@as(usize, 16), target.upload_byte_count);
    try std.testing.expectEqual(@as(u8, 0x5a), target.upload_first_byte);

    trace = .{};
    source = .{ .trace = &trace };
    target = .{ .trace = &trace };
    request.staging = null;
    try std.testing.expectError(error.LocalStageTransportMissingStaging, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{});

    var small_staging: [8]u8 align(64) = [_]u8{0} ** 8;
    request.staging = small_staging[0..];
    try std.testing.expectError(error.LocalStageTransportBufferTooSmall, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{});

    trace = .{};
    source = .{ .trace = &trace, .fail_download = error.InjectedDownloadFailure };
    target = .{ .trace = &trace };
    request.staging = staging_storage[0..];
    try std.testing.expectError(error.InjectedDownloadFailure, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{ .source_synchronize, .source_download });
}

test "inference bridge local_stage_transport executeLocalStageTransport executes peer copy without host staging" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .device, true, true);
    defer bundle.deinit();
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.device_peer_copy_in_process, bundle.decision.mode);
    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace };
    var staging_storage: [64]u8 align(64) = [_]u8{0xee} ** 64;
    var request = bundle.request();
    request.staging = staging_storage[0..];
    request.local_device_peer_copy_available = true;

    try executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request);
    try trace.expect(&.{ .source_synchronize, .source_peer_copy });
    try std.testing.expectEqual(@as(usize, 16), source.peer_copy_byte_count);
    for (staging_storage) |byte| {
        try std.testing.expectEqual(@as(u8, 0xee), byte);
    }

    trace = .{};
    source = .{ .trace = &trace, .peer_handles_sync = true };
    target = .{ .trace = &trace };
    try executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request);
    try trace.expect(&.{.source_peer_copy});

    trace = .{};
    source = .{ .trace = &trace, .fail_synchronize = error.InjectedSynchronizeFailure };
    target = .{ .trace = &trace };
    try std.testing.expectError(error.InjectedSynchronizeFailure, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{.source_synchronize});

    trace = .{};
    var no_peer_source = TestSourceStageNoPeer{ .trace = &trace };
    target = .{ .trace = &trace };
    try std.testing.expectError(error.LocalStageTransportPeerCopyUnsupported, executeLocalStageTransport(TestSourceStageNoPeer, TestTargetStage, &no_peer_source, &target, request));
    try trace.expect(&.{});
}

fn expectTransportMatrixCase(
    source_backend_kind: host_capability.HostBackendKind,
    target_backend_kind: host_capability.HostBackendKind,
    image_kind: TestImageKind,
    location_hint: tensor_frame.TensorFramePayloadLocationHint,
    peer_copy_available: bool,
    allow_borrow: bool,
    handoff_mode: host_capability.BoundaryHandoffMode,
    expected_mode: stage_transfer_mode.StageTransferMode,
    expected_trace: []const TraceEvent,
) !void {
    var bundle = try buildTestBundleWithBackends(
        std.testing.allocator,
        handoff_mode,
        image_kind,
        allow_borrow,
        peer_copy_available,
        source_backend_kind,
        target_backend_kind,
        location_hint,
    );
    defer bundle.deinit();
    try std.testing.expectEqual(expected_mode, bundle.decision.mode);

    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace };
    var staging_storage: [64]u8 align(64) = [_]u8{0} ** 64;
    var request = bundle.request();
    request.local_device_peer_copy_available = peer_copy_available;
    if (expected_mode == .device_download_then_copy) {
        request.staging = staging_storage[0..];
    }

    try executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request);
    try trace.expect(expected_trace);
}

test "inference bridge local_stage_transport executeLocalStageTransport covers local backend transfer matrix" {
    try expectTransportMatrixCase(.cpu, .cpu, .host, .cpu, false, true, .same_host_direct, .borrow_in_process, &.{ .source_synchronize, .target_borrow });
    try expectTransportMatrixCase(.cpu, .cuda, .host, .cpu, false, false, .local_in_process, .copy_in_process, &.{ .source_synchronize, .target_upload });
    try expectTransportMatrixCase(.cpu, .metal, .host, .cpu, false, false, .local_in_process, .copy_in_process, &.{ .source_synchronize, .target_upload });
    try expectTransportMatrixCase(.cuda, .cpu, .device, .{ .cuda = 0 }, false, false, .local_in_process, .device_download_then_copy, &.{ .source_synchronize, .source_download, .target_upload });
    try expectTransportMatrixCase(.metal, .cpu, .device, .{ .metal = 0 }, false, false, .local_in_process, .device_download_then_copy, &.{ .source_synchronize, .source_download, .target_upload });
    try expectTransportMatrixCase(.cuda, .cuda, .device, .{ .cuda = 0 }, true, false, .local_in_process, .device_peer_copy_in_process, &.{ .source_synchronize, .source_peer_copy });
    try expectTransportMatrixCase(.cuda, .cuda, .device, .{ .cuda = 0 }, false, false, .local_in_process, .device_download_then_copy, &.{ .source_synchronize, .source_download, .target_upload });
    try expectTransportMatrixCase(.cuda, .metal, .device, .{ .cuda = 0 }, true, false, .local_in_process, .device_download_then_copy, &.{ .source_synchronize, .source_download, .target_upload });
    try expectTransportMatrixCase(.metal, .cuda, .device, .{ .metal = 0 }, true, false, .local_in_process, .device_download_then_copy, &.{ .source_synchronize, .source_download, .target_upload });
    try expectTransportMatrixCase(.metal, .metal, .device, .{ .metal = 0 }, true, false, .local_in_process, .device_peer_copy_in_process, &.{ .source_synchronize, .source_peer_copy });
}

test "inference bridge local_stage_transport executeLocalStageTransport rejects remote modes before synchronization" {
    var host_remote = try buildTestBundle(std.testing.allocator, .remote_declared, .host, true, false);
    defer host_remote.deinit();
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.remote_stream, host_remote.decision.mode);
    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace };
    var request = host_remote.request();

    try std.testing.expectError(error.LocalStageTransportRemoteModeUnsupported, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{});

    var device_remote = try buildTestBundle(std.testing.allocator, .remote_declared, .device, true, false);
    defer device_remote.deinit();
    try std.testing.expectEqual(stage_transfer_mode.StageTransferMode.device_download_then_remote_stream, device_remote.decision.mode);
    trace = .{};
    source = .{ .trace = &trace };
    target = .{ .trace = &trace };
    request = device_remote.request();
    try std.testing.expectError(error.LocalStageTransportRemoteModeUnsupported, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{});
}

test "inference bridge local_stage_transport executeLocalStageTransport validates envelope and recomputed decision before mutation" {
    var bundle = try buildTestBundle(std.testing.allocator, .local_in_process, .host, true, false);
    defer bundle.deinit();
    var larger_payload = try buildTestBundle(std.testing.allocator, .local_in_process, .segments, true, false);
    defer larger_payload.deinit();
    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStage{ .trace = &trace };

    var request = bundle.request();
    var other_metadata = bundle.metadata.*;
    var wrong_metadata_image = bundle.image;
    wrong_metadata_image.metadata = &other_metadata;
    request.image = &wrong_metadata_image;
    try std.testing.expectError(error.LocalStageTransportMetadataMismatch, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{});

    request = bundle.request();
    request.envelope = &larger_payload.envelope;
    try std.testing.expectError(error.LocalStageTransportPayloadByteCountMismatch, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{});

    request = bundle.request();
    var wrong_kind = try stage_transport.buildStageTransportFailureEnvelope(.{
        .kind = .transfer_failed,
        .phase = .frame_handoff,
        .scope = .transport,
        .context = .{ .boundary_index = 0 },
        .source = .{ .domain = .transport },
    });
    request.envelope = &wrong_kind;
    try std.testing.expectError(error.LocalStageTransportEnvelopeMismatch, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{});

    var wrong_mode = bundle.envelope;
    wrong_mode.transfer_mode = .borrow_in_process;
    request = bundle.request();
    request.envelope = &wrong_mode;
    try std.testing.expectError(error.LocalStageTransportDecisionMismatch, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{});

    var wrong_decision = bundle.decision;
    wrong_decision.boundary_profile.max_activation_payload_bytes += 1;
    request = bundle.request();
    request.decision = wrong_decision;
    request.envelope = &bundle.envelope;
    try std.testing.expectError(error.LocalStageTransportDecisionMismatch, executeLocalStageTransport(TestSourceStage, TestTargetStage, &source, &target, request));
    try trace.expect(&.{});
}

test "inference bridge local_stage_transport executeLocalStageTransport rejects missing optional borrow target before synchronization" {
    var bundle = try buildTestBundle(std.testing.allocator, .same_host_direct, .host, true, false);
    defer bundle.deinit();
    var trace = Trace{};
    var source = TestSourceStage{ .trace = &trace };
    var target = TestTargetStageNoBorrow{ .trace = &trace };

    try std.testing.expectError(error.LocalStageTransportBorrowUnsupported, executeLocalStageTransport(TestSourceStage, TestTargetStageNoBorrow, &source, &target, bundle.request()));
    try trace.expect(&.{});
}
