//! Tests for backend interface contracts used by pipeline orchestration.

const std = @import("std");
const main = @import("main");

fn expectStageExecutorSurface(comptime executor: type) !void {
    inline for (.{
        "backendKind",
        "maxBatchSize",
        "prefillChunkRowsCap",
        "supportedBoundaryDTypes",
        "executeDecodeLayerRange",
        "executePrefillLayerRange",
    }) |decl_name| {
        try std.testing.expect(@hasDecl(executor, decl_name));
    }
    try std.testing.expect(@hasDecl(executor, "supports_local_stage_execution"));
}

fn expectNoStageOrchestrationSurface(comptime executor: type) !void {
    inline for (.{
        "buildStagePlan",
        "createLocalPipelineRuntime",
        "executeLocalStageTransport",
        "executeStageChain",
        "localPipelineRuntime",
        "nextStage",
        "previousStage",
        "remoteStage",
        "transferPipelineActivation",
    }) |decl_name| {
        try std.testing.expect(!@hasDecl(executor, decl_name));
    }
}

fn expectExternalActivationSurface(comptime endpoint: type) !void {
    inline for (.{
        "deviceLocationHint",
        "decodeExternalOutput",
        "prefillExternalOutput",
        "decodeExternalInput",
        "prefillExternalInput",
        "sideExternalInput",
    }) |decl_name| {
        try std.testing.expect(@hasDecl(endpoint, decl_name));
    }
    try std.testing.expect(@hasDecl(endpoint, "supports_transport_endpoint_descriptors"));
}

fn expectNoTransportOrchestrationSurface(comptime endpoint: type) !void {
    inline for (.{
        "buildActivationTransportContract",
        "downloadPipelineActivation",
        "executeLocalStageTransport",
        "handoffActivation",
        "nextStage",
        "previousStage",
        "remoteEndpoint",
        "selectTransferRoute",
        "transferPipelineActivation",
        "uploadPipelineActivation",
    }) |decl_name| {
        try std.testing.expect(!@hasDecl(endpoint, decl_name));
    }
}

test "backend interfaces expose only stage-local execution and external activation surfaces" {
    inline for (.{
        main.inference.backend.cpu.interface,
        main.inference.backend.cuda.interface,
        main.inference.backend.metal.interface,
    }) |interface| {
        try expectStageExecutorSurface(interface.stage_executor);
        try expectNoStageOrchestrationSurface(interface.stage_executor);
        try expectExternalActivationSurface(interface.transport_endpoint);
        try expectNoTransportOrchestrationSurface(interface.transport_endpoint);
    }
}
