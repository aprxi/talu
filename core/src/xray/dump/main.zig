//! talu-dump - Tensor dump CLI for debugging model integrations
//!
//! Runs a single forward pass and captures all tensor values at trace points
//! to an NPZ file for offline comparison against PyTorch reference.
//!
//! Usage:
//!   ./talu-dump --model path/to/model --prompt "Hello" -o /tmp/talu.npz
//!
//! This is dev-only tooling, NOT a product feature.

const std = @import("std");

// Import from lib module (added as dependency in build.zig)
const lib = @import("lib");
const io = lib.io;
const tokenizer_mod = lib.tokenizer;
const inference = lib.inference;
const FusedCpuBackend = inference.backend.FusedCpuBackend;

// Dump modules (accessed through lib since that's how the module is set up)
const capture_mod = lib.dump.capture;
const npz_mod = lib.dump.npz;

const Capture = capture_mod.Capture;
const NpzWriter = npz_mod.NpzWriter;
const progress_mod = lib.capi.progress;

const Args = struct {
    model_path: []const u8,
    prompt: []const u8,
    output_path: []const u8,
    max_tokens: usize,
    // Filters
    layer_filter: ?u16, // --layer <N>
    layer_range_end: ?u16, // --layer <A:B>
    point_filters: []const []const u8, // --point <name> (repeatable)
    stop_after_layer: ?u16, // --stop-after-layer <N>
};

fn parseArgs(allocator: std.mem.Allocator) !Args {
    var args = std.process.args();
    _ = args.skip(); // Skip program name

    var model_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var output_path: []const u8 = "/tmp/talu.npz";
    var max_tokens: usize = 1;

    // Filter args
    var layer_filter: ?u16 = null;
    var layer_range_end: ?u16 = null;
    var point_filters_list: std.ArrayListUnmanaged([]const u8) = .{};
    var stop_after_layer: ?u16 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            model_path = args.next();
        } else if (std.mem.eql(u8, arg, "--prompt") or std.mem.eql(u8, arg, "-p")) {
            prompt = args.next();
        } else if (std.mem.eql(u8, arg, "--output") or std.mem.eql(u8, arg, "-o")) {
            if (args.next()) |path| {
                output_path = path;
            }
        } else if (std.mem.eql(u8, arg, "--tokens") or std.mem.eql(u8, arg, "-n")) {
            if (args.next()) |tok_str| {
                max_tokens = std.fmt.parseInt(usize, tok_str, 10) catch 1;
            }
        } else if (std.mem.eql(u8, arg, "--layer") or std.mem.eql(u8, arg, "-l")) {
            // Parse --layer <N> or --layer <A:B>
            if (args.next()) |layer_str| {
                if (std.mem.indexOf(u8, layer_str, ":")) |colon_pos| {
                    // Range: A:B
                    layer_filter = std.fmt.parseInt(u16, layer_str[0..colon_pos], 10) catch null;
                    layer_range_end = std.fmt.parseInt(u16, layer_str[colon_pos + 1 ..], 10) catch null;
                } else {
                    // Single layer
                    layer_filter = std.fmt.parseInt(u16, layer_str, 10) catch null;
                }
            }
        } else if (std.mem.eql(u8, arg, "--point")) {
            // --point <name> (repeatable)
            if (args.next()) |point_name| {
                try point_filters_list.append(allocator, point_name);
            }
        } else if (std.mem.eql(u8, arg, "--stop-after-layer") or std.mem.eql(u8, arg, "-s")) {
            if (args.next()) |layer_str| {
                stop_after_layer = std.fmt.parseInt(u16, layer_str, 10) catch null;
            }
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        }
    }

    if (model_path == null) {
        std.debug.print("Error: --model is required\n\n", .{});
        printHelp();
        std.process.exit(1);
    }

    if (prompt == null) {
        prompt = "Hello";
    }

    return .{
        .model_path = model_path.?,
        .prompt = prompt.?,
        .output_path = output_path,
        .max_tokens = max_tokens,
        .layer_filter = layer_filter,
        .layer_range_end = layer_range_end,
        .point_filters = try point_filters_list.toOwnedSlice(allocator),
        .stop_after_layer = stop_after_layer,
    };
}

fn printHelp() void {
    std.debug.print(
        \\talu-dump - Tensor dump for model debugging
        \\
        \\Captures tensor values during inference to NPZ for comparison
        \\against PyTorch reference.
        \\
        \\USAGE:
        \\    talu-dump --model <path> [options]
        \\
        \\OPTIONS:
        \\    -m, --model <path>         Model path (required)
        \\    -p, --prompt <text>        Input prompt (default: "Hello")
        \\    -o, --output <path>        Output NPZ path (default: /tmp/talu.npz)
        \\    -n, --tokens <n>           Max tokens to generate (default: 1)
        \\    -h, --help                 Show this help
        \\
        \\FILTERING (for faster debugging):
        \\    -l, --layer <N>            Capture only layer N
        \\    -l, --layer <A:B>          Capture layers A through B (inclusive)
        \\        --point <name>         Capture only points containing <name> (repeatable)
        \\    -s, --stop-after-layer <N> Stop execution after layer N (big runtime win)
        \\
        \\EXAMPLES:
        \\    # Full dump
        \\    talu-dump -m models/Qwen/Qwen3-0.6B-GAF4 -p "Hello" -o /tmp/qwen.npz
        \\
        \\    # Only layer 5
        \\    talu-dump -m model -l 5 -o /tmp/layer5.npz
        \\
        \\    # Layers 0-3 with stop (fast!)
        \\    talu-dump -m model -l 0:3 -s 3 -o /tmp/first4.npz
        \\
        \\    # Only FFN norms
        \\    talu-dump -m model --point ffn_norm -o /tmp/ffn_norms.npz
        \\
        \\Then compare against PyTorch reference:
        \\    cd tools/archs
        \\    uv run python -m compare _reference/qwen3.npz /tmp/qwen.npz
        \\
    , .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Check that dump_tensors is enabled (compile-time flag from build_options)
    if (!capture_mod.DumpEnabled) {
        std.debug.print("Error: This binary was not built with tensor dump support.\n", .{});
        std.debug.print("Build with: zig build dump -Drelease\n", .{});
        std.process.exit(1);
    }

    const args = try parseArgs(allocator);

    std.debug.print("talu-dump: Tensor capture for model debugging\n", .{});
    std.debug.print("Model: {s}\n", .{args.model_path});
    std.debug.print("Prompt: {s}\n", .{args.prompt});
    std.debug.print("Output: {s}\n", .{args.output_path});

    // Print filter settings
    if (args.layer_filter != null or args.point_filters.len > 0 or args.stop_after_layer != null) {
        std.debug.print("Filters:", .{});
        if (args.layer_filter) |start| {
            if (args.layer_range_end) |end| {
                std.debug.print(" layers={d}:{d}", .{ start, end });
            } else {
                std.debug.print(" layer={d}", .{start});
            }
        }
        if (args.point_filters.len > 0) {
            std.debug.print(" points=[", .{});
            for (args.point_filters, 0..) |p, i| {
                if (i > 0) std.debug.print(",", .{});
                std.debug.print("{s}", .{p});
            }
            std.debug.print("]", .{});
        }
        if (args.stop_after_layer) |s| {
            std.debug.print(" stop_after={d}", .{s});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("\n", .{});

    // Initialize capture
    var cap = Capture.init(allocator);
    defer cap.deinit();

    // Apply filters
    if (args.layer_range_end) |end| {
        cap.setLayerRange(args.layer_filter.?, end);
    } else if (args.layer_filter) |layer| {
        cap.setLayerFilter(layer);
    }
    if (args.point_filters.len > 0) {
        cap.setPointFilters(args.point_filters);
    }
    if (args.stop_after_layer) |layer| {
        cap.setStopAfterLayer(layer);
    }

    // Set global capture for kernel instrumentation
    capture_mod.setGlobalCapture(&cap);
    defer capture_mod.clearGlobalCapture();

    // Load model
    std.debug.print("Loading model...\n", .{});

    // Resolve model bundle (handles HuggingFace paths, local dirs, etc.)
    var model_bundle = try io.repository.resolve(allocator, args.model_path, .{});
    defer model_bundle.deinit();

    var loaded = try io.loadModel(allocator, model_bundle.config_path(), model_bundle.weights_path() orelse return error.WeightsNotFound, progress_mod.ProgressContext.NONE);
    defer loaded.deinit();

    // Initialize tokenizer
    var tok = try tokenizer_mod.Tokenizer.initFromPath(allocator, model_bundle.tokenizer_path());
    defer tok.deinit();

    // Initialize backend (CPU only for dump, batch size 1)
    var backend = try FusedCpuBackend.init(allocator, &loaded, 1, progress_mod.ProgressContext.NONE);
    defer backend.deinit();

    // Tokenize prompt
    const tokens = try tok.encode(args.prompt);
    defer allocator.free(tokens);

    std.debug.print("Input tokens: {d}\n", .{tokens.len});
    std.debug.print("Running prefill...\n\n", .{});

    // Allocate logits buffer
    const vocab_size = backend.vocab_size;
    const logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits);

    // Enable capture
    cap.enable();

    // Run prefill (single forward pass through all layers)
    try backend.prefill(tokens, logits);

    cap.disable();

    // Find top token for display
    var max_idx: usize = 0;
    var max_val: f32 = logits[0];
    for (logits, 0..) |v, i| {
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    std.debug.print("Top token: {d} (logit: {d:.4})\n", .{ max_idx, max_val });
    std.debug.print("Captured {d} tensors\n", .{cap.tensors.items.len});

    // List captured tensors
    std.debug.print("\nCaptured trace points:\n", .{});
    for (cap.tensors.items) |tensor| {
        var shape_buf: [64]u8 = undefined;
        var pos: usize = 0;
        shape_buf[pos] = '(';
        pos += 1;
        for (0..tensor.ndim) |i| {
            const written = std.fmt.bufPrint(shape_buf[pos..], "{d}", .{tensor.shape[i]}) catch break;
            pos += written.len;
            if (i < tensor.ndim - 1) {
                shape_buf[pos] = ',';
                pos += 1;
                shape_buf[pos] = ' ';
                pos += 1;
            }
        }
        shape_buf[pos] = ')';
        pos += 1;
        std.debug.print("  {s}: {s}\n", .{ tensor.name, shape_buf[0..pos] });
    }

    // Write NPZ
    std.debug.print("\nWriting NPZ...\n", .{});
    var npz_writer = NpzWriter.init(allocator);
    defer npz_writer.deinit();

    try npz_writer.addAll(&cap);
    try npz_writer.write(args.output_path);

    std.debug.print("Wrote {s}\n", .{args.output_path});
    std.debug.print("\nDone! Compare with:\n", .{});
    std.debug.print("  cd tools/archs\n", .{});
    std.debug.print("  uv run python -m compare _reference/<model>.npz {s}\n", .{args.output_path});
}
