//! Tokenizer handle with model context.
//!
//! Combines a tokenizer with model-specific configuration (generation config,
//! resolved paths). This is the complete "loaded tokenizer" used by the C API.

const std = @import("std");
const api_mod = @import("api.zig");
const Tokenizer = api_mod.Tokenizer;
const gen_config_mod = @import("../io/config/generation.zig");
const GenerationConfig = gen_config_mod.GenerationConfig;
const io = @import("../io/root.zig");
const repository = io.repository;

/// A tokenizer with associated model context.
///
/// Combines the core tokenizer with:
/// - Resolved model directory path
/// - Generation config (EOS tokens, BOS handling)
/// - Resolved BOS token ID
///
/// Thread safety: NOT thread-safe. Create one instance per thread.
pub const TokenizerHandle = struct {
    allocator: std.mem.Allocator,
    /// The underlying tokenizer.
    tok: Tokenizer,
    model_dir: []const u8,
    gen_config: GenerationConfig,
    resolved_bos_id: ?i32 = null,

    /// Initialize a tokenizer handle directly from JSON content.
    ///
    /// Creates a minimal tokenizer without model directory or generation config.
    /// Useful for testing or standalone tokenization without a full model.
    /// Caller owns the returned handle and must call deinit() when done.
    pub fn initFromJson(allocator: std.mem.Allocator, json: []const u8) !*TokenizerHandle {
        var tok = try Tokenizer.initFromJson(allocator, json);
        errdefer tok.deinit();

        // Create empty model_dir (owned string required for deinit)
        const empty_dir = try allocator.dupe(u8, "");
        errdefer allocator.free(empty_dir);

        // Allocate and initialize handle with minimal config
        const handle = try allocator.create(TokenizerHandle);
        handle.* = .{
            .allocator = allocator,
            .tok = tok,
            .model_dir = empty_dir,
            .gen_config = GenerationConfig{},
            .resolved_bos_id = null,
        };

        return handle;
    }

    /// Initialize a tokenizer handle from a model path.
    ///
    /// Handles path resolution, tokenizer loading, and generation config.
    /// Caller owns the returned handle and must call deinit() when done.
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !*TokenizerHandle {
        // Resolve model path (handles HuggingFace cache format)
        const resolved_path = try repository.resolveModelPath(allocator, model_path, .{});
        errdefer allocator.free(resolved_path);

        // Load generation config (for EOS tokens, BOS handling)
        var gen_config = gen_config_mod.loadGenerationConfig(allocator, resolved_path) catch GenerationConfig{};
        errdefer gen_config.deinit(allocator);

        // Resolve model assets
        var assets = try repository.resolve(allocator, resolved_path);
        defer assets.deinit();

        // Load tokenizer from JSON or binary path
        var tok = if (assets.tokenizer_json()) |json|
            try Tokenizer.initFromJson(allocator, json)
        else
            try Tokenizer.initFromPath(allocator, assets.tokenizer_path());
        errdefer tok.deinit();

        // Add model-specific EOS tokens
        gen_config_mod.addEosFromTokenizer(allocator, tok.tokenizer_handle, &gen_config, "<end_of_turn>");
        gen_config_mod.addEosFromTokenizer(allocator, tok.tokenizer_handle, &gen_config, "<eos>");

        // Resolve BOS token ID from string if configured
        const resolved_bos_id: ?i32 = if (gen_config.bos_token_str) |bos|
            tok.tokenizer_handle.tokenToId(bos)
        else
            null;

        // Allocate and initialize handle
        const handle = try allocator.create(TokenizerHandle);
        handle.* = .{
            .allocator = allocator,
            .tok = tok,
            .model_dir = resolved_path,
            .gen_config = gen_config,
            .resolved_bos_id = resolved_bos_id,
        };

        return handle;
    }

    /// Free all resources owned by the tokenizer handle.
    pub fn deinit(self: *TokenizerHandle) void {
        self.tok.deinit();
        self.gen_config.deinit(self.allocator);
        self.allocator.free(self.model_dir);
        self.allocator.destroy(self);
    }

    /// Get tokens starting with a specific byte (for grammar engine optimization).
    pub fn getTokensStartingWith(self: *const TokenizerHandle, byte: u8) []const u32 {
        return self.tok.getTokensStartingWith(byte);
    }

    /// Get token bytes for grammar validation.
    pub fn tokenBytes(self: *const TokenizerHandle, token_id: usize) ?[]const u8 {
        return self.tok.tokenBytes(token_id);
    }

    /// Get vocabulary size.
    pub fn getVocabSize(self: *const TokenizerHandle) usize {
        return self.tok.vocab_size;
    }
};
