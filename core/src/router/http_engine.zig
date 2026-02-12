//! HTTP Engine - Remote inference via OpenAI-compatible APIs.
//!
//! Implements remote inference for OpenAI-compatible endpoints (vLLM, Ollama,
//! OpenAI, Llama.cpp server, etc.) using the /v1/chat/completions endpoint.
//!
//! ## Architecture
//!
//! HttpEngine mirrors the LocalEngine API but sends requests to a remote server:
//!   - `generate()` → POST /v1/chat/completions (non-streaming)
//!   - `stream()` → POST /v1/chat/completions (streaming with SSE)
//!
//! ## Usage
//!
//! ```zig
//! var engine = try HttpEngine.init(allocator, .{
//!     .base_url = "http://localhost:8000/v1",
//!     .api_key = "sk-...",
//!     .model = "org/model-name",
//! });
//! defer engine.deinit();
//!
//! const result = try engine.generate(&chat, .{});
//! defer result.deinit(allocator);
//! ```
//!
//! ## SSE Streaming
//!
//! The streaming implementation parses Server-Sent Events (SSE):
//!   - `data: {...}` lines contain JSON chunks
//!   - `data: [DONE]` signals stream completion
//!   - Content is extracted from `choices[0].delta.content`

const std = @import("std");
const responses_mod = @import("../responses/root.zig");
const Chat = responses_mod.Chat;
const protocol = @import("protocol/root.zig");
const io = @import("../io/root.zig");
const log = @import("../log.zig");

const c = @cImport({
    @cInclude("curl/curl.h");
});

// =============================================================================
// Configuration
// =============================================================================

const max_http_response_bytes: usize = 10 * 1024 * 1024;
const max_header_json_bytes: usize = 256 * 1024;
const max_sse_chunk_bytes: usize = 1 * 1024 * 1024;

/// Configuration for HttpEngine.
pub const HttpEngineConfig = struct {
    /// Base URL for the API (e.g., "http://localhost:8000/v1").
    /// If it doesn't end with /chat/completions, it will be appended.
    base_url: [:0]const u8,

    /// API key for authentication (optional for local servers).
    api_key: ?[:0]const u8 = null,

    /// Organization ID (optional).
    org_id: ?[:0]const u8 = null,

    /// Model name to use in requests (e.g., "gpt-4o", "org/model-name").
    model: [:0]const u8,

    /// Request timeout in milliseconds.
    timeout_ms: i32 = 60_000,

    /// Connection timeout in milliseconds (for initial TCP connect).
    /// Separate from request timeout for faster failure detection.
    connect_timeout_ms: i32 = 2_000,

    /// Maximum number of retries for failed requests.
    max_retries: i32 = 3,

    /// Custom HTTP headers as JSON object string.
    /// Format: {"Header-Name": "value", "Another-Header": "value2"}
    /// These are added to every request to this backend.
    custom_headers_json: ?[:0]const u8 = null,
};

// =============================================================================
// Generation Types
// =============================================================================

/// Generation options for remote inference.
pub const GenerateOptions = struct {
    /// Maximum tokens to generate.
    max_tokens: ?usize = null,

    /// Sampling temperature (0 = greedy).
    temperature: ?f32 = null,

    /// Top-k sampling.
    top_k: ?usize = null,

    /// Top-p (nucleus) sampling.
    top_p: ?f32 = null,

    /// Callback for streaming tokens. Called with each content chunk.
    stream_callback: ?StreamCallback = null,

    /// User data passed to the stream callback.
    callback_data: ?*anyopaque = null,

    /// Stop sequences.
    stop: ?[]const []const u8 = null,

    /// Extra body fields as JSON object string (without outer braces).
    /// These are appended to the request body for provider-specific parameters.
    /// Example: "\"repetition_penalty\": 1.1, \"top_a\": 0.5"
    extra_body_json: ?[]const u8 = null,

    /// Tool definitions as JSON array string (OpenAI tools format).
    /// When non-null, included in the request body as "tools": <this value>.
    tools_json: ?[]const u8 = null,

    /// Tool choice strategy. One of:
    ///   "auto"     - model decides (default when tools present)
    ///   "none"     - never call tools
    ///   "required" - must call a tool
    ///   "<name>"   - call a specific function by name
    tool_choice: ?[]const u8 = null,

    /// When true, preserve raw stream content without reasoning-tag filtering.
    /// This is consumed by router/iterator.zig.
    raw_output: bool = false,
};

/// Callback for streaming content chunks.
pub const StreamCallback = *const fn (content: []const u8, user_data: ?*anyopaque) bool;

/// Result from generation.
pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8,
};

pub const GenerationResult = struct {
    /// Generated text content.
    text: []const u8,

    /// Number of prompt tokens (if reported by API).
    prompt_tokens: usize,

    /// Number of completion tokens.
    completion_tokens: usize,

    /// Finish reason (stop, length, etc.).
    finish_reason: FinishReason,

    /// Tool calls requested by the model (if finish_reason == .tool_calls).
    tool_calls: []const ToolCall = &.{},

    /// Free the result's memory.
    pub fn deinit(self: *const GenerationResult, allocator: std.mem.Allocator) void {
        for (self.tool_calls) |tc| {
            allocator.free(tc.id);
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        if (self.tool_calls.len > 0) allocator.free(self.tool_calls);
        allocator.free(self.text);
    }
};

pub const FinishReason = enum {
    stop,
    length,
    tool_calls,
    content_filter,
    unknown,
};

// =============================================================================
// HTTP Engine
// =============================================================================

/// Model information returned by listModels.
pub const ModelInfo = struct {
    /// Model ID (e.g., "gpt-4o", "org/model-name").
    id: []const u8,

    /// Object type (usually "model").
    object: []const u8,

    /// Unix timestamp when the model was created (if available).
    created: ?i64,

    /// Owner/organization (if available).
    owned_by: []const u8,

    /// Free the model info's memory.
    pub fn deinit(self: *const ModelInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.object);
        allocator.free(self.owned_by);
    }
};

/// Result from listModels.
pub const ListModelsResult = struct {
    /// Array of model info structs.
    models: []ModelInfo,

    /// Free all models and the array.
    pub fn deinit(self: *const ListModelsResult, allocator: std.mem.Allocator) void {
        for (self.models) |*model| {
            model.deinit(allocator);
        }
        allocator.free(self.models);
    }
};

/// HTTP Engine for remote inference via OpenAI-compatible APIs.
pub const HttpEngine = struct {
    allocator: std.mem.Allocator,

    /// Base URL (e.g., "http://localhost:8000/v1").
    base_url: []u8,

    /// Full URL for chat completions endpoint.
    endpoint_url: []u8,

    /// API key (owned copy).
    api_key: ?[]u8,

    /// Organization ID (owned copy).
    org_id: ?[]u8,

    /// Model name (owned copy).
    model: []u8,

    /// Request timeout in milliseconds.
    timeout_ms: i32,

    /// Connection timeout in milliseconds.
    connect_timeout_ms: i32,

    /// Maximum retries.
    max_retries: i32,

    /// Custom HTTP headers as JSON object string (owned copy).
    custom_headers_json: ?[]u8,

    /// Initialize HttpEngine from configuration.
    pub fn init(allocator: std.mem.Allocator, config: HttpEngineConfig) !HttpEngine {
        // Build the full endpoint URL
        const base_url_input = config.base_url[0..config.base_url.len];
        const base_url = try buildBaseUrl(allocator, base_url_input);
        errdefer allocator.free(base_url);

        const endpoint_url = try buildEndpointUrl(allocator, base_url_input);
        errdefer allocator.free(endpoint_url);

        // Copy strings
        const api_key = if (config.api_key) |key|
            try allocator.dupe(u8, key[0..key.len])
        else
            null;
        errdefer if (api_key) |k| allocator.free(k);

        const org_id = if (config.org_id) |org|
            try allocator.dupe(u8, org[0..org.len])
        else
            null;
        errdefer if (org_id) |o| allocator.free(o);

        const model = try allocator.dupe(u8, config.model[0..config.model.len]);
        errdefer allocator.free(model);

        const custom_headers_json = if (config.custom_headers_json) |headers|
            try allocator.dupe(u8, headers[0..headers.len])
        else
            null;
        errdefer if (custom_headers_json) |h| allocator.free(h);

        return HttpEngine{
            .allocator = allocator,
            .base_url = base_url,
            .endpoint_url = endpoint_url,
            .api_key = api_key,
            .org_id = org_id,
            .model = model,
            .timeout_ms = config.timeout_ms,
            .connect_timeout_ms = config.connect_timeout_ms,
            .max_retries = config.max_retries,
            .custom_headers_json = custom_headers_json,
        };
    }

    /// Free all resources.
    pub fn deinit(self: *HttpEngine) void {
        self.allocator.free(self.base_url);
        self.allocator.free(self.endpoint_url);
        if (self.api_key) |k| self.allocator.free(k);
        if (self.org_id) |o| self.allocator.free(o);
        self.allocator.free(self.model);
        if (self.custom_headers_json) |h| self.allocator.free(h);
        self.* = undefined;
    }

    /// Generate a response for a chat (non-streaming).
    pub fn generate(self: *HttpEngine, chat: *Chat, opts: GenerateOptions) !GenerationResult {
        // Build request JSON
        const request_json = try self.buildRequestJson(chat, opts, false);
        defer self.allocator.free(request_json);

        log.debug("http_engine", "Sending request", .{
            .url = self.endpoint_url,
            .model = self.model,
            .stream = false,
            .body_len = request_json.len,
            .body = request_json[0..@min(request_json.len, 2048)],
        }, @src());

        // Send HTTP request
        const response = try self.sendRequest(request_json, null);
        defer self.allocator.free(response);

        log.debug("http_engine", "Response received", .{
            .len = response.len,
            .body = response[0..@min(response.len, 2048)],
        }, @src());

        // Parse response
        return self.parseResponse(response);
    }

    /// Generate a response with streaming.
    pub fn stream(self: *HttpEngine, chat: *Chat, opts: GenerateOptions) !GenerationResult {
        // Build request JSON with streaming enabled
        const request_json = try self.buildRequestJson(chat, opts, true);
        defer self.allocator.free(request_json);

        log.debug("http_engine", "Sending streaming request", .{
            .url = self.endpoint_url,
            .model = self.model,
            .stream = true,
            .body_len = request_json.len,
            .body = request_json[0..@min(request_json.len, 2048)],
        }, @src());

        // Create streaming context
        var stream_ctx = StreamingContext{
            .allocator = self.allocator,
            .callback = opts.stream_callback,
            .callback_data = opts.callback_data,
            .content_buffer = .{},
            .line_buffer = .{},
            .finish_reason = .unknown,
            .prompt_tokens = 0,
            .completion_tokens = 0,
            .tool_calls = .{},
        };
        defer stream_ctx.line_buffer.deinit(self.allocator);

        // Send request with streaming callback
        const response = self.sendRequest(request_json, &stream_ctx) catch |err| {
            stream_ctx.content_buffer.deinit(self.allocator);
            stream_ctx.deinitToolCalls();
            return err;
        };
        defer if (response.len > 0) self.allocator.free(response);

        // Convert accumulated streaming tool calls to owned ToolCall slices
        var tool_calls_owned: []const ToolCall = &.{};
        if (stream_ctx.tool_calls.items.len > 0) {
            var owned = try self.allocator.alloc(ToolCall, stream_ctx.tool_calls.items.len);
            errdefer self.allocator.free(owned);

            var built: usize = 0;
            errdefer {
                for (owned[0..built]) |tc| {
                    self.allocator.free(tc.id);
                    self.allocator.free(tc.name);
                    self.allocator.free(tc.arguments);
                }
            }

            for (stream_ctx.tool_calls.items, 0..) |*stc, i| {
                owned[i] = .{
                    .id = try stc.id.toOwnedSlice(self.allocator),
                    .name = try stc.name.toOwnedSlice(self.allocator),
                    .arguments = try stc.arguments.toOwnedSlice(self.allocator),
                };
                built += 1;
            }
            tool_calls_owned = owned;
        }
        stream_ctx.deinitToolCalls();

        // Return accumulated content
        const text = try stream_ctx.content_buffer.toOwnedSlice(self.allocator);

        return GenerationResult{
            .text = text,
            .prompt_tokens = stream_ctx.prompt_tokens,
            .completion_tokens = stream_ctx.completion_tokens,
            .finish_reason = stream_ctx.finish_reason,
            .tool_calls = tool_calls_owned,
        };
    }

    /// List available models from the remote endpoint.
    ///
    /// Calls GET /v1/models and returns a list of available model IDs.
    /// This is useful for discovering what models are available on a vLLM,
    /// Ollama, or other OpenAI-compatible server.
    ///
    /// Uses a short timeout (1 second) and no retries since this is a
    /// discovery operation that should fail fast if the server is unavailable.
    pub fn listModels(self: *HttpEngine) !ListModelsResult {
        // Build models endpoint URL
        const models_url = try std.fmt.allocPrint(self.allocator, "{s}/models", .{self.base_url});
        defer self.allocator.free(models_url);

        log.debug("http_engine", "Listing models", .{
            .url = models_url,
        }, @src());

        // Send GET request with short timeout and no retries (discovery should be fast)
        const response = try self.sendGetRequestWithOptions(models_url, .{
            .timeout_ms = 1_000, // 1 second total timeout
            .connect_timeout_ms = 1_000, // 1 second connect timeout
            .max_retries = 0, // No retries for discovery
        });
        defer self.allocator.free(response);

        // Parse response
        return self.parseModelsResponse(response);
    }

    // =========================================================================
    // Internal Implementation
    // =========================================================================

    /// Build JSON request body.
    ///
    /// Uses the protocol.completions serializer to convert Items to the
    /// legacy Chat Completions format required by OpenAI-compatible APIs.
    fn buildRequestJson(self: *HttpEngine, chat: *Chat, opts: GenerateOptions, enable_stream: bool) ![]u8 {
        // Serialize messages using protocol layer (single source of truth)
        const messages_json = try protocol.completions.serialize(
            self.allocator,
            chat.conv,
            .{}, // Default options
        );
        defer self.allocator.free(messages_json);

        var json_buffer = std.ArrayListUnmanaged(u8){};
        errdefer json_buffer.deinit(self.allocator);
        var writer = json_buffer.writer(self.allocator);

        try writer.writeAll("{");

        // Model
        try writer.print("\"model\":\"{s}\"", .{self.model});

        // Messages (from protocol serializer)
        try writer.writeAll(",\"messages\":");
        try writer.writeAll(messages_json);

        // Streaming
        if (enable_stream) {
            try writer.writeAll(",\"stream\":true");
        }

        // Optional parameters
        if (opts.max_tokens) |max| {
            try writer.print(",\"max_tokens\":{d}", .{max});
        }
        if (opts.temperature) |temp| {
            try writer.print(",\"temperature\":{d:.6}", .{temp});
        }
        if (opts.top_p) |top_p| {
            try writer.print(",\"top_p\":{d:.6}", .{top_p});
        }

        // Stop sequences
        if (opts.stop) |stops| {
            try writer.writeAll(",\"stop\":[");
            for (stops, 0..) |s, i| {
                if (i > 0) try writer.writeAll(",");
                try writer.writeAll("\"");
                try writeJsonEscaped(writer, s);
                try writer.writeAll("\"");
            }
            try writer.writeAll("]");
        }

        // Tools
        if (opts.tools_json) |tools| {
            if (tools.len > 0) {
                try writer.writeAll(",\"tools\":");
                try writer.writeAll(tools);
            }
        }
        if (opts.tool_choice) |choice| {
            if (choice.len > 0) {
                if (std.mem.eql(u8, choice, "auto") or
                    std.mem.eql(u8, choice, "none") or
                    std.mem.eql(u8, choice, "required"))
                {
                    try writer.print(",\"tool_choice\":\"{s}\"", .{choice});
                } else {
                    try writer.writeAll(",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"");
                    try writeJsonEscaped(writer, choice);
                    try writer.writeAll("\"}}");
                }
            }
        }

        // Extra body fields (provider-specific parameters)
        // The extra_body_json is expected to be the inner content of a JSON object
        // (without the outer braces), e.g.: "\"repetition_penalty\": 1.1, \"top_a\": 0.5"
        if (opts.extra_body_json) |extra| {
            if (extra.len > 0) {
                // Check if extra starts with '{' - if so, strip braces
                const content = if (extra[0] == '{' and extra[extra.len - 1] == '}')
                    extra[1 .. extra.len - 1]
                else
                    extra;
                if (content.len > 0) {
                    try writer.writeAll(",");
                    try writer.writeAll(content);
                }
            }
        }

        try writer.writeAll("}");
        return json_buffer.toOwnedSlice(self.allocator);
    }

    /// Send HTTP POST request.
    /// When `stream_ctx` is non-null, data is forwarded to the SSE parser
    /// incrementally while also being buffered for error diagnostics.
    fn sendRequest(
        self: *HttpEngine,
        body: []const u8,
        stream_ctx: ?*StreamingContext,
    ) ![]u8 {
        const curl_handle = c.curl_easy_init() orelse return error.CurlInitFailed;
        defer c.curl_easy_cleanup(curl_handle);

        // URL
        const url_z = try self.allocator.dupeZ(u8, self.endpoint_url);
        defer self.allocator.free(url_z);
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_URL, url_z.ptr) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        if (c.curl_easy_setopt(
            curl_handle,
            c.CURLOPT_MAXFILESIZE_LARGE,
            @as(c.curl_off_t, @intCast(max_http_response_bytes)),
        ) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        // POST method
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_POST, @as(c_long, 1)) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        // Request body
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_POSTFIELDS, body.ptr) != c.CURLE_OK)
            return error.CurlSetOptFailed;
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_POSTFIELDSIZE, @as(c_long, @intCast(body.len))) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        // Headers
        var headers: ?*c.struct_curl_slist = null;
        headers = c.curl_slist_append(headers, "Content-Type: application/json");
        headers = c.curl_slist_append(headers, "Accept: application/json");

        if (self.api_key) |key| {
            const auth_header = try std.fmt.allocPrint(self.allocator, "Authorization: Bearer {s}", .{key});
            defer self.allocator.free(auth_header);
            const auth_header_z = try self.allocator.dupeZ(u8, auth_header);
            defer self.allocator.free(auth_header_z);
            headers = c.curl_slist_append(headers, auth_header_z.ptr);
        }

        if (self.org_id) |org| {
            const org_header = try std.fmt.allocPrint(self.allocator, "OpenAI-Organization: {s}", .{org});
            defer self.allocator.free(org_header);
            const org_header_z = try self.allocator.dupeZ(u8, org_header);
            defer self.allocator.free(org_header_z);
            headers = c.curl_slist_append(headers, org_header_z.ptr);
        }

        // Add custom headers from JSON
        // Format: {"Header-Name": "value", "Another-Header": "value2"}
        if (self.custom_headers_json) |json| {
            const parsed = try io.json.parseValue(self.allocator, json, .{
                .max_size_bytes = max_header_json_bytes,
                .max_value_bytes = max_header_json_bytes,
                .max_string_bytes = max_header_json_bytes,
            });
            defer parsed.deinit();

            if (parsed.value == .object) {
                var it = parsed.value.object.iterator();
                while (it.next()) |entry| {
                    const value = entry.value_ptr.*;
                    if (value != .string) continue;
                    const header_str = try std.fmt.allocPrint(self.allocator, "{s}: {s}", .{ entry.key_ptr.*, value.string });
                    defer self.allocator.free(header_str);
                    const header_z = try self.allocator.dupeZ(u8, header_str);
                    defer self.allocator.free(header_z);
                    headers = c.curl_slist_append(headers, header_z.ptr);
                }
            }
        }

        defer if (headers) |h| c.curl_slist_free_all(h);
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_HTTPHEADER, headers) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        // Timeout
        if (self.timeout_ms > 0) {
            if (c.curl_easy_setopt(curl_handle, c.CURLOPT_TIMEOUT_MS, @as(c_long, self.timeout_ms)) != c.CURLE_OK)
                return error.CurlSetOptFailed;
        }
        if (self.connect_timeout_ms > 0) {
            if (c.curl_easy_setopt(curl_handle, c.CURLOPT_CONNECTTIMEOUT_MS, @as(c_long, self.connect_timeout_ms)) != c.CURLE_OK)
                return error.CurlSetOptFailed;
        }

        // Always buffer the full response via curlWriteCallback.
        // In streaming mode, curlWriteCallback also forwards data to the
        // SSE parser via streaming_ctx, giving us both incremental token
        // delivery and a complete response buffer for error diagnostics.
        var response_buffer = CurlWriteContext{
            .allocator = self.allocator,
            .data = std.ArrayListUnmanaged(u8){},
            .streaming_ctx = stream_ctx,
            .max_size = max_http_response_bytes,
            .exceeded = false,
        };
        errdefer response_buffer.data.deinit(self.allocator);

        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_WRITEFUNCTION, @as(*const anyopaque, @ptrCast(&curlWriteCallback))) != c.CURLE_OK)
            return error.CurlSetOptFailed;
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_WRITEDATA, @as(*anyopaque, @ptrCast(&response_buffer))) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        // SSL configuration
        const native_os = @import("builtin").os.tag;
        if (native_os == .macos) {
            _ = c.curl_easy_setopt(curl_handle, c.CURLOPT_CAINFO, "/etc/ssl/cert.pem");
        }

        // Perform request with retries
        var last_error: c.CURLcode = c.CURLE_OK;
        var attempt: i32 = 0;
        while (attempt <= self.max_retries) : (attempt += 1) {
            last_error = c.curl_easy_perform(curl_handle);
            if (last_error == c.CURLE_OK) break;

            if (attempt < self.max_retries) {
                std.Thread.sleep(std.time.ns_per_s * @as(u64, @intCast(attempt + 1)));
            }
        }

        if (last_error != c.CURLE_OK) {
            if (response_buffer.exceeded) return error.ResponseTooLarge;
            return error.CurlPerformFailed;
        }
        if (response_buffer.exceeded) return error.ResponseTooLarge;

        // Check HTTP status
        var status_code: c_long = 0;
        _ = c.curl_easy_getinfo(curl_handle, c.CURLINFO_RESPONSE_CODE, &status_code);
        if (status_code >= 400) {
            log.warn("http_engine", "HTTP error", .{
                .status = status_code,
                .response = if (response_buffer.data.items.len > 0)
                    response_buffer.data.items[0..@min(response_buffer.data.items.len, 1024)]
                else
                    "",
            });
            return httpStatusToError(status_code);
        }

        if (stream_ctx != null) {
            // Streaming mode — content was already forwarded to the SSE
            // parser by curlWriteCallback. Free the buffer and return empty.
            response_buffer.data.deinit(self.allocator);
            return try self.allocator.alloc(u8, 0);
        }

        return response_buffer.data.toOwnedSlice(self.allocator);
    }

    /// Parse non-streaming response JSON.
    fn parseResponse(self: *HttpEngine, response: []const u8) !GenerationResult {
        // OpenAI response format:
        // {
        //   "choices": [{
        //     "message": {
        //       "content": "...",
        //       "tool_calls": [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
        //     },
        //     "finish_reason": "stop"
        //   }],
        //   "usage": {"prompt_tokens": N, "completion_tokens": N}
        // }

        var content: []const u8 = "";
        var finish_reason: FinishReason = .unknown;
        var prompt_tokens: usize = 0;
        var completion_tokens: usize = 0;

        var tool_calls = std.ArrayListUnmanaged(ToolCall){};
        errdefer {
            for (tool_calls.items) |tc| {
                self.allocator.free(tc.id);
                self.allocator.free(tc.name);
                self.allocator.free(tc.arguments);
            }
            tool_calls.deinit(self.allocator);
        }

        const parsed = try io.json.parseValue(self.allocator, response, .{
            .max_size_bytes = max_http_response_bytes,
            .max_value_bytes = max_http_response_bytes,
            .max_string_bytes = max_http_response_bytes,
        });
        defer parsed.deinit();

        if (parsed.value == .object) {
            if (getObjectField(parsed.value, "choices")) |choices_val| {
                if (choices_val == .array and choices_val.array.items.len > 0) {
                    const choice = choices_val.array.items[0];
                    if (choice == .object) {
                        if (getStringField(choice, "finish_reason")) |reason| {
                            finish_reason = parseFinishReason(reason);
                        }
                        if (getObjectField(choice, "message")) |message_val| {
                            if (message_val == .object) {
                                if (getStringField(message_val, "content")) |content_str| {
                                    content = content_str;
                                }
                                if (getObjectField(message_val, "tool_calls")) |tool_calls_val| {
                                    if (tool_calls_val == .array) {
                                        for (tool_calls_val.array.items) |tc_val| {
                                            if (tc_val != .object) continue;
                                            const id = getStringField(tc_val, "id") orelse continue;
                                            const function_val = getObjectField(tc_val, "function") orelse continue;
                                            const name = getStringField(function_val, "name") orelse continue;
                                            const arguments = getStringField(function_val, "arguments") orelse continue;

                                            const owned_id = try self.allocator.dupe(u8, id);
                                            errdefer self.allocator.free(owned_id);
                                            const owned_name = try self.allocator.dupe(u8, name);
                                            errdefer self.allocator.free(owned_name);
                                            const owned_args = try self.allocator.dupe(u8, arguments);
                                            errdefer self.allocator.free(owned_args);

                                            try tool_calls.append(self.allocator, .{
                                                .id = owned_id,
                                                .name = owned_name,
                                                .arguments = owned_args,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (getObjectField(parsed.value, "usage")) |usage_val| {
                if (usage_val == .object) {
                    if (getIntField(usage_val, "prompt_tokens")) |n| {
                        if (n >= 0) prompt_tokens = @intCast(n);
                    }
                    if (getIntField(usage_val, "completion_tokens")) |n| {
                        if (n >= 0) completion_tokens = @intCast(n);
                    }
                }
            }
        }

        if (tool_calls.items.len > 0) {
            finish_reason = .tool_calls;
        }

        log.debug("http_engine", "Parsed response", .{
            .finish_reason = @tagName(finish_reason),
            .tool_calls = tool_calls.items.len,
            .prompt_tokens = prompt_tokens,
            .completion_tokens = completion_tokens,
            .content_len = content.len,
            .content_preview = content[0..@min(content.len, 200)],
        }, @src());

        const text = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(text);
        const owned_tool_calls = if (tool_calls.items.len > 0)
            try tool_calls.toOwnedSlice(self.allocator)
        else
            &.{};

        return GenerationResult{
            .text = text,
            .prompt_tokens = prompt_tokens,
            .completion_tokens = completion_tokens,
            .finish_reason = finish_reason,
            .tool_calls = owned_tool_calls,
        };
    }

    /// Options for GET requests.
    const GetRequestOptions = struct {
        /// Override request timeout (null = use engine default).
        timeout_ms: ?i32 = null,
        /// Override connect timeout (null = use engine default).
        connect_timeout_ms: ?i32 = null,
        /// Override max retries (null = use engine default).
        max_retries: ?i32 = null,
    };

    /// Send HTTP GET request.
    fn sendGetRequest(self: *HttpEngine, url: []const u8) ![]u8 {
        return self.sendGetRequestWithOptions(url, .{});
    }

    /// Send HTTP GET request with custom options.
    fn sendGetRequestWithOptions(self: *HttpEngine, url: []const u8, opts: GetRequestOptions) ![]u8 {
        const timeout_ms = opts.timeout_ms orelse self.timeout_ms;
        const connect_timeout_ms = opts.connect_timeout_ms orelse self.connect_timeout_ms;
        const max_retries = opts.max_retries orelse self.max_retries;
        const curl_handle = c.curl_easy_init() orelse return error.CurlInitFailed;
        defer c.curl_easy_cleanup(curl_handle);

        // URL
        const url_z = try self.allocator.dupeZ(u8, url);
        defer self.allocator.free(url_z);
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_URL, url_z.ptr) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        if (c.curl_easy_setopt(
            curl_handle,
            c.CURLOPT_MAXFILESIZE_LARGE,
            @as(c.curl_off_t, @intCast(max_http_response_bytes)),
        ) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        // GET is the default method, no need to set explicitly

        // Headers
        var headers: ?*c.struct_curl_slist = null;
        headers = c.curl_slist_append(headers, "Accept: application/json");

        if (self.api_key) |key| {
            const auth_header = try std.fmt.allocPrint(self.allocator, "Authorization: Bearer {s}", .{key});
            defer self.allocator.free(auth_header);
            const auth_header_z = try self.allocator.dupeZ(u8, auth_header);
            defer self.allocator.free(auth_header_z);
            headers = c.curl_slist_append(headers, auth_header_z.ptr);
        }

        if (self.org_id) |org| {
            const org_header = try std.fmt.allocPrint(self.allocator, "OpenAI-Organization: {s}", .{org});
            defer self.allocator.free(org_header);
            const org_header_z = try self.allocator.dupeZ(u8, org_header);
            defer self.allocator.free(org_header_z);
            headers = c.curl_slist_append(headers, org_header_z.ptr);
        }

        // Add custom headers from JSON
        // Format: {"Header-Name": "value", "Another-Header": "value2"}
        if (self.custom_headers_json) |json| {
            const parsed = try io.json.parseValue(self.allocator, json, .{
                .max_size_bytes = max_header_json_bytes,
                .max_value_bytes = max_header_json_bytes,
                .max_string_bytes = max_header_json_bytes,
            });
            defer parsed.deinit();

            if (parsed.value == .object) {
                var it = parsed.value.object.iterator();
                while (it.next()) |entry| {
                    const value = entry.value_ptr.*;
                    if (value != .string) continue;
                    const header_str = try std.fmt.allocPrint(self.allocator, "{s}: {s}", .{ entry.key_ptr.*, value.string });
                    defer self.allocator.free(header_str);
                    const header_z = try self.allocator.dupeZ(u8, header_str);
                    defer self.allocator.free(header_z);
                    headers = c.curl_slist_append(headers, header_z.ptr);
                }
            }
        }

        defer if (headers) |h| c.curl_slist_free_all(h);
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_HTTPHEADER, headers) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        // Timeout (use local variables from opts)
        if (timeout_ms > 0) {
            if (c.curl_easy_setopt(curl_handle, c.CURLOPT_TIMEOUT_MS, @as(c_long, timeout_ms)) != c.CURLE_OK)
                return error.CurlSetOptFailed;
        }
        if (connect_timeout_ms > 0) {
            if (c.curl_easy_setopt(curl_handle, c.CURLOPT_CONNECTTIMEOUT_MS, @as(c_long, connect_timeout_ms)) != c.CURLE_OK)
                return error.CurlSetOptFailed;
        }

        // Response handling
        var response_buffer = CurlWriteContext{
            .allocator = self.allocator,
            .data = std.ArrayListUnmanaged(u8){},
            .streaming_ctx = null,
            .max_size = max_http_response_bytes,
            .exceeded = false,
        };
        errdefer response_buffer.data.deinit(self.allocator);

        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_WRITEFUNCTION, @as(*const anyopaque, @ptrCast(&curlWriteCallback))) != c.CURLE_OK)
            return error.CurlSetOptFailed;
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_WRITEDATA, @as(*anyopaque, @ptrCast(&response_buffer))) != c.CURLE_OK)
            return error.CurlSetOptFailed;

        // SSL configuration
        const native_os = @import("builtin").os.tag;
        if (native_os == .macos) {
            _ = c.curl_easy_setopt(curl_handle, c.CURLOPT_CAINFO, "/etc/ssl/cert.pem");
        }

        // Perform request with retries (uses local max_retries from opts)
        var last_error: c.CURLcode = c.CURLE_OK;
        var attempt: i32 = 0;
        while (attempt <= max_retries) : (attempt += 1) {
            last_error = c.curl_easy_perform(curl_handle);
            if (last_error == c.CURLE_OK) break;

            if (attempt < max_retries) {
                std.Thread.sleep(std.time.ns_per_s * @as(u64, @intCast(attempt + 1)));
            }
        }

        if (last_error != c.CURLE_OK) {
            if (response_buffer.exceeded) return error.ResponseTooLarge;
            return error.CurlPerformFailed;
        }
        if (response_buffer.exceeded) return error.ResponseTooLarge;

        // Check HTTP status (GET request)
        var status_code: c_long = 0;
        _ = c.curl_easy_getinfo(curl_handle, c.CURLINFO_RESPONSE_CODE, &status_code);
        if (status_code >= 400) {
            log.warn("http_engine", "HTTP GET error", .{
                .status = status_code,
                .response = if (response_buffer.data.items.len > 0)
                    response_buffer.data.items[0..@min(response_buffer.data.items.len, 200)]
                else
                    "",
            });
            return httpStatusToError(status_code);
        }

        return response_buffer.data.toOwnedSlice(self.allocator);
    }

    /// Parse models list response JSON.
    /// Format: {"data": [{"id": "model-1", "object": "model", "created": 123, "owned_by": "org"}]}
    fn parseModelsResponse(self: *HttpEngine, response: []const u8) !ListModelsResult {
        var models = std.ArrayListUnmanaged(ModelInfo){};
        errdefer {
            for (models.items) |*m| m.deinit(self.allocator);
            models.deinit(self.allocator);
        }

        const parsed = try io.json.parseValue(self.allocator, response, .{
            .max_size_bytes = max_http_response_bytes,
            .max_value_bytes = max_http_response_bytes,
            .max_string_bytes = max_http_response_bytes,
        });
        defer parsed.deinit();

        const data_val = getObjectField(parsed.value, "data") orelse return ListModelsResult{ .models = &.{} };
        if (data_val != .array) return ListModelsResult{ .models = &.{} };

        for (data_val.array.items) |entry| {
            if (entry != .object) continue;
            const id = getStringField(entry, "id") orelse "";
            if (id.len == 0) continue;
            const object = getStringField(entry, "object") orelse "model";
            const owned_by = getStringField(entry, "owned_by") orelse "";
            const created = getIntField(entry, "created");

            const model_info = ModelInfo{
                .id = try self.allocator.dupe(u8, id),
                .object = try self.allocator.dupe(u8, object),
                .owned_by = try self.allocator.dupe(u8, owned_by),
                .created = created,
            };
            try models.append(self.allocator, model_info);
        }

        return ListModelsResult{
            .models = try models.toOwnedSlice(self.allocator),
        };
    }
};

// =============================================================================
// Streaming Support
// =============================================================================

/// A tool call being accumulated from SSE deltas.
const StreamingToolCall = struct {
    id: std.ArrayListUnmanaged(u8),
    name: std.ArrayListUnmanaged(u8),
    arguments: std.ArrayListUnmanaged(u8),

    fn deinit(self: *StreamingToolCall, alloc: std.mem.Allocator) void {
        self.id.deinit(alloc);
        self.name.deinit(alloc);
        self.arguments.deinit(alloc);
    }
};

/// Context for streaming response handling.
const StreamingContext = struct {
    allocator: std.mem.Allocator,
    callback: ?StreamCallback,
    callback_data: ?*anyopaque,
    content_buffer: std.ArrayListUnmanaged(u8),
    line_buffer: std.ArrayListUnmanaged(u8),
    finish_reason: FinishReason,
    prompt_tokens: usize,
    completion_tokens: usize,
    tool_calls: std.ArrayListUnmanaged(StreamingToolCall),

    fn deinitToolCalls(self: *StreamingContext) void {
        for (self.tool_calls.items) |*tc| tc.deinit(self.allocator);
        self.tool_calls.deinit(self.allocator);
    }
};

/// Process buffered SSE lines.
fn processSSELines(ctx: *StreamingContext) !void {
    while (true) {
        // Find newline
        const newline_pos = std.mem.indexOf(u8, ctx.line_buffer.items, "\n") orelse break;

        // Extract line (without newline)
        var line = ctx.line_buffer.items[0..newline_pos];

        // Remove carriage return if present
        if (line.len > 0 and line[line.len - 1] == '\r') {
            line = line[0 .. line.len - 1];
        }

        // Process the line
        try processSSELine(ctx, line);

        // Remove processed line from buffer
        const remaining = ctx.line_buffer.items[newline_pos + 1 ..];
        std.mem.copyForwards(u8, ctx.line_buffer.items[0..remaining.len], remaining);
        ctx.line_buffer.shrinkRetainingCapacity(remaining.len);
    }
}

/// Accumulate a single tool call delta into the streaming context.
///
/// OpenAI streaming sends tool calls incrementally:
///   {"index":0, "id":"call_abc", "function":{"name":"search","arguments":""}}
///   {"index":0, "function":{"arguments":"{\"q"}}
///   {"index":0, "function":{"arguments":"uery\"}"}}
///
/// The `index` field identifies which tool call slot to append to.
fn accumulateToolCallDelta(ctx: *StreamingContext, tc_delta: std.json.Value) !void {
    const index: usize = blk: {
        if (getIntField(tc_delta, "index")) |idx| {
            if (idx >= 0) break :blk @intCast(idx);
        }
        break :blk 0;
    };

    // Grow tool_calls list if needed
    while (ctx.tool_calls.items.len <= index) {
        try ctx.tool_calls.append(ctx.allocator, .{
            .id = .{},
            .name = .{},
            .arguments = .{},
        });
    }

    var tc = &ctx.tool_calls.items[index];

    if (getStringField(tc_delta, "id")) |id| {
        try tc.id.appendSlice(ctx.allocator, id);
    }
    if (getObjectField(tc_delta, "function")) |func| {
        if (getStringField(func, "name")) |name| {
            try tc.name.appendSlice(ctx.allocator, name);
        }
        if (getStringField(func, "arguments")) |args| {
            try tc.arguments.appendSlice(ctx.allocator, args);
        }
    }
}

/// Process a single SSE line.
fn processSSELine(ctx: *StreamingContext, line: []const u8) !void {
    // Skip empty lines and comments
    if (line.len == 0 or line[0] == ':') return;

    // Parse "data: ..." lines
    if (!std.mem.startsWith(u8, line, "data:")) return;

    var data = line[5..];
    // Skip leading space after "data:"
    if (data.len > 0 and data[0] == ' ') {
        data = data[1..];
    }

    // Check for [DONE] signal
    if (std.mem.eql(u8, data, "[DONE]")) {
        return;
    }

    const parsed = try io.json.parseValue(ctx.allocator, data, .{
        .max_size_bytes = max_sse_chunk_bytes,
        .max_value_bytes = max_sse_chunk_bytes,
        .max_string_bytes = max_sse_chunk_bytes,
    });
    defer parsed.deinit();

    // Parse JSON chunk
    // {"choices":[{"delta":{"content":"..."}}]}
    if (parsed.value == .object) {
        if (getObjectField(parsed.value, "choices")) |choices| {
            if (choices == .array and choices.array.items.len > 0) {
                const choice = choices.array.items[0];
                if (choice == .object) {
                    if (getObjectField(choice, "delta")) |delta| {
                        if (delta == .object) {
                            if (getStringField(delta, "content")) |content| {
                                // Accumulate content
                                try ctx.content_buffer.appendSlice(ctx.allocator, content);

                                // Call user callback if provided
                                if (ctx.callback) |cb| {
                                    _ = cb(content, ctx.callback_data);
                                }
                            }

                            // Accumulate tool call deltas
                            if (getObjectField(delta, "tool_calls")) |tc_val| {
                                if (tc_val == .array) {
                                    for (tc_val.array.items) |tc_delta| {
                                        if (tc_delta != .object) continue;
                                        try accumulateToolCallDelta(ctx, tc_delta);
                                    }
                                }
                            }
                        }
                    }

                    if (getStringField(choice, "finish_reason")) |reason| {
                        ctx.finish_reason = parseFinishReason(reason);
                    }
                }
            }
        }

        // Parse usage from final chunk (some providers include it)
        if (getObjectField(parsed.value, "usage")) |usage_val| {
            if (usage_val == .object) {
                if (getIntField(usage_val, "prompt_tokens")) |n| {
                    if (n >= 0) ctx.prompt_tokens = @intCast(n);
                }
                if (getIntField(usage_val, "completion_tokens")) |n| {
                    if (n >= 0) ctx.completion_tokens = @intCast(n);
                }
            }
        }
    }
}

// =============================================================================
// Curl Helpers
// =============================================================================

const CurlWriteContext = struct {
    allocator: std.mem.Allocator,
    data: std.ArrayListUnmanaged(u8),
    streaming_ctx: ?*StreamingContext,
    max_size: usize,
    exceeded: bool,
};

fn curlWriteCallback(data: [*c]u8, size: usize, nmemb: usize, user_data: *anyopaque) callconv(.c) usize {
    const ctx: *CurlWriteContext = @ptrCast(@alignCast(user_data));
    const total_size = size * nmemb;
    const bytes = data[0..total_size];

    if (ctx.data.items.len + bytes.len > ctx.max_size) {
        ctx.exceeded = true;
        return 0;
    }
    ctx.data.appendSlice(ctx.allocator, bytes) catch return 0;

    // In streaming mode, also forward data to the SSE parser.
    // This lets us buffer the full response for error diagnostics
    // while still streaming tokens to the caller incrementally.
    if (ctx.streaming_ctx) |stream_ctx| {
        stream_ctx.line_buffer.appendSlice(stream_ctx.allocator, bytes) catch return 0;
        processSSELines(stream_ctx) catch return 0;
    }

    return total_size;
}

// =============================================================================
// URL Helpers
// =============================================================================

/// Check if url ends with a versioned API path segment like /v1, /v4, etc.
/// Matches "/v" followed by one or more digits at the end of the string.
fn endsWithVersionPath(url: []const u8) bool {
    // Need at least "/vN" = 3 chars
    if (url.len < 3) return false;
    // Walk backwards past digits
    var i = url.len;
    while (i > 0 and url[i - 1] >= '0' and url[i - 1] <= '9') {
        i -= 1;
    }
    // Must have consumed at least one digit
    if (i == url.len) return false;
    // Check for "/v" prefix
    if (i < 2) return false;
    return url[i - 1] == 'v' and url[i - 2] == '/';
}

/// Build the normalized base URL for constructing sub-endpoints like /models.
///
/// If the URL already ends with a versioned path (e.g. /v1, /v4), it is
/// returned as-is. Otherwise /v1 is appended (OpenAI convention).
fn buildBaseUrl(allocator: std.mem.Allocator, base_url: []const u8) ![]u8 {
    // Remove trailing slash if present
    var url = base_url;
    while (url.len > 0 and url[url.len - 1] == '/') {
        url = url[0 .. url.len - 1];
    }

    // If URL ends with /chat/completions, strip it to get base
    if (std.mem.endsWith(u8, url, "/chat/completions")) {
        const base = url[0 .. url.len - "/chat/completions".len];
        return try allocator.dupe(u8, base);
    }

    // If URL already ends with a versioned path (/v1, /v4, etc.), return as-is
    if (endsWithVersionPath(url)) {
        return try allocator.dupe(u8, url);
    }

    // Default: append /v1 (OpenAI convention)
    return try std.fmt.allocPrint(allocator, "{s}/v1", .{url});
}

/// Build the full chat/completions endpoint URL from base_url.
///
/// Recognises URLs that already end with /chat/completions, a versioned path,
/// or a bare host, and constructs the correct endpoint for each case.
fn buildEndpointUrl(allocator: std.mem.Allocator, base_url: []const u8) ![]u8 {
    // Remove trailing slash if present
    var url = base_url;
    while (url.len > 0 and url[url.len - 1] == '/') {
        url = url[0 .. url.len - 1];
    }

    // Check if URL already ends with /chat/completions
    if (std.mem.endsWith(u8, url, "/chat/completions")) {
        return try allocator.dupe(u8, url);
    }

    // If URL ends with a versioned path (/v1, /v4, etc.), append /chat/completions
    if (endsWithVersionPath(url)) {
        return try std.fmt.allocPrint(allocator, "{s}/chat/completions", .{url});
    }

    // Default: append full path with /v1 (OpenAI convention)
    return try std.fmt.allocPrint(allocator, "{s}/v1/chat/completions", .{url});
}

fn httpStatusToError(status: c_long) error{
    Unauthorized,
    NotFound,
    RateLimited,
    BadRequest,
    ServerError,
    HttpError,
} {
    return switch (status) {
        400 => error.BadRequest,
        401, 403 => error.Unauthorized,
        404 => error.NotFound,
        429 => error.RateLimited,
        500, 502, 503, 504 => error.ServerError,
        else => error.HttpError,
    };
}

// =============================================================================
// JSON Helpers
// =============================================================================

fn getObjectField(value: std.json.Value, field: []const u8) ?std.json.Value {
    if (value != .object) return null;
    return value.object.get(field);
}

fn getStringField(value: std.json.Value, field: []const u8) ?[]const u8 {
    const val = getObjectField(value, field) orelse return null;
    return switch (val) {
        .string => |s| s,
        else => null,
    };
}

fn getIntField(value: std.json.Value, field: []const u8) ?i64 {
    const val = getObjectField(value, field) orelse return null;
    return switch (val) {
        .integer => |i| i,
        .float => |f| if (f >= 0) @intFromFloat(f) else null,
        else => null,
    };
}

/// Write JSON-escaped string.
fn writeJsonEscaped(writer: anytype, s: []const u8) !void {
    for (s) |char| {
        switch (char) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => {
                // Control characters (excluding \t, \n, \r which are handled above)
                try writer.print("\\u{x:0>4}", .{@as(u16, char)});
            },
            else => try writer.writeByte(char),
        }
    }
}

/// Parse finish reason string.
fn parseFinishReason(reason: []const u8) FinishReason {
    if (std.mem.eql(u8, reason, "stop")) return .stop;
    if (std.mem.eql(u8, reason, "length")) return .length;
    if (std.mem.eql(u8, reason, "tool_calls")) return .tool_calls;
    if (std.mem.eql(u8, reason, "content_filter")) return .content_filter;
    return .unknown;
}

// =============================================================================
// Completions Projection
// =============================================================================
//
// NOTE: The Completions format conversion logic has been moved to:
//   router/protocol/completions.zig
//
// This ensures a single source of truth for format conversion and keeps
// the messages module (core/src/messages/) as a pure Item-based data model.
//
// To serialize a Conversation to Completions format:
//   protocol.completions.serialize(allocator, conv, .{})

// =============================================================================
// Tests
// =============================================================================

test "endsWithVersionPath detects versioned API paths" {
    // Standard /v1
    try std.testing.expect(endsWithVersionPath("/v1"));
    try std.testing.expect(endsWithVersionPath("http://localhost:8000/v1"));
    try std.testing.expect(endsWithVersionPath("https://openrouter.ai/api/v1"));

    // Non-v1 versions
    try std.testing.expect(endsWithVersionPath("https://api.z.ai/api/paas/v4"));
    try std.testing.expect(endsWithVersionPath("https://example.com/api/v2"));
    try std.testing.expect(endsWithVersionPath("https://example.com/v10"));

    // Not a version path
    try std.testing.expect(!endsWithVersionPath("http://localhost:8000"));
    try std.testing.expect(!endsWithVersionPath("http://localhost:8000/api"));
    try std.testing.expect(!endsWithVersionPath("http://localhost:8000/v1/chat/completions"));
    try std.testing.expect(!endsWithVersionPath("/v"));
    try std.testing.expect(!endsWithVersionPath("v1"));
    try std.testing.expect(!endsWithVersionPath(""));
}

test "buildEndpointUrl handles various base URLs" {
    const allocator = std.testing.allocator;

    // URL ending with /v1
    {
        const url = try buildEndpointUrl(allocator, "http://localhost:8000/v1");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("http://localhost:8000/v1/chat/completions", url);
    }

    // URL already complete
    {
        const url = try buildEndpointUrl(allocator, "http://localhost:8000/v1/chat/completions");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("http://localhost:8000/v1/chat/completions", url);
    }

    // URL with trailing slash
    {
        const url = try buildEndpointUrl(allocator, "http://localhost:8000/v1/");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("http://localhost:8000/v1/chat/completions", url);
    }

    // Base URL without /v1
    {
        const url = try buildEndpointUrl(allocator, "http://localhost:8000");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("http://localhost:8000/v1/chat/completions", url);
    }

    // Zai-style URL with /v4 (non-v1 version path)
    {
        const url = try buildEndpointUrl(allocator, "https://api.z.ai/api/paas/v4");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("https://api.z.ai/api/paas/v4/chat/completions", url);
    }

    // Zai-style URL with trailing slash
    {
        const url = try buildEndpointUrl(allocator, "https://api.z.ai/api/paas/v4/");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("https://api.z.ai/api/paas/v4/chat/completions", url);
    }
}

test "parseFinishReason handles all cases" {
    try std.testing.expectEqual(FinishReason.stop, parseFinishReason("stop"));
    try std.testing.expectEqual(FinishReason.length, parseFinishReason("length"));
    try std.testing.expectEqual(FinishReason.tool_calls, parseFinishReason("tool_calls"));
    try std.testing.expectEqual(FinishReason.content_filter, parseFinishReason("content_filter"));
    try std.testing.expectEqual(FinishReason.unknown, parseFinishReason("other"));
}

test "writeJsonEscaped handles special characters" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    try writeJsonEscaped(buffer.writer(allocator), "Hello\n\"World\"\t\\");

    try std.testing.expectEqualStrings("Hello\\n\\\"World\\\"\\t\\\\", buffer.items);
}

test "httpStatusToError maps status codes" {
    try std.testing.expectEqual(error.BadRequest, httpStatusToError(400));
    try std.testing.expectEqual(error.Unauthorized, httpStatusToError(401));
    try std.testing.expectEqual(error.Unauthorized, httpStatusToError(403));
    try std.testing.expectEqual(error.NotFound, httpStatusToError(404));
    try std.testing.expectEqual(error.RateLimited, httpStatusToError(429));
    try std.testing.expectEqual(error.ServerError, httpStatusToError(500));
    try std.testing.expectEqual(error.ServerError, httpStatusToError(503));
    try std.testing.expectEqual(error.HttpError, httpStatusToError(418));
}

test "HttpEngineConfig has sensible defaults" {
    const config = HttpEngineConfig{
        .base_url = "http://localhost:8000/v1",
        .model = "test-model",
    };
    try std.testing.expect(config.api_key == null);
    try std.testing.expect(config.org_id == null);
    try std.testing.expectEqual(@as(i32, 60_000), config.timeout_ms);
    try std.testing.expectEqual(@as(i32, 3), config.max_retries);
}

test "HttpEngine.init creates engine with owned copies" {
    const allocator = std.testing.allocator;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .api_key = "sk-test-key",
        .org_id = "org-123",
        .model = "gpt-4o",
        .timeout_ms = 30_000,
        .max_retries = 5,
    });
    defer engine.deinit();

    // Verify owned copies were made
    try std.testing.expectEqualStrings("http://localhost:8000/v1", engine.base_url);
    try std.testing.expectEqualStrings("http://localhost:8000/v1/chat/completions", engine.endpoint_url);
    try std.testing.expectEqualStrings("sk-test-key", engine.api_key.?);
    try std.testing.expectEqualStrings("org-123", engine.org_id.?);
    try std.testing.expectEqualStrings("gpt-4o", engine.model);
    try std.testing.expectEqual(@as(i32, 30_000), engine.timeout_ms);
    try std.testing.expectEqual(@as(i32, 5), engine.max_retries);
}

test "HttpEngine.init handles optional fields as null" {
    const allocator = std.testing.allocator;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .model = "test-model",
    });
    defer engine.deinit();

    try std.testing.expect(engine.api_key == null);
    try std.testing.expect(engine.org_id == null);
}

test "buildBaseUrl normalizes various URL formats" {
    const allocator = std.testing.allocator;

    // URL ending with /v1
    {
        const url = try buildBaseUrl(allocator, "http://localhost:8000/v1");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("http://localhost:8000/v1", url);
    }

    // URL with /chat/completions - should strip to base
    {
        const url = try buildBaseUrl(allocator, "http://localhost:8000/v1/chat/completions");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("http://localhost:8000/v1", url);
    }

    // URL without /v1 - should append
    {
        const url = try buildBaseUrl(allocator, "http://localhost:8000");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("http://localhost:8000/v1", url);
    }

    // URL with trailing slash
    {
        const url = try buildBaseUrl(allocator, "http://localhost:8000/");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("http://localhost:8000/v1", url);
    }

    // Zai-style URL with /v4 - should NOT append /v1
    {
        const url = try buildBaseUrl(allocator, "https://api.z.ai/api/paas/v4");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("https://api.z.ai/api/paas/v4", url);
    }

    // Zai-style with /chat/completions - should strip to v4 base
    {
        const url = try buildBaseUrl(allocator, "https://api.z.ai/api/paas/v4/chat/completions");
        defer allocator.free(url);
        try std.testing.expectEqualStrings("https://api.z.ai/api/paas/v4", url);
    }
}

test "GenerationResult.deinit frees tool_calls" {
    const allocator = std.testing.allocator;
    const text = try allocator.dupe(u8, "generated text");
    errdefer allocator.free(text);
    const tc_id = try allocator.dupe(u8, "call_123");
    errdefer allocator.free(tc_id);
    const tc_name = try allocator.dupe(u8, "get_weather");
    errdefer allocator.free(tc_name);
    const tc_args = try allocator.dupe(u8, "{\"location\":\"Paris\"}");
    errdefer allocator.free(tc_args);
    var tool_calls = try allocator.alloc(ToolCall, 1);
    tool_calls[0] = .{ .id = tc_id, .name = tc_name, .arguments = tc_args };

    const result = GenerationResult{
        .text = text,
        .prompt_tokens = 10,
        .completion_tokens = 5,
        .finish_reason = .stop,
        .tool_calls = tool_calls,
    };
    result.deinit(allocator);
    // No leak = success (allocator will panic if leaked)
}

test "ModelInfo.deinit frees all strings" {
    const allocator = std.testing.allocator;

    const info = ModelInfo{
        .id = try allocator.dupe(u8, "gpt-4o"),
        .object = try allocator.dupe(u8, "model"),
        .created = 1234567890,
        .owned_by = try allocator.dupe(u8, "openai"),
    };
    info.deinit(allocator);
    // No leak = success
}

test "ListModelsResult.deinit frees all models" {
    const allocator = std.testing.allocator;

    var models = try allocator.alloc(ModelInfo, 2);
    errdefer allocator.free(models);

    const id0 = try allocator.dupe(u8, "model-1");
    errdefer allocator.free(id0);
    const obj0 = try allocator.dupe(u8, "model");
    errdefer allocator.free(obj0);
    const owner0 = try allocator.dupe(u8, "owner");
    errdefer allocator.free(owner0);

    models[0] = .{
        .id = id0,
        .object = obj0,
        .created = null,
        .owned_by = owner0,
    };

    const id1 = try allocator.dupe(u8, "model-2");
    errdefer allocator.free(id1);
    const obj1 = try allocator.dupe(u8, "model");
    errdefer allocator.free(obj1);
    const owner1 = try allocator.dupe(u8, "owner");

    models[1] = .{
        .id = id1,
        .object = obj1,
        .created = 12345,
        .owned_by = owner1,
    };

    const result = ListModelsResult{ .models = models };
    result.deinit(allocator);
    // No leak = success
}

test "GenerateOptions has sensible defaults" {
    const opts = GenerateOptions{};
    try std.testing.expect(opts.max_tokens == null);
    try std.testing.expect(opts.temperature == null);
    try std.testing.expect(opts.top_k == null);
    try std.testing.expect(opts.top_p == null);
    try std.testing.expect(opts.stream_callback == null);
    try std.testing.expect(opts.callback_data == null);
    try std.testing.expect(opts.stop == null);
}

test "writeJsonEscaped handles control characters" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    // Test carriage return
    try writeJsonEscaped(buffer.writer(allocator), "line1\rline2");
    try std.testing.expectEqualStrings("line1\\rline2", buffer.items);
}

test "writeJsonEscaped handles empty string" {
    const allocator = std.testing.allocator;
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);

    try writeJsonEscaped(buffer.writer(allocator), "");
    try std.testing.expectEqual(@as(usize, 0), buffer.items.len);
}

test "FinishReason enum values" {
    // Ensure all enum variants exist and are distinct
    try std.testing.expect(FinishReason.stop != FinishReason.length);
    try std.testing.expect(FinishReason.length != FinishReason.tool_calls);
    try std.testing.expect(FinishReason.tool_calls != FinishReason.content_filter);
    try std.testing.expect(FinishReason.content_filter != FinishReason.unknown);
}

// =============================================================================
// Mock API Response Tests
// =============================================================================
// These tests verify parsing logic using mock API responses without HTTP calls.
// Real HTTP integration tests are in bindings/python/tests/.
//
// Test Strategy (following io/transport/hf.zig pattern):
// - Network functions (generate, stream, listModels): signature verification only
// - Parsing functions (parseResponse, parseModelsResponse): comprehensive mock tests
// - SSE parsing (processSSELine, processSSELines): comprehensive mock tests
// - Helper functions: unit tests above
//
// Network behavior is tested in:
// - bindings/python/tests/model/test_remote_backend.py (real vLLM servers)
// - Manual testing with vLLM, Ollama, OpenAI endpoints

// -----------------------------------------------------------------------------
// Network-dependent function coverage (signature verification)
// -----------------------------------------------------------------------------

test "HttpEngine.generate: signature verification" {
    // Verify function signature compiles correctly.
    // Cannot unit test: requires network access to OpenAI-compatible server.
    // Parsing logic tested via parseResponse mock tests below.
    const F = @TypeOf(HttpEngine.generate);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 3), info.params.len); // self, chat, options
}

test "HttpEngine.stream: signature verification" {
    // Verify function signature compiles correctly.
    // Cannot unit test: requires network access to OpenAI-compatible server.
    // SSE parsing logic tested via processSSELine mock tests below.
    const F = @TypeOf(HttpEngine.stream);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 3), info.params.len); // self, chat, options
}

test "HttpEngine.listModels: signature verification" {
    // Verify function signature compiles correctly.
    // Cannot unit test: requires network access to OpenAI-compatible server.
    // Parsing logic tested via parseModelsResponse mock tests below.
    const F = @TypeOf(HttpEngine.listModels);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 1), info.params.len); // self
}

// -----------------------------------------------------------------------------
// Response parsing tests (mock API responses)
// -----------------------------------------------------------------------------

test "parseResponse: mock OpenAI chat completion response" {
    const allocator = std.testing.allocator;

    // Mock response from OpenAI API
    const mock_response =
        \\{
        \\  "id": "chatcmpl-123",
        \\  "object": "chat.completion",
        \\  "created": 1677652288,
        \\  "model": "gpt-4o",
        \\  "choices": [{
        \\    "index": 0,
        \\    "message": {
        \\      "role": "assistant",
        \\      "content": "Hello! How can I help you today?"
        \\    },
        \\    "finish_reason": "stop"
        \\  }],
        \\  "usage": {
        \\    "prompt_tokens": 9,
        \\    "completion_tokens": 12,
        \\    "total_tokens": 21
        \\  }
        \\}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "https://api.openai.com/v1",
        .model = "gpt-4o",
    });
    defer engine.deinit();

    const result = try engine.parseResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("Hello! How can I help you today?", result.text);
    try std.testing.expectEqual(FinishReason.stop, result.finish_reason);
    try std.testing.expectEqual(@as(usize, 9), result.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 12), result.completion_tokens);
}

test "parseResponse: mock response with length finish reason" {
    const allocator = std.testing.allocator;

    const mock_response =
        \\{
        \\  "choices": [{
        \\    "message": {"content": "This is a truncated"},
        \\    "finish_reason": "length"
        \\  }],
        \\  "usage": {"prompt_tokens": 5, "completion_tokens": 100}
        \\}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .model = "test",
    });
    defer engine.deinit();

    const result = try engine.parseResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("This is a truncated", result.text);
    try std.testing.expectEqual(FinishReason.length, result.finish_reason);
}

test "parseResponse: mock vLLM response format" {
    const allocator = std.testing.allocator;

    // vLLM uses same format but may have different fields
    const mock_response =
        \\{
        \\  "id": "cmpl-abc123",
        \\  "object": "chat.completion",
        \\  "choices": [{
        \\    "index": 0,
        \\    "message": {
        \\      "role": "assistant",
        \\      "content": "The answer is 42."
        \\    },
        \\    "finish_reason": "stop"
        \\  }],
        \\  "usage": {
        \\    "prompt_tokens": 15,
        \\    "completion_tokens": 5
        \\  }
        \\}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .model = "org/model-name",
    });
    defer engine.deinit();

    const result = try engine.parseResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("The answer is 42.", result.text);
    try std.testing.expectEqual(@as(usize, 15), result.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 5), result.completion_tokens);
}

test "parseResponse: mock response with empty content" {
    const allocator = std.testing.allocator;

    const mock_response =
        \\{
        \\  "choices": [{
        \\    "message": {"content": ""},
        \\    "finish_reason": "stop"
        \\  }],
        \\  "usage": {"prompt_tokens": 5, "completion_tokens": 0}
        \\}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost/v1",
        .model = "test",
    });
    defer engine.deinit();

    const result = try engine.parseResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqualStrings("", result.text);
    try std.testing.expectEqual(@as(usize, 0), result.completion_tokens);
}

test "parseResponse: extracts tool_calls" {
    const allocator = std.testing.allocator;

    const mock_response =
        \\{
        \\  "choices": [{
        \\    "message": {
        \\      "content": null,
        \\      "tool_calls": [{
        \\        "id": "call_abc123",
        \\        "type": "function",
        \\        "function": {"name": "search", "arguments": "{\"query\":\"zig\"}"}
        \\      }]
        \\    },
        \\    "finish_reason": "tool_calls"
        \\  }],
        \\  "usage": {"prompt_tokens": 7, "completion_tokens": 1}
        \\}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost/v1",
        .model = "test",
    });
    defer engine.deinit();

    const result = try engine.parseResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqual(FinishReason.tool_calls, result.finish_reason);
    try std.testing.expectEqualStrings("", result.text);
    try std.testing.expectEqual(@as(usize, 1), result.tool_calls.len);
    try std.testing.expectEqualStrings("call_abc123", result.tool_calls[0].id);
    try std.testing.expectEqualStrings("search", result.tool_calls[0].name);
    try std.testing.expectEqualStrings("{\"query\":\"zig\"}", result.tool_calls[0].arguments);
}

test "parseResponse: handles mixed content + tools" {
    const allocator = std.testing.allocator;

    const mock_response =
        \\{
        \\  "choices": [{
        \\    "message": {
        \\      "content": "Let me check.",
        \\      "tool_calls": [{
        \\        "id": "call_def456",
        \\        "type": "function",
        \\        "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
        \\      }]
        \\    },
        \\    "finish_reason": "tool_calls"
        \\  }]
        \\}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost/v1",
        .model = "test",
    });
    defer engine.deinit();

    const result = try engine.parseResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqual(FinishReason.tool_calls, result.finish_reason);
    try std.testing.expectEqualStrings("Let me check.", result.text);
    try std.testing.expectEqual(@as(usize, 1), result.tool_calls.len);
    try std.testing.expectEqualStrings("get_weather", result.tool_calls[0].name);
}

test "parseResponse: handles malformed JSON" {
    const allocator = std.testing.allocator;

    const mock_response = "{\"choices\": [";

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost/v1",
        .model = "test",
    });
    defer engine.deinit();

    try std.testing.expectError(io.json.ParseError.InvalidJson, engine.parseResponse(mock_response));
}

test "parseModelsResponse: mock OpenAI models list" {
    const allocator = std.testing.allocator;

    const mock_response =
        \\{
        \\  "object": "list",
        \\  "data": [
        \\    {
        \\      "id": "gpt-4o",
        \\      "object": "model",
        \\      "created": 1686935002,
        \\      "owned_by": "openai"
        \\    },
        \\    {
        \\      "id": "gpt-4o-mini",
        \\      "object": "model",
        \\      "created": 1721172741,
        \\      "owned_by": "system"
        \\    }
        \\  ]
        \\}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "https://api.openai.com/v1",
        .model = "gpt-4o",
    });
    defer engine.deinit();

    const result = try engine.parseModelsResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), result.models.len);
    try std.testing.expectEqualStrings("gpt-4o", result.models[0].id);
    try std.testing.expectEqualStrings("openai", result.models[0].owned_by);
    try std.testing.expectEqual(@as(?i64, 1686935002), result.models[0].created);
    try std.testing.expectEqualStrings("gpt-4o-mini", result.models[1].id);
}

test "parseModelsResponse: mock vLLM models list" {
    const allocator = std.testing.allocator;

    // vLLM typically returns single model
    const mock_response =
        \\{
        \\  "object": "list",
        \\  "data": [
        \\    {
        \\      "id": "org/model-name",
        \\      "object": "model",
        \\      "created": 1234567890,
        \\      "owned_by": "vllm"
        \\    }
        \\  ]
        \\}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .model = "org/model-name",
    });
    defer engine.deinit();

    const result = try engine.parseModelsResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), result.models.len);
    try std.testing.expectEqualStrings("org/model-name", result.models[0].id);
}

test "parseModelsResponse: mock empty models list" {
    const allocator = std.testing.allocator;

    const mock_response =
        \\{"object": "list", "data": []}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost/v1",
        .model = "test",
    });
    defer engine.deinit();

    const result = try engine.parseModelsResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), result.models.len);
}

test "parseModelsResponse: mock Ollama models format" {
    const allocator = std.testing.allocator;

    // Ollama may have slightly different format
    const mock_response =
        \\{
        \\  "object": "list",
        \\  "data": [
        \\    {"id": "llama3:latest", "object": "model", "owned_by": "library"},
        \\    {"id": "codellama:7b", "object": "model", "owned_by": "library"}
        \\  ]
        \\}
    ;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:11434/v1",
        .model = "llama3:latest",
    });
    defer engine.deinit();

    const result = try engine.parseModelsResponse(mock_response);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), result.models.len);
    try std.testing.expectEqualStrings("llama3:latest", result.models[0].id);
    try std.testing.expectEqualStrings("codellama:7b", result.models[1].id);
}

test "processSSELine: mock streaming content chunk" {
    const allocator = std.testing.allocator;

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = null,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    // Mock SSE chunk from OpenAI streaming
    const sse_line = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}";
    try processSSELine(&ctx, sse_line);

    try std.testing.expectEqualStrings("Hello", ctx.content_buffer.items);
}

test "processSSELine: mock multiple streaming chunks" {
    const allocator = std.testing.allocator;

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = null,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    // Simulate multiple chunks
    try processSSELine(&ctx, "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}");
    try processSSELine(&ctx, "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}");
    try processSSELine(&ctx, "data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}]}");

    try std.testing.expectEqualStrings("Hello world!", ctx.content_buffer.items);
}

test "processSSELine: mock DONE signal" {
    const allocator = std.testing.allocator;

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = null,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    try processSSELine(&ctx, "data: {\"choices\":[{\"delta\":{\"content\":\"Test\"}}]}");
    try processSSELine(&ctx, "data: [DONE]");

    // Content should remain, DONE is just a signal
    try std.testing.expectEqualStrings("Test", ctx.content_buffer.items);
}

test "processSSELine: mock finish_reason in stream" {
    const allocator = std.testing.allocator;

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = null,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    // Final chunk often contains finish_reason
    try processSSELine(&ctx, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}");

    try std.testing.expectEqual(FinishReason.stop, ctx.finish_reason);
}

test "processSSELine: ignores empty lines and comments" {
    const allocator = std.testing.allocator;

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = null,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    // Empty line and comment should be ignored
    try processSSELine(&ctx, "");
    try processSSELine(&ctx, ": this is a comment");
    try processSSELine(&ctx, "data: {\"choices\":[{\"delta\":{\"content\":\"X\"}}]}");

    try std.testing.expectEqualStrings("X", ctx.content_buffer.items);
}

test "processSSELines: mock full SSE stream" {
    const allocator = std.testing.allocator;

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = null,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    // Simulate raw SSE data arriving
    const sse_data =
        \\data: {"choices":[{"delta":{"content":"Hi"}}]}
        \\
        \\data: {"choices":[{"delta":{"content":"!"}}]}
        \\
        \\data: [DONE]
        \\
    ;

    try ctx.line_buffer.appendSlice(allocator, sse_data);
    try processSSELines(&ctx);

    try std.testing.expectEqualStrings("Hi!", ctx.content_buffer.items);
}

test "streaming callback receives chunks" {
    const allocator = std.testing.allocator;

    const TestState = struct {
        var chunks: std.ArrayListUnmanaged([]const u8) = .{};
        var alloc: std.mem.Allocator = undefined;

        fn callback(content: []const u8, _: ?*anyopaque) bool {
            const copy = alloc.dupe(u8, content) catch return false;
            chunks.append(alloc, copy) catch {
                alloc.free(copy);
                return false;
            };
            return true;
        }

        fn deinit() void {
            for (chunks.items) |chunk| {
                alloc.free(chunk);
            }
            chunks.deinit(alloc);
            chunks = .{};
        }
    };

    TestState.alloc = allocator;
    defer TestState.deinit();

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = TestState.callback,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    try processSSELine(&ctx, "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}");
    try processSSELine(&ctx, "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}");

    try std.testing.expectEqual(@as(usize, 2), TestState.chunks.items.len);
    try std.testing.expectEqualStrings("A", TestState.chunks.items[0]);
    try std.testing.expectEqualStrings("B", TestState.chunks.items[1]);
}

test "HttpEngineConfig custom_headers_json default" {
    const config = HttpEngineConfig{
        .base_url = "http://localhost:8000/v1",
        .model = "test-model",
    };
    try std.testing.expect(config.custom_headers_json == null);
}

test "processSSELine: accumulates tool call deltas" {
    const allocator = std.testing.allocator;

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = null,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    // First chunk: tool call ID + function name + partial args
    try processSSELine(&ctx,
        \\data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}
    );

    // Second chunk: argument fragment
    try processSSELine(&ctx,
        \\data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q"}}]},"finish_reason":null}]}
    );

    // Third chunk: argument fragment
    try processSSELine(&ctx,
        \\data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"uery\"}"}}]},"finish_reason":null}]}
    );

    // Final chunk: finish reason
    try processSSELine(&ctx,
        \\data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
    );

    try std.testing.expectEqual(@as(usize, 1), ctx.tool_calls.items.len);
    try std.testing.expectEqualStrings("call_abc", ctx.tool_calls.items[0].id.items);
    try std.testing.expectEqualStrings("search", ctx.tool_calls.items[0].name.items);
    try std.testing.expectEqualStrings("{\"query\"}", ctx.tool_calls.items[0].arguments.items);
    try std.testing.expectEqual(FinishReason.tool_calls, ctx.finish_reason);
}

test "processSSELine: accumulates multiple tool calls" {
    const allocator = std.testing.allocator;

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = null,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    // Two tool calls in one chunk
    try processSSELine(&ctx,
        \\data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":"{}"}},{"index":1,"id":"call_2","function":{"name":"weather","arguments":""}}]},"finish_reason":null}]}
    );

    // Second tool call gets more args
    try processSSELine(&ctx,
        \\data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"loc\":\"Paris\"}"}}]},"finish_reason":null}]}
    );

    try std.testing.expectEqual(@as(usize, 2), ctx.tool_calls.items.len);
    try std.testing.expectEqualStrings("call_1", ctx.tool_calls.items[0].id.items);
    try std.testing.expectEqualStrings("search", ctx.tool_calls.items[0].name.items);
    try std.testing.expectEqualStrings("{}", ctx.tool_calls.items[0].arguments.items);
    try std.testing.expectEqualStrings("call_2", ctx.tool_calls.items[1].id.items);
    try std.testing.expectEqualStrings("weather", ctx.tool_calls.items[1].name.items);
    try std.testing.expectEqualStrings("{\"loc\":\"Paris\"}", ctx.tool_calls.items[1].arguments.items);
}

test "processSSELine: SSE usage in final chunk" {
    const allocator = std.testing.allocator;

    var ctx = StreamingContext{
        .allocator = allocator,
        .callback = null,
        .callback_data = null,
        .content_buffer = std.ArrayListUnmanaged(u8){},
        .line_buffer = std.ArrayListUnmanaged(u8){},
        .finish_reason = .unknown,
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .tool_calls = .{},
    };
    defer ctx.content_buffer.deinit(allocator);
    defer ctx.line_buffer.deinit(allocator);
    defer ctx.deinitToolCalls();

    try processSSELine(&ctx,
        \\data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":3}}
    );

    try std.testing.expectEqualStrings("hi", ctx.content_buffer.items);
    try std.testing.expectEqual(@as(usize, 15), ctx.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 3), ctx.completion_tokens);
    try std.testing.expectEqual(FinishReason.stop, ctx.finish_reason);
}

test "HttpEngine.init stores custom_headers_json" {
    const allocator = std.testing.allocator;

    var engine = try HttpEngine.init(allocator, .{
        .base_url = "http://localhost:8000/v1",
        .model = "test-model",
        .custom_headers_json = "{\"X-Custom\": \"value\"}",
    });
    defer engine.deinit();

    try std.testing.expect(engine.custom_headers_json != null);
    try std.testing.expectEqualStrings("{\"X-Custom\": \"value\"}", engine.custom_headers_json.?);
}
