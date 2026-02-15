//! C API for file/image inspect + image decode/convert/encode/model input.

const std = @import("std");
const image = @import("../image/root.zig");
const exif = @import("../image/exif.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");

const allocator = std.heap.c_allocator;
const default_limits = image.Limits{};

const c = @cImport({
    @cInclude("spng.h");
    @cInclude("webp/decode.h");
});

const tjhandle = ?*anyopaque;
extern fn tjInitDecompress() tjhandle;
extern fn tjDestroy(handle: tjhandle) c_int;
extern fn tjDecompressHeader3(
    handle: tjhandle,
    jpeg_buf: [*]const u8,
    jpeg_size: c_ulong,
    width: *c_int,
    height: *c_int,
    jpeg_subsamp: *c_int,
    jpeg_colorspace: *c_int,
) c_int;

const MAGIC_NONE: c_int = 0;
const MAGIC_MIME_TYPE: c_int = 0x0000010;

extern fn magic_open(flags: c_int) ?*anyopaque;
extern fn magic_close(cookie: ?*anyopaque) void;
extern fn magic_load_buffers(cookie: ?*anyopaque, buffers: [*]?*anyopaque, sizes: [*]usize, n: usize) c_int;
extern fn magic_buffer(cookie: ?*anyopaque, buffer: ?*const anyopaque, length: usize) ?[*:0]const u8;

pub const TaluImage = extern struct {
    data: ?[*]u8,
    len: usize,
    width: u32,
    height: u32,
    stride: u32,
    format: c_int, // 0=gray8, 1=rgb8, 2=rgba8
};

pub const TaluImageDecodeOptions = extern struct {
    max_input_bytes: usize,
    max_dimension: u32,
    max_pixels: u64,
    max_output_bytes: usize,
    prefer_format: c_int, // 0=gray8, 1=rgb8, 2=rgba8
    apply_orientation: u8,
    alpha_mode: c_int, // 0=keep, 1=discard, 2=composite
    alpha_background_r: u8,
    alpha_background_g: u8,
    alpha_background_b: u8,
    _reserved: [21]u8,
};

pub const TaluImageResizeOptions = extern struct {
    enabled: u8,
    out_w: u32,
    out_h: u32,
    fit_mode: c_int, // 0=stretch, 1=contain, 2=cover
    filter: c_int, // 0=nearest, 1=bilinear, 2=bicubic
    pad_r: u8,
    pad_g: u8,
    pad_b: u8,
    _reserved: [21]u8,
};

pub const TaluImageConvertOptions = extern struct {
    format: c_int,
    alpha_mode: c_int,
    alpha_background_r: u8,
    alpha_background_g: u8,
    alpha_background_b: u8,
    resize: TaluImageResizeOptions,
    max_input_bytes: usize,
    max_dimension: u32,
    max_pixels: u64,
    max_output_bytes: usize,
};

pub const TaluModelInputSpec = extern struct {
    width: u32,
    height: u32,
    dtype: c_int, // 0=u8, 1=f32
    layout: c_int, // 0=nhwc, 1=nchw
    normalize: c_int, // 0=none, 1=zero_to_one, 2=imagenet
    fit_mode: c_int, // 0=stretch, 1=contain, 2=cover
    filter: c_int, // 0=nearest, 1=bilinear, 2=bicubic
    pad_r: u8,
    pad_g: u8,
    pad_b: u8,
    _reserved0: [5]u8,
    max_input_bytes: usize,
    max_dimension: u32,
    max_pixels: u64,
    max_output_bytes: usize,
};

pub const TaluModelBuffer = extern struct {
    data: ?[*]u8,
    len: usize,
    width: u32,
    height: u32,
    channels: u8,
    layout: c_int,
    dtype: c_int,
    _reserved: [6]u8,
};

pub const TaluImageEncodeOptions = extern struct {
    format: c_int, // 0=jpeg, 1=png
    jpeg_quality: u8,
    _reserved: [27]u8,
};

pub const TaluFileInfo = extern struct {
    kind: c_int, // 0=unknown, 1=image
    mime_ptr: ?[*]u8,
    mime_len: usize,
    description_ptr: ?[*]u8,
    description_len: usize,
    image_format: c_int, // 0=unknown, 1=jpeg, 2=png, 3=webp
    width: u32,
    height: u32,
    exif_orientation: u8, // 1..8 for images, 0 if unknown
    _reserved: [7]u8,
};

pub const TaluFileTransformOptions = extern struct {
    resize_enabled: u8,
    out_w: u32,
    out_h: u32,
    fit_mode: c_int, // 0=stretch, 1=contain, 2=cover
    filter: c_int, // 0=nearest, 1=bilinear, 2=bicubic
    pad_r: u8,
    pad_g: u8,
    pad_b: u8,
    output_format: c_int, // 0=auto, 1=jpeg, 2=png
    jpeg_quality: u8,
    max_input_bytes: usize,
    max_dimension: u32,
    max_pixels: u64,
    max_output_bytes: usize,
    _reserved: [14]u8,
};

fn inspectFileImpl(input: []const u8, out: *TaluFileInfo) !void {
    if (try detectMagicString(input, MAGIC_MIME_TYPE)) |mime| {
        out.mime_ptr = mime.ptr;
        out.mime_len = mime.len;
        if (std.mem.startsWith(u8, mime, "image/")) out.kind = 1;
    }
    if (try detectMagicString(input, MAGIC_NONE)) |desc| {
        out.description_ptr = desc.ptr;
        out.description_len = desc.len;
    }
    const fmt = image.detectFormat(input);
    if (fmt) |image_fmt| {
        out.kind = 1;
        out.image_format = imageFormatToFileC(image_fmt);
        if (probeImageMeta(input, image_fmt)) |meta| {
            out.width = meta.width;
            out.height = meta.height;
            out.exif_orientation = meta.exif_orientation;
        } else |_| {}
    }
    try ensureFallbackClassification(out, input, fmt);
}

pub export fn talu_file_inspect(
    bytes: ?[*]const u8,
    bytes_len: usize,
    out_info: ?*TaluFileInfo,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_info orelse {
        capi_error.setError(error.InvalidArgument, "out_info is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(TaluFileInfo);
    if (bytes == null or bytes_len == 0) {
        capi_error.setError(error.InvalidArgument, "bytes is null or empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    inspectFileImpl(bytes.?[0..bytes_len], out) catch |err| {
        capi_error.setError(err, "file inspect failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_file_info_free(info: ?*TaluFileInfo) callconv(.c) void {
    if (info) |p| {
        if (p.mime_ptr) |ptr| {
            if (p.mime_len > 0) allocator.free(ptr[0..p.mime_len]);
        }
        if (p.description_ptr) |ptr| {
            if (p.description_len > 0) allocator.free(ptr[0..p.description_len]);
        }
        p.* = std.mem.zeroes(TaluFileInfo);
    }
}

pub export fn talu_file_transform(
    bytes: ?[*]const u8,
    bytes_len: usize,
    opts: ?*const TaluFileTransformOptions,
    out_bytes: ?*?[*]u8,
    out_len: ?*usize,
    out_info: ?*TaluFileInfo,
) callconv(.c) i32 {
    capi_error.clearError();

    const out_ptr = out_bytes orelse {
        capi_error.setError(error.InvalidArgument, "out_bytes is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_n = out_len orelse {
        capi_error.setError(error.InvalidArgument, "out_len is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out_ptr.* = null;
    out_n.* = 0;
    if (out_info) |info| info.* = std.mem.zeroes(TaluFileInfo);

    if (bytes == null or bytes_len == 0) {
        capi_error.setError(error.InvalidArgument, "bytes is null or empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const input = bytes.?[0..bytes_len];
    const transform_opts = parseFileTransformOptions(opts) catch |err| {
        capi_error.setError(err, "invalid file transform options", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const input_format = image.detectFormat(input) orelse {
        capi_error.setError(error.UnsupportedImageFormat, "unsupported input format for transform", .{});
        return @intFromEnum(error_codes.ErrorCode.convert_unsupported_format);
    };

    const result = image.transformImage(allocator, input, input_format, .{
        .limits = transform_opts.limits,
        .resize = transform_opts.resize,
        .output_format = transform_opts.output_format,
        .jpeg_quality = transform_opts.jpeg_quality,
        .pad_color = transform_opts.pad_color,
    }) catch |err| {
        capi_error.setError(err, "file transform failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_ptr.* = if (result.data.len == 0) null else result.data.ptr;
    out_n.* = result.data.len;
    if (out_info) |info| {
        info.* = std.mem.zeroes(TaluFileInfo);
        info.kind = 1;
        info.image_format = switch (result.encode_format) { .jpeg => 1, .png => 2 };
        info.width = result.width;
        info.height = result.height;
        info.exif_orientation = 1;
    }
    return 0;
}

pub export fn talu_file_bytes_free(bytes: ?[*]u8, len: usize) callconv(.c) void {
    if (bytes) |ptr| {
        if (len > 0) allocator.free(ptr[0..len]);
    }
}

pub export fn talu_image_decode(
    bytes: ?[*]const u8,
    bytes_len: usize,
    opts: ?*const TaluImageDecodeOptions,
    out_image: ?*TaluImage,
) callconv(.c) i32 {
    capi_error.clearError();

    const out = out_image orelse {
        capi_error.setError(error.InvalidArgument, "out_image is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(TaluImage);

    if (bytes == null or bytes_len == 0) {
        capi_error.setError(error.InvalidArgument, "bytes is null or empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const decode_opts = parseDecodeOptions(opts) catch |err| {
        capi_error.setError(err, "invalid decode options", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const decoded = image.decode(allocator, bytes.?[0..bytes_len], decode_opts) catch |err| {
        capi_error.setError(err, "image decode failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = toCImage(decoded);
    return 0;
}

pub export fn talu_image_convert(
    src: ?*const TaluImage,
    opts: ?*const TaluImageConvertOptions,
    out_image: ?*TaluImage,
) callconv(.c) i32 {
    capi_error.clearError();

    const out = out_image orelse {
        capi_error.setError(error.InvalidArgument, "out_image is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(TaluImage);

    const src_img = src orelse {
        capi_error.setError(error.InvalidArgument, "src is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const borrowed = toBorrowedImage(src_img) catch |err| {
        capi_error.setError(err, "invalid source image", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const spec = parseConvertOptions(opts) catch |err| {
        capi_error.setError(err, "invalid convert options", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const converted = image.convert(allocator, borrowed, spec) catch |err| {
        capi_error.setError(err, "image convert failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = toCImage(converted);
    return 0;
}

pub export fn talu_image_to_model_input(
    bytes: ?[*]const u8,
    bytes_len: usize,
    spec: ?*const TaluModelInputSpec,
    out_buffer: ?*TaluModelBuffer,
) callconv(.c) i32 {
    capi_error.clearError();

    const out = out_buffer orelse {
        capi_error.setError(error.InvalidArgument, "out_buffer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(TaluModelBuffer);

    if (bytes == null or bytes_len == 0) {
        capi_error.setError(error.InvalidArgument, "bytes is null or empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const model_spec = parseModelInputSpec(spec) catch |err| {
        capi_error.setError(err, "invalid model input spec", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const model_buf = image.toModelInput(allocator, bytes.?[0..bytes_len], model_spec) catch |err| {
        capi_error.setError(err, "to_model_input failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = .{
        .data = if (model_buf.data.len == 0) null else model_buf.data.ptr,
        .len = model_buf.data.len,
        .width = model_buf.width,
        .height = model_buf.height,
        .channels = model_buf.channels,
        .layout = layoutToC(model_buf.layout),
        .dtype = dtypeToC(model_buf.dtype),
        ._reserved = std.mem.zeroes([6]u8),
    };
    return 0;
}

pub export fn talu_image_encode(
    src: ?*const TaluImage,
    opts: ?*const TaluImageEncodeOptions,
    out_bytes: ?*?[*]u8,
    out_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const out_ptr = out_bytes orelse {
        capi_error.setError(error.InvalidArgument, "out_bytes is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out_n = out_len orelse {
        capi_error.setError(error.InvalidArgument, "out_len is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out_ptr.* = null;
    out_n.* = 0;

    const src_img = src orelse {
        capi_error.setError(error.InvalidArgument, "src is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const borrowed = toBorrowedImage(src_img) catch |err| {
        capi_error.setError(err, "invalid source image", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const encode_opts = parseEncodeOptions(opts) catch |err| {
        capi_error.setError(err, "invalid encode options", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    const encoded = image.encode(allocator, borrowed, encode_opts) catch |err| {
        capi_error.setError(err, "image encode failed: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_ptr.* = if (encoded.len == 0) null else encoded.ptr;
    out_n.* = encoded.len;
    return 0;
}

pub export fn talu_image_free(img: ?*TaluImage) callconv(.c) void {
    if (img) |p| {
        if (p.data) |ptr| {
            if (p.len > 0) allocator.free(ptr[0..p.len]);
        }
        p.* = std.mem.zeroes(TaluImage);
    }
}

pub export fn talu_model_buffer_free(buf: ?*TaluModelBuffer) callconv(.c) void {
    if (buf) |p| {
        if (p.data) |ptr| {
            if (p.len > 0) allocator.free(ptr[0..p.len]);
        }
        p.* = std.mem.zeroes(TaluModelBuffer);
    }
}

pub export fn talu_image_encode_free(bytes: ?[*]u8, len: usize) callconv(.c) void {
    if (bytes) |ptr| {
        if (len > 0) allocator.free(ptr[0..len]);
    }
}

fn toCImage(img: image.Image) TaluImage {
    return .{
        .data = if (img.data.len == 0) null else img.data.ptr,
        .len = img.data.len,
        .width = img.width,
        .height = img.height,
        .stride = img.stride,
        .format = pixelFormatToC(img.format),
    };
}

fn toBorrowedImage(src: *const TaluImage) !image.Image {
    const fmt = pixelFormatFromC(src.format) catch return error.InvalidArgument;
    const ptr = src.data orelse return error.InvalidArgument;

    const bpp = image.bytesPerPixel(fmt);
    const min_stride = src.width * bpp;
    if (src.stride < min_stride) return error.InvalidArgument;

    const required_u64 = @as(u64, src.stride) * @as(u64, src.height);
    const required = std.math.cast(usize, required_u64) orelse return error.InvalidArgument;
    if (src.len < required) return error.InvalidArgument;

    return .{
        .width = src.width,
        .height = src.height,
        .stride = src.stride,
        .format = fmt,
        .data = ptr[0..src.len],
    };
}

fn parseDecodeOptions(opts: ?*const TaluImageDecodeOptions) !image.DecodeOptions {
    if (opts == null) return .{};
    const o = opts.?;
    return .{
        .limits = .{
            .max_input_bytes = if (o.max_input_bytes == 0) default_limits.max_input_bytes else o.max_input_bytes,
            .max_dimension = if (o.max_dimension == 0) default_limits.max_dimension else o.max_dimension,
            .max_pixels = if (o.max_pixels == 0) default_limits.max_pixels else o.max_pixels,
            .max_output_bytes = if (o.max_output_bytes == 0) default_limits.max_output_bytes else o.max_output_bytes,
        },
        .prefer_format = try pixelFormatFromC(o.prefer_format),
        .apply_orientation = o.apply_orientation != 0,
        .alpha = try alphaModeFromC(o.alpha_mode),
        .alpha_background = .{
            .r = o.alpha_background_r,
            .g = o.alpha_background_g,
            .b = o.alpha_background_b,
        },
    };
}

fn parseConvertOptions(opts: ?*const TaluImageConvertOptions) !image.ConvertSpec {
    if (opts == null) return .{};
    const o = opts.?;

    var spec: image.ConvertSpec = .{
        .format = try pixelFormatFromC(o.format),
        .alpha = try alphaModeFromC(o.alpha_mode),
        .alpha_background = .{
            .r = o.alpha_background_r,
            .g = o.alpha_background_g,
            .b = o.alpha_background_b,
        },
        .limits = .{
            .max_input_bytes = if (o.max_input_bytes == 0) default_limits.max_input_bytes else o.max_input_bytes,
            .max_dimension = if (o.max_dimension == 0) default_limits.max_dimension else o.max_dimension,
            .max_pixels = if (o.max_pixels == 0) default_limits.max_pixels else o.max_pixels,
            .max_output_bytes = if (o.max_output_bytes == 0) default_limits.max_output_bytes else o.max_output_bytes,
        },
    };

    if (o.resize.enabled != 0) {
        spec.resize = .{
            .out_w = o.resize.out_w,
            .out_h = o.resize.out_h,
            .fit = try fitModeFromC(o.resize.fit_mode),
            .filter = try filterFromC(o.resize.filter),
            .pad_color = .{ .r = o.resize.pad_r, .g = o.resize.pad_g, .b = o.resize.pad_b },
        };
    }

    return spec;
}

fn parseModelInputSpec(spec: ?*const TaluModelInputSpec) !image.ModelInputSpec {
    if (spec == null) return error.InvalidArgument;
    const s = spec.?;

    return .{
        .width = s.width,
        .height = s.height,
        .dtype = try dtypeFromC(s.dtype),
        .layout = try layoutFromC(s.layout),
        .normalize = try normalizeFromC(s.normalize),
        .fit = try fitModeFromC(s.fit_mode),
        .filter = try filterFromC(s.filter),
        .pad_color = .{ .r = s.pad_r, .g = s.pad_g, .b = s.pad_b },
        .limits = .{
            .max_input_bytes = if (s.max_input_bytes == 0) default_limits.max_input_bytes else s.max_input_bytes,
            .max_dimension = if (s.max_dimension == 0) default_limits.max_dimension else s.max_dimension,
            .max_pixels = if (s.max_pixels == 0) default_limits.max_pixels else s.max_pixels,
            .max_output_bytes = if (s.max_output_bytes == 0) default_limits.max_output_bytes else s.max_output_bytes,
        },
    };
}

fn parseEncodeOptions(opts: ?*const TaluImageEncodeOptions) !image.EncodeOptions {
    if (opts == null) return error.InvalidArgument;
    const o = opts.?;
    return .{
        .format = try encodeFormatFromC(o.format),
        .jpeg_quality = if (o.jpeg_quality == 0) 85 else o.jpeg_quality,
    };
}

const ParsedFileTransformOptions = struct {
    limits: image.Limits,
    resize: ?image.ResizeOptions,
    output_format: ?image.EncodeFormat,
    jpeg_quality: u8,
    pad_color: image.Rgb8,
};

fn parseFileTransformOptions(opts: ?*const TaluFileTransformOptions) !ParsedFileTransformOptions {
    if (opts == null) {
        return .{
            .limits = default_limits,
            .resize = null,
            .output_format = null,
            .jpeg_quality = 85,
            .pad_color = .{ .r = 0, .g = 0, .b = 0 },
        };
    }

    const o = opts.?;
    const lim: image.Limits = .{
        .max_input_bytes = if (o.max_input_bytes == 0) default_limits.max_input_bytes else o.max_input_bytes,
        .max_dimension = if (o.max_dimension == 0) default_limits.max_dimension else o.max_dimension,
        .max_pixels = if (o.max_pixels == 0) default_limits.max_pixels else o.max_pixels,
        .max_output_bytes = if (o.max_output_bytes == 0) default_limits.max_output_bytes else o.max_output_bytes,
    };

    const resize_opts: ?image.ResizeOptions = if (o.resize_enabled != 0) blk: {
        if (o.out_w == 0 or o.out_h == 0) return error.InvalidImageDimensions;
        break :blk .{
            .out_w = o.out_w,
            .out_h = o.out_h,
            .fit = try fitModeFromC(o.fit_mode),
            .filter = try filterFromC(o.filter),
            .pad_color = .{ .r = o.pad_r, .g = o.pad_g, .b = o.pad_b },
        };
    } else null;

    return .{
        .limits = lim,
        .resize = resize_opts,
        .output_format = try fileOutputFormatFromC(o.output_format),
        .jpeg_quality = if (o.jpeg_quality == 0) 85 else o.jpeg_quality,
        .pad_color = .{ .r = o.pad_r, .g = o.pad_g, .b = o.pad_b },
    };
}

const ImageMeta = struct {
    width: u32,
    height: u32,
    exif_orientation: u8,
};

fn probeImageMeta(bytes: []const u8, fmt: image.Format) !ImageMeta {
    return switch (fmt) {
        .jpeg => try probeJpegMeta(bytes),
        .png => try probePngMeta(bytes),
        .webp => try probeWebpMeta(bytes),
    };
}

fn probeJpegMeta(bytes: []const u8) !ImageMeta {
    const handle = tjInitDecompress() orelse return error.JpegInitFailed;
    defer _ = tjDestroy(handle);

    var w: c_int = 0;
    var h: c_int = 0;
    var subsamp: c_int = 0;
    var colorspace: c_int = 0;

    if (tjDecompressHeader3(
        handle,
        bytes.ptr,
        @intCast(bytes.len),
        &w,
        &h,
        &subsamp,
        &colorspace,
    ) != 0) return error.JpegHeaderFailed;
    if (w <= 0 or h <= 0) return error.InvalidImageDimensions;

    return .{
        .width = @intCast(w),
        .height = @intCast(h),
        .exif_orientation = @intFromEnum(exif.parseJpegOrientation(bytes)),
    };
}

fn probePngMeta(bytes: []const u8) !ImageMeta {
    const ctx = c.spng_ctx_new(0) orelse return error.PngInitFailed;
    defer c.spng_ctx_free(ctx);

    if (c.spng_set_png_buffer(ctx, bytes.ptr, bytes.len) != 0) return error.PngSetBufferFailed;

    var ihdr: c.struct_spng_ihdr = std.mem.zeroes(c.struct_spng_ihdr);
    if (c.spng_get_ihdr(ctx, &ihdr) != 0) return error.PngHeaderFailed;
    if (ihdr.width == 0 or ihdr.height == 0) return error.InvalidImageDimensions;

    return .{
        .width = ihdr.width,
        .height = ihdr.height,
        .exif_orientation = 1,
    };
}

fn probeWebpMeta(bytes: []const u8) !ImageMeta {
    var w: c_int = 0;
    var h: c_int = 0;
    if (c.WebPGetInfo(bytes.ptr, bytes.len, &w, &h) == 0) return error.WebpHeaderFailed;
    if (w <= 0 or h <= 0) return error.InvalidImageDimensions;
    return .{
        .width = @intCast(w),
        .height = @intCast(h),
        .exif_orientation = 1,
    };
}

fn detectMagicString(bytes: []const u8, flags: c_int) !?[]u8 {
    if (bytes.len == 0) return null;
    const cookie = magic_open(flags) orelse return null;
    defer magic_close(cookie);

    const magic_db = @import("magic_db").data;
    if (magic_db.len == 0) return null;

    const db_ptr: ?*anyopaque = @ptrCast(@constCast(magic_db.ptr));
    var db_ptrs = [_]?*anyopaque{db_ptr};
    var db_sizes = [_]usize{magic_db.len};
    if (magic_load_buffers(cookie, &db_ptrs, &db_sizes, 1) != 0) return null;

    const raw = magic_buffer(cookie, bytes.ptr, bytes.len) orelse return null;
    const value = std.mem.sliceTo(raw, 0);
    if (value.len == 0) return null;
    return try allocator.dupe(u8, value);
}

fn ensureFallbackClassification(out: *TaluFileInfo, bytes: []const u8, fmt: ?image.Format) !void {
    if (out.mime_ptr == null) {
        const mime = if (fmt) |image_fmt|
            imageMime(image_fmt)
        else if (looksLikeText(bytes))
            "text/plain"
        else
            "application/octet-stream";
        try setOwnedBytes(&out.mime_ptr, &out.mime_len, mime);
    }

    if (out.description_ptr == null) {
        const desc = if (fmt) |image_fmt|
            imageDescription(image_fmt)
        else if (looksLikeText(bytes))
            "ASCII text"
        else
            "data";
        try setOwnedBytes(&out.description_ptr, &out.description_len, desc);
    }
}

fn setOwnedBytes(out_ptr: *?[*]u8, out_len: *usize, value: []const u8) !void {
    const owned = try allocator.dupe(u8, value);
    out_ptr.* = if (owned.len == 0) null else owned.ptr;
    out_len.* = owned.len;
}

fn imageMime(fmt: image.Format) []const u8 {
    return switch (fmt) {
        .jpeg => "image/jpeg",
        .png => "image/png",
        .webp => "image/webp",
    };
}

fn imageDescription(fmt: image.Format) []const u8 {
    return switch (fmt) {
        .jpeg => "JPEG image data",
        .png => "PNG image data",
        .webp => "WebP image data",
    };
}

fn looksLikeText(bytes: []const u8) bool {
    if (bytes.len == 0) return false;

    const sample_len = @min(bytes.len, 4096);
    const sample = bytes[0..sample_len];
    var non_text: usize = 0;

    for (sample) |b| {
        switch (b) {
            0x09, 0x0A, 0x0D => {},
            0x20...0x7E => {},
            else => non_text += 1,
        }
    }

    return non_text * 20 <= sample_len;
}

fn fileOutputFormatFromC(v: c_int) !?image.EncodeFormat {
    return switch (v) {
        0 => null,
        1 => .jpeg,
        2 => .png,
        else => error.InvalidArgument,
    };
}

fn imageFormatToFileC(v: image.Format) c_int {
    return switch (v) {
        .jpeg => 1,
        .png => 2,
        .webp => 3,
    };
}

fn pixelFormatFromC(v: c_int) !image.PixelFormat {
    return switch (v) {
        0 => .gray8,
        1 => .rgb8,
        2 => .rgba8,
        else => error.InvalidArgument,
    };
}

fn pixelFormatToC(v: image.PixelFormat) c_int {
    return switch (v) {
        .gray8 => 0,
        .rgb8 => 1,
        .rgba8 => 2,
    };
}

fn alphaModeFromC(v: c_int) !image.AlphaMode {
    return switch (v) {
        0 => .keep,
        1 => .discard,
        2 => .composite,
        else => error.InvalidArgument,
    };
}

fn fitModeFromC(v: c_int) !image.FitMode {
    return switch (v) {
        0 => .stretch,
        1 => .contain,
        2 => .cover,
        else => error.InvalidArgument,
    };
}

fn filterFromC(v: c_int) !image.ResizeFilter {
    return switch (v) {
        0 => .nearest,
        1 => .bilinear,
        2 => .bicubic,
        else => error.InvalidArgument,
    };
}

fn dtypeFromC(v: c_int) !image.DType {
    return switch (v) {
        0 => .u8,
        1 => .f32,
        else => error.InvalidArgument,
    };
}

fn dtypeToC(v: image.DType) c_int {
    return switch (v) {
        .u8 => 0,
        .f32 => 1,
    };
}

fn layoutFromC(v: c_int) !image.TensorLayout {
    return switch (v) {
        0 => .nhwc,
        1 => .nchw,
        else => error.InvalidArgument,
    };
}

fn layoutToC(v: image.TensorLayout) c_int {
    return switch (v) {
        .nhwc => 0,
        .nchw => 1,
    };
}

fn normalizeFromC(v: c_int) !image.Normalize {
    return switch (v) {
        0 => .none,
        1 => .zero_to_one,
        2 => .imagenet,
        else => error.InvalidArgument,
    };
}

fn encodeFormatFromC(v: c_int) !image.EncodeFormat {
    return switch (v) {
        0 => .jpeg,
        1 => .png,
        else => error.InvalidArgument,
    };
}
