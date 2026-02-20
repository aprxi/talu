//! Image subsystem: decode, convert, encode, and prepare model inputs.

const std = @import("std");

const pixel = @import("pixel.zig");
const limits_mod = @import("limits.zig");
const sniff_mod = @import("sniff.zig");
const exif = @import("exif.zig");
const alpha_mod = @import("alpha.zig");
const resize_mod = @import("resize.zig");
const convert_mod = @import("convert.zig");
const model_input_mod = @import("model_input.zig");
const encode_mod = @import("encode.zig");
const preprocess_mod = @import("preprocess.zig");
const codecs = @import("codecs/root.zig");

pub const Image = pixel.Image;
pub const PixelFormat = pixel.PixelFormat;
pub const Rgb8 = pixel.Rgb8;
pub const ColorPolicy = pixel.ColorPolicy;
pub const Limits = limits_mod.Limits;
pub const Format = sniff_mod.Format;
pub const Orientation = exif.Orientation;
pub const AlphaMode = alpha_mod.AlphaMode;
pub const ResizeFilter = resize_mod.ResizeFilter;
pub const FitMode = resize_mod.FitMode;
pub const ResizeOptions = resize_mod.ResizeOptions;
pub const ConvertSpec = convert_mod.ConvertSpec;
pub const TensorLayout = model_input_mod.TensorLayout;
pub const DType = model_input_mod.DType;
pub const Normalize = model_input_mod.Normalize;
pub const ModelInputSpec = model_input_mod.ModelInputSpec;
pub const ModelBuffer = model_input_mod.ModelBuffer;
pub const EncodeFormat = encode_mod.EncodeFormat;
pub const EncodeOptions = encode_mod.EncodeOptions;
pub const VisionNormalize = preprocess_mod.VisionNormalize;
pub const PlanarF32Spec = preprocess_mod.PlanarF32Spec;
pub const SmartResizeOptions = preprocess_mod.SmartResizeOptions;
pub const SmartResizeResult = preprocess_mod.SmartResizeResult;
pub const VisionGrid = preprocess_mod.VisionGrid;
pub const TokenCountOptions = preprocess_mod.TokenCountOptions;
pub const ExplicitResize = preprocess_mod.ExplicitResize;
pub const VisionPreprocessOptions = preprocess_mod.VisionPreprocessOptions;
pub const VisionPreprocessResult = preprocess_mod.VisionPreprocessResult;
pub const bytesPerPixel = pixel.bytesPerPixel;

pub const DecodeOptions = struct {
    limits: Limits = .{},
    prefer_format: PixelFormat = .rgb8,
    apply_orientation: bool = true,
    alpha: AlphaMode = .composite,
    alpha_background: Rgb8 = .{ .r = 0, .g = 0, .b = 0 },
    color: ColorPolicy = .assume_srgb,
};

pub fn detectFormat(bytes: []const u8) ?Format {
    return sniff_mod.detectWithFallback(bytes);
}

pub fn decode(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    opts: DecodeOptions,
) !Image {
    _ = opts.color;

    try opts.limits.validateInput(bytes.len);

    const format = detectFormat(bytes) orelse return error.UnsupportedImageFormat;

    var decoded = switch (format) {
        .jpeg => try codecs.jpeg.decode(allocator, bytes, opts.limits, opts.apply_orientation),
        .png => try codecs.png.decode(allocator, bytes, opts.limits),
        .webp => try codecs.webp.decode(allocator, bytes, opts.limits),
        .pdf => try codecs.pdf.decode(allocator, bytes, opts.limits),
    };
    errdefer decoded.deinit(allocator);

    const converted = try convert_mod.convert(allocator, decoded, .{
        .format = opts.prefer_format,
        .alpha = opts.alpha,
        .alpha_background = opts.alpha_background,
        .limits = opts.limits,
    });

    decoded.deinit(allocator);
    return converted;
}

pub const convert = convert_mod.convert;
pub const smartResize = preprocess_mod.smartResize;
pub const toPlanarF32 = preprocess_mod.toPlanarF32;
pub const calculateVisionGrid = preprocess_mod.calculateVisionGrid;
pub const calculateMergedTokenCount = preprocess_mod.calculateMergedTokenCount;
pub const calculateTokenCountForImage = preprocess_mod.calculateTokenCountForImage;
pub const preprocessImage = preprocess_mod.preprocessImage;

pub fn toModelInput(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    spec: ModelInputSpec,
) !ModelBuffer {
    var decoded = try decode(allocator, bytes, .{
        .limits = spec.limits,
        .prefer_format = .rgb8,
        .apply_orientation = true,
        .alpha = .composite,
        .alpha_background = spec.pad_color,
    });
    defer decoded.deinit(allocator);

    var resized = try convert_mod.convert(allocator, decoded, .{
        .format = .rgb8,
        .resize = .{
            .out_w = spec.width,
            .out_h = spec.height,
            .fit = spec.fit,
            .filter = spec.filter,
            .pad_color = spec.pad_color,
        },
        .alpha = .composite,
        .alpha_background = spec.pad_color,
        .limits = spec.limits,
    });
    defer resized.deinit(allocator);

    return model_input_mod.packModelInput(allocator, resized, spec);
}

pub const encode = encode_mod.encode;
pub const codecs_internal = codecs;
pub const pdf = codecs.pdf;

pub const TransformSpec = struct {
    limits: Limits = .{},
    resize: ?ResizeOptions = null,
    output_format: ?EncodeFormat = null,
    jpeg_quality: u8 = 85,
    pad_color: Rgb8 = .{ .r = 0, .g = 0, .b = 0 },
};

pub const TransformResult = struct {
    data: []u8,
    width: u32,
    height: u32,
    encode_format: EncodeFormat,
};

/// Decode, optionally resize, and re-encode an image in a single pipeline.
/// The input format is auto-detected. Caller owns the returned data.
pub fn transformImage(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    input_format: Format,
    spec: TransformSpec,
) !TransformResult {
    var decoded = try decode(allocator, bytes, .{
        .limits = spec.limits,
        .prefer_format = .rgb8,
        .apply_orientation = true,
        .alpha = .composite,
        .alpha_background = spec.pad_color,
    });
    defer decoded.deinit(allocator);

    var final_img = decoded;
    var final_owned = false;
    if (spec.resize) |resize_opts| {
        final_img = try convert_mod.convert(allocator, decoded, .{
            .format = .rgb8,
            .resize = resize_opts,
            .alpha = .composite,
            .alpha_background = spec.pad_color,
            .limits = spec.limits,
        });
        final_owned = true;
    }
    defer if (final_owned) final_img.deinit(allocator);

    const out_format = spec.output_format orelse switch (input_format) {
        .jpeg => EncodeFormat.jpeg,
        .png, .webp, .pdf => EncodeFormat.png,
    };
    const encoded = try encode_mod.encode(allocator, final_img, .{
        .format = out_format,
        .jpeg_quality = spec.jpeg_quality,
    });

    return .{
        .data = encoded,
        .width = final_img.width,
        .height = final_img.height,
        .encode_format = out_format,
    };
}

/// Render a single PDF page, optionally resize, and encode to output format.
/// Combines renderPage → convert(rgba→rgb + resize) → encode in one pipeline.
/// Caller owns the returned data.
pub fn transformPdfPage(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    page_index: u32,
    dpi: u32,
    spec: TransformSpec,
) !TransformResult {
    var decoded = try codecs.pdf.renderPage(
        allocator,
        bytes,
        page_index,
        if (dpi == 0) 150 else dpi,
        spec.limits,
    );
    defer decoded.deinit(allocator);

    // Convert RGBA→RGB + optional resize.
    var final_img = try convert_mod.convert(allocator, decoded, .{
        .format = .rgb8,
        .resize = spec.resize,
        .alpha = .composite,
        .alpha_background = spec.pad_color,
        .limits = spec.limits,
    });
    defer final_img.deinit(allocator);

    const out_format = spec.output_format orelse EncodeFormat.png;
    const encoded = try encode_mod.encode(allocator, final_img, .{
        .format = out_format,
        .jpeg_quality = spec.jpeg_quality,
    });

    return .{
        .data = encoded,
        .width = final_img.width,
        .height = final_img.height,
        .encode_format = out_format,
    };
}
