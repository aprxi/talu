//! PDF page renderer using PDFium.

const std = @import("std");
const pixel = @import("../pixel.zig");
const limits_mod = @import("../limits.zig");

// PDFium opaque handle types
const FPDF_DOCUMENT = ?*anyopaque;
const FPDF_PAGE = ?*anyopaque;
const FPDF_BITMAP = ?*anyopaque;

// Bitmap format constants
const FPDFBitmap_BGRA: c_int = 4;

// Render flags
const FPDF_ANNOT: c_int = 0x01;
const FPDF_PRINTING: c_int = 0x800;
const FPDF_REVERSE_BYTE_ORDER: c_int = 0x10;

extern fn FPDF_InitLibrary() void;
extern fn FPDF_DestroyLibrary() void;
extern fn FPDF_LoadMemDocument(data_buf: ?*const anyopaque, size: c_int, password: ?[*:0]const u8) FPDF_DOCUMENT;
extern fn FPDF_GetPageCount(document: FPDF_DOCUMENT) c_int;
extern fn FPDF_LoadPage(document: FPDF_DOCUMENT, page_index: c_int) FPDF_PAGE;
extern fn FPDF_GetPageWidthF(page: FPDF_PAGE) f32;
extern fn FPDF_GetPageHeightF(page: FPDF_PAGE) f32;
extern fn FPDF_ClosePage(page: FPDF_PAGE) void;
extern fn FPDF_CloseDocument(document: FPDF_DOCUMENT) void;

extern fn FPDFBitmap_Create(width: c_int, height: c_int, alpha: c_int) FPDF_BITMAP;
extern fn FPDFBitmap_CreateEx(width: c_int, height: c_int, format: c_int, first_scan: ?*anyopaque, stride: c_int) FPDF_BITMAP;
extern fn FPDFBitmap_FillRect(bitmap: FPDF_BITMAP, left: c_int, top: c_int, width: c_int, height: c_int, color: c_ulong) void;
extern fn FPDFBitmap_GetBuffer(bitmap: FPDF_BITMAP) ?[*]u8;
extern fn FPDFBitmap_GetStride(bitmap: FPDF_BITMAP) c_int;
extern fn FPDFBitmap_Destroy(bitmap: FPDF_BITMAP) void;
extern fn FPDF_RenderPageBitmap(bitmap: FPDF_BITMAP, page: FPDF_PAGE, start_x: c_int, start_y: c_int, size_x: c_int, size_y: c_int, rotate: c_int, flags: c_int) void;

var init_done: bool = false;
var init_mutex: std.Thread.Mutex = .{};

fn ensureInit() void {
    init_mutex.lock();
    defer init_mutex.unlock();
    if (!init_done) {
        FPDF_InitLibrary();
        init_done = true;
    }
}

pub const PageDimensions = struct {
    width_points: f32,
    height_points: f32,
};

pub fn pageCount(bytes: []const u8) !u32 {
    ensureInit();

    const doc = FPDF_LoadMemDocument(bytes.ptr, @intCast(bytes.len), null);
    if (doc == null) return error.PdfLoadFailed;
    defer FPDF_CloseDocument(doc);

    const count = FPDF_GetPageCount(doc);
    if (count <= 0) return error.PdfNoPages;
    return @intCast(count);
}

pub fn pageDimensions(bytes: []const u8, page_index: u32) !PageDimensions {
    ensureInit();

    const doc = FPDF_LoadMemDocument(bytes.ptr, @intCast(bytes.len), null);
    if (doc == null) return error.PdfLoadFailed;
    defer FPDF_CloseDocument(doc);

    const page = FPDF_LoadPage(doc, @intCast(page_index));
    if (page == null) return error.PdfPageLoadFailed;
    defer FPDF_ClosePage(page);

    return .{
        .width_points = FPDF_GetPageWidthF(page),
        .height_points = FPDF_GetPageHeightF(page),
    };
}

/// Render a PDF page to an RGBA8 image.
/// DPI controls resolution: 72 DPI = 1:1 with PDF points.
pub fn renderPage(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    page_index: u32,
    dpi: u32,
    lim: limits_mod.Limits,
) !pixel.Image {
    ensureInit();

    const doc = FPDF_LoadMemDocument(bytes.ptr, @intCast(bytes.len), null);
    if (doc == null) return error.PdfLoadFailed;
    defer FPDF_CloseDocument(doc);

    const page = FPDF_LoadPage(doc, @intCast(page_index));
    if (page == null) return error.PdfPageLoadFailed;
    defer FPDF_ClosePage(page);

    const page_w = FPDF_GetPageWidthF(page);
    const page_h = FPDF_GetPageHeightF(page);
    if (page_w <= 0 or page_h <= 0) return error.InvalidImageDimensions;

    const scale: f32 = @as(f32, @floatFromInt(dpi)) / 72.0;
    const px_w: u32 = @intFromFloat(@ceil(page_w * scale));
    const px_h: u32 = @intFromFloat(@ceil(page_h * scale));

    if (px_w == 0 or px_h == 0) return error.InvalidImageDimensions;
    try lim.validateDims(px_w, px_h);

    const stride = px_w * 4;
    const buf_size = @as(usize, stride) * @as(usize, px_h);
    const buf = try allocator.alloc(u8, buf_size);
    errdefer allocator.free(buf);

    // Create bitmap backed by our buffer (BGRA format with REVERSE_BYTE_ORDER = RGBA)
    const bitmap = FPDFBitmap_CreateEx(
        @intCast(px_w),
        @intCast(px_h),
        FPDFBitmap_BGRA,
        buf.ptr,
        @intCast(stride),
    );
    if (bitmap == null) {
        allocator.free(buf);
        return error.PdfRenderFailed;
    }
    // Don't destroy — we own the buffer, and FPDFBitmap_CreateEx with external
    // buffer means FPDFBitmap_Destroy only frees the handle, not the buffer.
    defer FPDFBitmap_Destroy(bitmap);

    // Fill with white background
    FPDFBitmap_FillRect(bitmap, 0, 0, @intCast(px_w), @intCast(px_h), 0xFFFFFFFF);

    // Render with REVERSE_BYTE_ORDER to get RGBA instead of BGRA
    FPDF_RenderPageBitmap(
        bitmap,
        page,
        0,
        0,
        @intCast(px_w),
        @intCast(px_h),
        0, // no rotation
        FPDF_ANNOT | FPDF_PRINTING | FPDF_REVERSE_BYTE_ORDER,
    );

    return pixel.Image{
        .width = px_w,
        .height = px_h,
        .format = .rgba8,
        .data = buf,
        .stride = stride,
    };
}

/// Decode a PDF page as an image (default: page 0, 150 DPI).
pub fn decode(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    lim: limits_mod.Limits,
) !pixel.Image {
    return renderPage(allocator, bytes, 0, 150, lim);
}
