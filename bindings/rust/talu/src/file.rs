//! Safe wrappers for file inspect/transform APIs.

use crate::error::error_from_last_or;
use crate::Result;
use std::ffi::c_void;

/// File classification.
///
/// - `Image` — raster image (JPEG, PNG, WebP). The file IS pixels;
///   intrinsic width/height/orientation are in `ImageInfo`.
/// - `Document` — rendered format (PDF, future: DOCX). The file DESCRIBES
///   content that becomes pixels when rendered. No intrinsic dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Unknown,
    Image,
    Document,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Unknown,
    Jpeg,
    Png,
    Webp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Jpeg,
    Png,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMode {
    Stretch,
    Contain,
    Cover,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilter {
    Nearest,
    Bilinear,
    Bicubic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageInfo {
    pub format: ImageFormat,
    pub width: u32,
    pub height: u32,
    pub exif_orientation: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileInfo {
    pub kind: FileKind,
    pub mime: String,
    pub description: String,
    pub image: Option<ImageInfo>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeOptions {
    pub width: u32,
    pub height: u32,
    pub fit: FitMode,
    pub filter: ResizeFilter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Limits {
    pub max_input_bytes: Option<usize>,
    pub max_dimension: Option<u32>,
    pub max_pixels: Option<u64>,
    pub max_output_bytes: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformOptions {
    pub resize: Option<ResizeOptions>,
    pub output_format: Option<OutputFormat>,
    pub jpeg_quality: u8,
    pub pad_rgb: (u8, u8, u8),
    pub limits: Limits,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            resize: None,
            output_format: None,
            jpeg_quality: 85,
            pad_rgb: (0, 0, 0),
            limits: Limits::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformResult {
    pub bytes: Vec<u8>,
    pub image: Option<ImageInfo>,
}

pub fn inspect_bytes(bytes: &[u8]) -> Result<FileInfo> {
    if bytes.is_empty() {
        return Err(crate::error::Error::generic("bytes cannot be empty"));
    }

    let mut c_info = talu_sys::TaluFileInfo::default();
    let mut c_image = talu_sys::TaluImageInfo::default();
    // SAFETY: bytes pointer/len are valid for the call; out-structs are valid.
    let rc = unsafe {
        talu_sys::talu_file_inspect(bytes.as_ptr(), bytes.len(), &mut c_info, &mut c_image)
    };
    if rc != 0 {
        return Err(error_from_last_or("Failed to inspect file"));
    }

    let info = file_info_from_c(&c_info, &c_image);

    // SAFETY: c_info was initialized by talu_file_inspect and may own heap strings.
    unsafe { talu_sys::talu_file_info_free(&mut c_info) };

    Ok(info)
}

pub fn transform_image_bytes(bytes: &[u8], opts: TransformOptions) -> Result<TransformResult> {
    if bytes.is_empty() {
        return Err(crate::error::Error::generic("bytes cannot be empty"));
    }

    let mut c_out_ptr: *const u8 = std::ptr::null();
    let mut c_out_len: usize = 0;
    let mut c_image = talu_sys::TaluImageInfo::default();
    let c_opts = to_c_transform_options(opts);

    // SAFETY: bytes pointer/len are valid; output pointers are valid writable out-params.
    let rc = unsafe {
        talu_sys::talu_file_transform(
            bytes.as_ptr(),
            bytes.len(),
            &c_opts,
            (&mut c_out_ptr as *mut *const u8).cast::<c_void>(),
            (&mut c_out_len as *mut usize).cast::<c_void>(),
            &mut c_image,
        )
    };
    if rc != 0 {
        return Err(error_from_last_or("Failed to transform file"));
    }

    let out_bytes = if c_out_ptr.is_null() || c_out_len == 0 {
        Vec::new()
    } else {
        // SAFETY: pointer/len were produced by the C API and are valid for copy.
        let slice = unsafe { std::slice::from_raw_parts(c_out_ptr, c_out_len) };
        slice.to_vec()
    };

    // SAFETY: out buffer was allocated by C API and must be freed once copied.
    unsafe { talu_sys::talu_file_bytes_free(c_out_ptr, c_out_len) };

    Ok(TransformResult {
        bytes: out_bytes,
        image: image_info_from_c(&c_image),
    })
}

fn file_info_from_c(c_info: &talu_sys::TaluFileInfo, c_image: &talu_sys::TaluImageInfo) -> FileInfo {
    let mime = copy_c_bytes(c_info.mime_ptr, c_info.mime_len);
    let description = copy_c_bytes(c_info.description_ptr, c_info.description_len);
    let kind = file_kind_from_c(c_info.kind);

    FileInfo {
        kind,
        mime,
        description,
        image: image_info_from_c(c_image),
    }
}

fn image_info_from_c(c_image: &talu_sys::TaluImageInfo) -> Option<ImageInfo> {
    if c_image.format == 0 && c_image.width == 0 && c_image.height == 0 {
        return None;
    }
    Some(ImageInfo {
        format: image_format_from_c(c_image.format),
        width: c_image.width,
        height: c_image.height,
        exif_orientation: c_image.orientation,
    })
}

fn copy_c_bytes(ptr: *const u8, len: usize) -> String {
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    // SAFETY: ptr/len come from the C API and are valid until free call.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    String::from_utf8_lossy(bytes).into_owned()
}

fn file_kind_from_c(v: i32) -> FileKind {
    match v {
        1 => FileKind::Image,
        2 => FileKind::Document,
        _ => FileKind::Unknown,
    }
}

fn image_format_from_c(v: i32) -> ImageFormat {
    match v {
        1 => ImageFormat::Jpeg,
        2 => ImageFormat::Png,
        3 => ImageFormat::Webp,
        _ => ImageFormat::Unknown,
    }
}

fn output_format_to_c(v: Option<OutputFormat>) -> i32 {
    match v {
        None => 0,
        Some(OutputFormat::Jpeg) => 1,
        Some(OutputFormat::Png) => 2,
    }
}

fn fit_mode_to_c(v: FitMode) -> i32 {
    match v {
        FitMode::Stretch => 0,
        FitMode::Contain => 1,
        FitMode::Cover => 2,
    }
}

fn resize_filter_to_c(v: ResizeFilter) -> i32 {
    match v {
        ResizeFilter::Nearest => 0,
        ResizeFilter::Bilinear => 1,
        ResizeFilter::Bicubic => 2,
    }
}

fn to_c_transform_options(opts: TransformOptions) -> talu_sys::TaluFileTransformOptions {
    let mut out = talu_sys::TaluFileTransformOptions::default();
    out.output_format = output_format_to_c(opts.output_format);
    out.jpeg_quality = opts.jpeg_quality;
    out.pad_r = opts.pad_rgb.0;
    out.pad_g = opts.pad_rgb.1;
    out.pad_b = opts.pad_rgb.2;
    out.max_input_bytes = opts.limits.max_input_bytes.unwrap_or(0);
    out.max_dimension = opts.limits.max_dimension.unwrap_or(0);
    out.max_pixels = opts.limits.max_pixels.unwrap_or(0);
    out.max_output_bytes = opts.limits.max_output_bytes.unwrap_or(0);

    if let Some(resize) = opts.resize {
        out.resize_enabled = 1;
        out.out_w = resize.width;
        out.out_h = resize.height;
        out.fit_mode = fit_mode_to_c(resize.fit);
        out.filter = resize_filter_to_c(resize.filter);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_format_mapping() {
        assert_eq!(output_format_to_c(None), 0);
        assert_eq!(output_format_to_c(Some(OutputFormat::Jpeg)), 1);
        assert_eq!(output_format_to_c(Some(OutputFormat::Png)), 2);
    }

    #[test]
    fn file_kind_mapping() {
        assert_eq!(file_kind_from_c(0), FileKind::Unknown);
        assert_eq!(file_kind_from_c(1), FileKind::Image);
        assert_eq!(file_kind_from_c(2), FileKind::Document);
    }
}
