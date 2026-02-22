//! `transform_image_bytes` safe wrapper tests.
//!
//! Validates: identity re-encode, format conversion, resize with all fit
//! modes, quality parameter, and error paths.

use talu::file::{
    self, FitMode, ImageFormat, Limits, OutputFormat, ResizeOptions, TransformOptions,
};

use super::*;

// ---------------------------------------------------------------------------
// Identity transform: re-encode without options
// ---------------------------------------------------------------------------

/// PNG identity re-encode produces valid PNG output with correct dimensions and format.
#[test]
fn transform_png_identity_produces_valid_png() {
    let opts = TransformOptions::default();
    let result = file::transform_image_bytes(RED_PNG, opts).expect("transform failed");

    assert!(!result.bytes.is_empty());
    let (w, h) = png_dimensions(&result.bytes);
    assert_eq!((w, h), (1, 1));

    let image = result.image.expect("output should have image metadata");
    assert_eq!(image.format, ImageFormat::Png);
    assert_eq!(image.width, 1);
    assert_eq!(image.height, 1);
}

/// JPEG identity re-encode produces valid JPEG output with magic bytes and metadata.
#[test]
fn transform_jpeg_identity_produces_valid_jpeg() {
    let opts = TransformOptions::default();
    let result = file::transform_image_bytes(RED_JPEG, opts).expect("transform failed");

    assert!(
        result.bytes.len() >= 3
            && result.bytes[0] == 0xFF
            && result.bytes[1] == 0xD8
            && result.bytes[2] == 0xFF,
        "output should start with JPEG magic bytes, got {:02X?}",
        &result.bytes[..result.bytes.len().min(4)]
    );
    let image = result.image.expect("output should have image metadata");
    assert_eq!(image.format, ImageFormat::Jpeg);
    assert_eq!(image.width, 1);
    assert_eq!(image.height, 1);
}

// ---------------------------------------------------------------------------
// Format conversion
// ---------------------------------------------------------------------------

/// PNG to JPEG conversion produces valid JPEG with correct magic bytes.
#[test]
fn transform_png_to_jpeg_produces_valid_jpeg() {
    let opts = TransformOptions {
        output_format: Some(OutputFormat::Jpeg),
        jpeg_quality: 90,
        ..Default::default()
    };
    let result = file::transform_image_bytes(RED_PNG, opts).expect("transform failed");

    assert!(
        result.bytes.len() >= 3
            && result.bytes[0] == 0xFF
            && result.bytes[1] == 0xD8
            && result.bytes[2] == 0xFF,
        "output should start with JPEG magic, got {:02X?}",
        &result.bytes[..result.bytes.len().min(4)]
    );
    let image = result.image.expect("image metadata");
    assert_eq!(image.format, ImageFormat::Jpeg);
}

/// JPEG to PNG conversion produces valid PNG with correct dimensions.
#[test]
fn transform_jpeg_to_png_produces_valid_png() {
    let opts = TransformOptions {
        output_format: Some(OutputFormat::Png),
        ..Default::default()
    };
    let result = file::transform_image_bytes(RED_JPEG, opts).expect("transform failed");

    let (w, h) = png_dimensions(&result.bytes);
    assert_eq!((w, h), (1, 1));
    let image = result.image.expect("image metadata");
    assert_eq!(image.format, ImageFormat::Png);
}

/// WebP input can be decoded and re-encoded to PNG.
#[test]
fn transform_webp_to_png_produces_valid_png() {
    let opts = TransformOptions {
        output_format: Some(OutputFormat::Png),
        ..Default::default()
    };
    let result = file::transform_image_bytes(RED_WEBP, opts).expect("transform failed");

    let (w, h) = png_dimensions(&result.bytes);
    assert_eq!((w, h), (1, 1));
    let image = result.image.expect("image metadata");
    assert_eq!(image.format, ImageFormat::Png);
}

// ---------------------------------------------------------------------------
// Resize: all three fit modes
// ---------------------------------------------------------------------------

/// Cover mode produces exact target dimensions by cropping.
#[test]
fn transform_resize_cover_produces_exact_dimensions() {
    let opts = TransformOptions {
        resize: Some(ResizeOptions {
            width: 1,
            height: 1,
            fit: FitMode::Cover,
            filter: talu::file::ResizeFilter::Bicubic,
        }),
        output_format: Some(OutputFormat::Png),
        ..Default::default()
    };
    // 2x3 input → cover to 1x1 crops to exact dimensions.
    let result = file::transform_image_bytes(BLUE_JPEG_2X3, opts).expect("transform failed");

    let (w, h) = png_dimensions(&result.bytes);
    assert_eq!(
        (w, h),
        (1, 1),
        "cover mode should produce exact target dimensions"
    );
}

/// Contain mode produces exact target dimensions with padding.
#[test]
fn transform_resize_contain_pads_to_target() {
    let opts = TransformOptions {
        resize: Some(ResizeOptions {
            width: 10,
            height: 10,
            fit: FitMode::Contain,
            filter: talu::file::ResizeFilter::Bicubic,
        }),
        output_format: Some(OutputFormat::Png),
        ..Default::default()
    };
    // 2x3 input → contain into 10x10: scales to fit then pads to 10x10.
    let result = file::transform_image_bytes(BLUE_JPEG_2X3, opts).expect("transform failed");

    let (w, h) = png_dimensions(&result.bytes);
    assert_eq!(
        (w, h),
        (10, 10),
        "contain mode should produce target dimensions with padding"
    );
}

/// Stretch mode distorts to exact target dimensions without padding.
#[test]
fn transform_resize_stretch_produces_exact_dimensions() {
    let opts = TransformOptions {
        resize: Some(ResizeOptions {
            width: 10,
            height: 5,
            fit: FitMode::Stretch,
            filter: talu::file::ResizeFilter::Bicubic,
        }),
        output_format: Some(OutputFormat::Png),
        ..Default::default()
    };
    let result = file::transform_image_bytes(BLUE_JPEG_2X3, opts).expect("transform failed");

    let (w, h) = png_dimensions(&result.bytes);
    assert_eq!(
        (w, h),
        (10, 5),
        "stretch mode should produce exact target dimensions"
    );
}

// ---------------------------------------------------------------------------
// Output metadata
// ---------------------------------------------------------------------------

/// Transform result includes image metadata with correct output dimensions.
#[test]
fn transform_result_contains_output_metadata() {
    let opts = TransformOptions {
        resize: Some(ResizeOptions {
            width: 4,
            height: 6,
            fit: FitMode::Stretch,
            filter: talu::file::ResizeFilter::Bicubic,
        }),
        output_format: Some(OutputFormat::Png),
        ..Default::default()
    };
    let result = file::transform_image_bytes(BLUE_JPEG_2X3, opts).expect("transform failed");

    let image = result
        .image
        .expect("transform result should include image metadata");
    assert_eq!(image.format, ImageFormat::Png);
    assert_eq!(image.width, 4);
    assert_eq!(image.height, 6);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

/// Empty bytes produce an error.
#[test]
fn transform_empty_bytes_returns_error() {
    let opts = TransformOptions::default();
    let result = file::transform_image_bytes(b"", opts);
    assert!(result.is_err(), "empty bytes should produce an error");
}

/// Non-image bytes produce an error.
#[test]
fn transform_non_image_bytes_returns_error() {
    let opts = TransformOptions::default();
    let result = file::transform_image_bytes(b"this is not an image", opts);
    assert!(result.is_err(), "non-image bytes should produce an error");
}

// ---------------------------------------------------------------------------
// JPEG quality parameter
// ---------------------------------------------------------------------------

/// Lower JPEG quality produces smaller output than higher quality.
#[test]
fn transform_jpeg_quality_affects_output_size() {
    let low_q = TransformOptions {
        output_format: Some(OutputFormat::Jpeg),
        jpeg_quality: 1,
        ..Default::default()
    };
    let high_q = TransformOptions {
        output_format: Some(OutputFormat::Jpeg),
        jpeg_quality: 100,
        ..Default::default()
    };
    // Use the 2x3 image (more pixels) so quality difference is measurable.
    let low = file::transform_image_bytes(BLUE_JPEG_2X3, low_q).expect("low quality failed");
    let high = file::transform_image_bytes(BLUE_JPEG_2X3, high_q).expect("high quality failed");

    assert!(
        low.bytes.len() < high.bytes.len(),
        "quality=1 ({} bytes) should produce smaller output than quality=100 ({} bytes)",
        low.bytes.len(),
        high.bytes.len()
    );
}

// ---------------------------------------------------------------------------
// Resize filter variants
// ---------------------------------------------------------------------------

/// Nearest-neighbor filter produces valid output with correct dimensions.
#[test]
fn transform_resize_nearest_produces_correct_dimensions() {
    let opts = TransformOptions {
        resize: Some(ResizeOptions {
            width: 4,
            height: 4,
            fit: FitMode::Stretch,
            filter: talu::file::ResizeFilter::Nearest,
        }),
        output_format: Some(OutputFormat::Png),
        ..Default::default()
    };
    let result = file::transform_image_bytes(BLUE_JPEG_2X3, opts).expect("transform failed");

    let (w, h) = png_dimensions(&result.bytes);
    assert_eq!(
        (w, h),
        (4, 4),
        "nearest filter should produce exact target dimensions"
    );
}

/// Bilinear filter produces valid output with correct dimensions.
#[test]
fn transform_resize_bilinear_produces_correct_dimensions() {
    let opts = TransformOptions {
        resize: Some(ResizeOptions {
            width: 4,
            height: 4,
            fit: FitMode::Stretch,
            filter: talu::file::ResizeFilter::Bilinear,
        }),
        output_format: Some(OutputFormat::Png),
        ..Default::default()
    };
    let result = file::transform_image_bytes(BLUE_JPEG_2X3, opts).expect("transform failed");

    let (w, h) = png_dimensions(&result.bytes);
    assert_eq!(
        (w, h),
        (4, 4),
        "bilinear filter should produce exact target dimensions"
    );
}

// ---------------------------------------------------------------------------
// Contain mode with custom padding color
// ---------------------------------------------------------------------------

/// Non-default pad_rgb color succeeds without error.
#[test]
fn transform_contain_with_custom_pad_color_succeeds() {
    let opts = TransformOptions {
        resize: Some(ResizeOptions {
            width: 10,
            height: 10,
            fit: FitMode::Contain,
            filter: talu::file::ResizeFilter::Bicubic,
        }),
        output_format: Some(OutputFormat::Png),
        pad_rgb: (255, 0, 0), // red padding instead of default black
        ..Default::default()
    };
    let result = file::transform_image_bytes(BLUE_JPEG_2X3, opts).expect("transform failed");

    let (w, h) = png_dimensions(&result.bytes);
    assert_eq!((w, h), (10, 10));
}

// ---------------------------------------------------------------------------
// Limits enforcement
// ---------------------------------------------------------------------------

/// max_input_bytes rejects input that exceeds the limit.
#[test]
fn transform_max_input_bytes_rejects_oversized_input() {
    let opts = TransformOptions {
        limits: Limits {
            max_input_bytes: Some(1), // 1 byte — any real image exceeds this
            ..Default::default()
        },
        ..Default::default()
    };
    let result = file::transform_image_bytes(RED_PNG, opts);
    assert!(
        result.is_err(),
        "should reject input exceeding max_input_bytes"
    );
}
