//! `inspect_bytes` safe wrapper tests.
//!
//! Validates: content-based MIME detection, image metadata extraction
//! (format, dimensions, EXIF orientation), kind classification
//! (Image, Document, Text, Audio, Video, Binary), and error paths.

use talu::file::{self, FileKind, ImageFormat};

use super::*;

// ---------------------------------------------------------------------------
// Image detection: PNG, JPEG, WebP
// ---------------------------------------------------------------------------

/// PNG bytes produce FileKind::Image with correct format and dimensions.
#[test]
fn inspect_png_returns_image_with_metadata() {
    let info = file::inspect_bytes(RED_PNG).expect("inspect_bytes failed");

    assert_eq!(info.kind, FileKind::Image);
    assert_eq!(info.mime, "image/png");

    let image = info
        .image
        .expect("image metadata should be present for PNG");
    assert_eq!(image.format, ImageFormat::Png);
    assert_eq!(image.width, 1);
    assert_eq!(image.height, 1);
    // PNG has no EXIF; Zig core returns 1 (normal) as default.
    assert_eq!(image.exif_orientation, 1);
}

/// JPEG bytes produce FileKind::Image with correct format and dimensions.
#[test]
fn inspect_jpeg_returns_image_with_metadata() {
    let info = file::inspect_bytes(RED_JPEG).expect("inspect_bytes failed");

    assert_eq!(info.kind, FileKind::Image);
    assert_eq!(info.mime, "image/jpeg");

    let image = info
        .image
        .expect("image metadata should be present for JPEG");
    assert_eq!(image.format, ImageFormat::Jpeg);
    assert_eq!(image.width, 1);
    assert_eq!(image.height, 1);
}

/// WebP bytes produce FileKind::Image with correct format and dimensions.
#[test]
fn inspect_webp_returns_image_with_metadata() {
    let info = file::inspect_bytes(RED_WEBP).expect("inspect_bytes failed");

    assert_eq!(info.kind, FileKind::Image);
    assert_eq!(info.mime, "image/webp");

    let image = info
        .image
        .expect("image metadata should be present for WebP");
    assert_eq!(image.format, ImageFormat::Webp);
    assert_eq!(image.width, 1);
    assert_eq!(image.height, 1);
}

/// Non-square JPEG reports correct width and height.
#[test]
fn inspect_non_square_image_reports_correct_dimensions() {
    let info = file::inspect_bytes(BLUE_JPEG_2X3).expect("inspect_bytes failed");

    assert_eq!(info.kind, FileKind::Image);
    let image = info.image.expect("image metadata");
    assert_eq!(image.format, ImageFormat::Jpeg);
    assert_eq!(image.width, 2);
    assert_eq!(image.height, 3);
}

/// JPEG with EXIF APP1 segment reports correct orientation value.
#[test]
fn inspect_jpeg_with_exif_reports_orientation() {
    let info = file::inspect_bytes(EXIF_ROTATE90).expect("inspect_bytes failed");

    assert_eq!(info.kind, FileKind::Image);
    let image = info.image.expect("image metadata");
    assert_eq!(image.format, ImageFormat::Jpeg);
    // EXIF orientation 6 = 90 degrees clockwise.
    assert_eq!(
        image.exif_orientation, 6,
        "expected EXIF orientation 6 (rotate 90 CW)"
    );
}

// ---------------------------------------------------------------------------
// Non-image detection: text and binary
// ---------------------------------------------------------------------------

/// ASCII text produces FileKind::Text with text/plain MIME.
#[test]
fn inspect_ascii_text_returns_text_kind() {
    let info = file::inspect_bytes(b"Hello, world!").expect("inspect_bytes failed");

    assert_eq!(info.kind, FileKind::Text);
    assert_eq!(info.mime, "text/plain");
    assert!(info.image.is_none());
    assert!(
        !info.description.is_empty(),
        "description should be non-empty"
    );
}

/// Binary data (sequential bytes with many non-printable chars) produces
/// FileKind::Binary with a non-text MIME.
#[test]
fn inspect_binary_data_returns_binary_kind() {
    let garbage: Vec<u8> = (0..128).collect();
    let info = file::inspect_bytes(&garbage).expect("inspect_bytes failed");

    assert_eq!(info.kind, FileKind::Binary);
    assert!(
        !info.mime.starts_with("text/"),
        "binary data should not have text/ MIME, got: {}",
        info.mime
    );
    assert!(info.image.is_none());
}

/// ZIP magic bytes produce FileKind::Binary with non-text MIME.
#[test]
fn inspect_zip_bytes_returns_binary_kind() {
    #[rustfmt::skip]
    let zip: &[u8] = &[
        0x50, 0x4B, 0x03, 0x04, // PK signature
        0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x61,
    ];
    let info = file::inspect_bytes(zip).expect("inspect_bytes failed");

    assert_eq!(info.kind, FileKind::Binary);
    assert!(info.image.is_none());
    assert!(
        !info.mime.starts_with("image/"),
        "ZIP should not be detected as image, got: {}",
        info.mime
    );
}

/// ELF magic bytes produce FileKind::Binary with non-text/non-image MIME.
#[test]
fn inspect_elf_bytes_returns_binary_kind() {
    #[rustfmt::skip]
    let elf: &[u8] = &[
        0x7F, 0x45, 0x4C, 0x46, // \x7fELF
        0x02, 0x01, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x3E, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    let info = file::inspect_bytes(elf).expect("inspect_bytes failed");

    assert_eq!(info.kind, FileKind::Binary);
    assert!(info.image.is_none());
    assert!(
        !info.mime.starts_with("text/") && !info.mime.starts_with("image/"),
        "ELF should not be text or image, got: {}",
        info.mime
    );
}

// ---------------------------------------------------------------------------
// Metadata contract: description is always populated
// ---------------------------------------------------------------------------

/// Description is a non-empty string for all inspectable file types.
#[test]
fn inspect_always_produces_non_empty_description() {
    let garbage: Vec<u8> = (0..128).collect();
    let inputs: &[(&[u8], &str)] = &[
        (RED_PNG, "PNG"),
        (RED_JPEG, "JPEG"),
        (RED_WEBP, "WebP"),
        (b"Hello, world!", "text"),
        (&garbage, "binary"),
    ];
    for (bytes, label) in inputs {
        let info = file::inspect_bytes(bytes).unwrap_or_else(|e| {
            panic!("inspect_bytes failed for {label}: {e}");
        });
        assert!(
            !info.description.is_empty(),
            "description should be non-empty for {label}"
        );
    }
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

/// Empty bytes produce an error (not a panic).
#[test]
fn inspect_empty_bytes_returns_error() {
    let result = file::inspect_bytes(b"");
    assert!(result.is_err(), "empty bytes should produce an error");
}

/// Truncated PNG (magic bytes only, no IHDR) still succeeds but may
/// report zero dimensions — verifies the wrapper handles partial data
/// without panicking.
#[test]
fn inspect_truncated_png_does_not_panic() {
    // First 8 bytes of a PNG (signature only, no IHDR chunk).
    let truncated = &RED_PNG[..8.min(RED_PNG.len())];
    let result = file::inspect_bytes(truncated);
    // Either succeeds with partial info or returns an error — both are acceptable.
    // The critical check is that it does not panic or segfault.
    if let Ok(info) = result {
        // If it succeeds, kind should still be Image (magic bytes detected).
        assert_eq!(info.kind, FileKind::Image);
    }
}
