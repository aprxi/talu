//! File inspect/transform C-API tests.
//!
//! Validates the safe Rust wrappers (`talu::file`) over the Zig core's
//! `talu_file_inspect` and `talu_file_transform`.

mod inspect;
mod transform;

// Test images compiled into the test binary.
const RED_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/1x1_red.png"
));
const RED_JPEG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/1x1_red.jpg"
));
const RED_WEBP: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/1x1_red.webp"
));
const BLUE_JPEG_2X3: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/2x3_blue.jpg"
));
const EXIF_ROTATE90: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../core/tests/image/corpus/exif_rotate90.jpg"
));

/// Extract width and height from raw PNG bytes by reading the IHDR chunk.
fn png_dimensions(data: &[u8]) -> (u32, u32) {
    assert!(data.len() >= 24, "PNG too small for IHDR");
    assert_eq!(&data[0..8], b"\x89PNG\r\n\x1a\n", "not a valid PNG");
    let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
    let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);
    (width, height)
}
