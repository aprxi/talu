use anyhow::{anyhow, bail, Context, Result};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::{FileArgs, FileFitArg, FileFormatArg};

fn is_stdin(path: &Option<PathBuf>) -> bool {
    match path {
        None => true,
        Some(p) => p.as_os_str() == "-",
    }
}

pub(super) fn cmd_file(args: FileArgs) -> Result<()> {
    let from_stdin = is_stdin(&args.path);
    let bytes = if from_stdin {
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .context("Error: failed to read from stdin")?;
        if buf.is_empty() {
            bail!("Error: no data received on stdin.");
        }
        buf
    } else {
        let input_path = args.path.as_ref().unwrap();
        fs::read(input_path)
            .with_context(|| format!("Error: failed to read {}", input_path.display()))?
    };

    let transform_requested = args.resize.is_some()
        || args.fit.is_some()
        || args.output.is_some()
        || args.format.is_some()
        || args.quality.is_some();

    if !transform_requested {
        let info = talu::file::inspect_bytes(&bytes)?;
        if from_stdin {
            print_file_info(Path::new("<stdin>"), &info);
        } else {
            print_file_info(args.path.as_deref().unwrap(), &info);
        }
        if info.kind == talu::file::FileKind::Document && info.mime == "application/pdf" {
            if let Ok(pages) = talu::file::pdf_page_count(&bytes) {
                println!("Pages: {pages}");
            }
        }
        return Ok(());
    }

    if from_stdin && args.output.is_none() {
        bail!("Error: --output is required when reading from stdin.");
    }

    let resize = if let Some(spec) = args.resize.as_deref() {
        let (width, height) = parse_resize(spec)?;
        Some(talu::file::ResizeOptions {
            width,
            height,
            fit: args.fit.unwrap_or(FileFitArg::Contain).into(),
            filter: talu::file::ResizeFilter::Bicubic,
        })
    } else {
        None
    };

    let input_path_ref = args.path.as_deref().unwrap_or_else(|| Path::new("stdin"));
    let inferred_format = infer_output_format(args.format, args.output.as_deref(), input_path_ref)?;
    let output_path = resolve_output_path(args.output, input_path_ref, inferred_format)?;
    if !from_stdin {
        ensure_not_in_place(input_path_ref, &output_path)?;
    }

    let opts = talu::file::TransformOptions {
        resize,
        output_format: Some(inferred_format),
        jpeg_quality: args.quality.unwrap_or(85),
        pad_rgb: (0, 0, 0),
        limits: talu::file::Limits::default(),
    };

    // Check if input is PDF — if so, render each page as a separate image.
    let info = talu::file::inspect_bytes(&bytes)?;
    if info.kind == talu::file::FileKind::Document && info.mime == "application/pdf" {
        let page_count = talu::file::pdf_page_count(&bytes)?;
        if page_count == 0 {
            bail!("Error: PDF has no pages.");
        }
        for page in 0..page_count {
            let result = talu::file::pdf_transform_page(&bytes, page, 150, opts)?;
            let page_path = numbered_output_path(&output_path, page, page_count);
            fs::write(&page_path, &result.bytes)
                .with_context(|| format!("Error: failed to write {}", page_path.display()))?;
            println!("{}", page_path.display());
        }
    } else {
        let result = talu::file::transform_image_bytes(&bytes, opts)?;
        fs::write(&output_path, &result.bytes)
            .with_context(|| format!("Error: failed to write {}", output_path.display()))?;
        println!("{}", output_path.display());
    }

    Ok(())
}

fn print_file_info(path: &Path, info: &talu::file::FileInfo) {
    println!("Path: {}", path.display());
    println!(
        "Kind: {}",
        match info.kind {
            talu::file::FileKind::Binary => "binary",
            talu::file::FileKind::Image => "image",
            talu::file::FileKind::Document => "document",
            talu::file::FileKind::Audio => "audio",
            talu::file::FileKind::Video => "video",
            talu::file::FileKind::Text => "text",
        }
    );

    if !info.mime.is_empty() {
        println!("MIME: {}", info.mime);
    }
    if !info.description.is_empty() {
        println!("Description: {}", info.description);
    }

    if let Some(image) = info.image {
        println!(
            "Format: {}",
            match image.format {
                talu::file::ImageFormat::Jpeg => "jpeg",
                talu::file::ImageFormat::Png => "png",
                talu::file::ImageFormat::Webp => "webp",
                talu::file::ImageFormat::Unknown => "unknown",
            }
        );
        println!("Size: {}x{}", image.width, image.height);
        if image.exif_orientation > 0 {
            println!("EXIF Orientation: {}", image.exif_orientation);
        }
    }
}

fn parse_resize(value: &str) -> Result<(u32, u32)> {
    let (w, h) = value
        .split_once(['x', 'X'])
        .ok_or_else(|| anyhow!("Error: --resize must be in WxH format, e.g. 512x512"))?;
    let width: u32 = w
        .parse()
        .map_err(|_| anyhow!("Error: invalid resize width '{}'", w))?;
    let height: u32 = h
        .parse()
        .map_err(|_| anyhow!("Error: invalid resize height '{}'", h))?;
    if width == 0 || height == 0 {
        bail!("Error: resize dimensions must be > 0.");
    }
    Ok((width, height))
}

fn infer_output_format(
    explicit: Option<FileFormatArg>,
    output: Option<&Path>,
    input: &Path,
) -> Result<talu::file::OutputFormat> {
    if let Some(fmt) = explicit {
        if let Some(out_path) = output {
            if let Some(ext_fmt) = format_from_extension(out_path) {
                let fmt_value: talu::file::OutputFormat = fmt.into();
                if ext_fmt != fmt_value {
                    bail!(
                        "Error: --format does not match output file extension ({}).",
                        out_path.display()
                    );
                }
            }
        }
        return Ok(fmt.into());
    }

    if let Some(out_path) = output {
        if let Some(fmt) = format_from_extension(out_path) {
            return Ok(fmt);
        }
        bail!("Error: cannot infer output format from extension; pass --format jpeg|png.");
    }

    let input_ext = input
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    Ok(match input_ext.as_str() {
        "jpg" | "jpeg" => talu::file::OutputFormat::Jpeg,
        _ => talu::file::OutputFormat::Png,
    })
}

fn resolve_output_path(
    output: Option<PathBuf>,
    input: &Path,
    format: talu::file::OutputFormat,
) -> Result<PathBuf> {
    if let Some(path) = output {
        return Ok(path);
    }

    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("file");
    let ext = match format {
        talu::file::OutputFormat::Jpeg => "jpg",
        talu::file::OutputFormat::Png => "png",
    };

    let mut idx: usize = 0;
    loop {
        let name = if idx == 0 {
            format!("{stem}_processed.{ext}")
        } else {
            format!("{stem}_processed_{idx}.{ext}")
        };
        let candidate = parent.join(name);
        if !candidate.exists() {
            return Ok(candidate);
        }
        idx += 1;
    }
}

fn ensure_not_in_place(input: &Path, output: &Path) -> Result<()> {
    if input == output {
        bail!("Error: output path must differ from input path.");
    }
    Ok(())
}

/// For multi-page output, insert a page number before the extension.
/// Single-page PDFs write to `path` unchanged; multi-page PDFs produce
/// `stem_1.ext`, `stem_2.ext`, ... (1-indexed for user clarity).
fn numbered_output_path(base: &Path, page_index: u32, total_pages: u32) -> PathBuf {
    if total_pages <= 1 {
        return base.to_path_buf();
    }
    let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
    let ext = base.extension().and_then(|s| s.to_str()).unwrap_or("png");
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!("{stem}_{}.{ext}", page_index + 1))
}

fn format_from_extension(path: &Path) -> Option<talu::file::OutputFormat> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "jpg" | "jpeg" => Some(talu::file::OutputFormat::Jpeg),
        "png" => Some(talu::file::OutputFormat::Png),
        _ => None,
    }
}

impl From<FileFitArg> for talu::file::FitMode {
    fn from(value: FileFitArg) -> Self {
        match value {
            FileFitArg::Stretch => talu::file::FitMode::Stretch,
            FileFitArg::Contain => talu::file::FitMode::Contain,
            FileFitArg::Cover => talu::file::FitMode::Cover,
        }
    }
}

impl From<FileFormatArg> for talu::file::OutputFormat {
    fn from(value: FileFormatArg) -> Self {
        match value {
            FileFormatArg::Jpeg => talu::file::OutputFormat::Jpeg,
            FileFormatArg::Png => talu::file::OutputFormat::Png,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_resize_valid() {
        let (w, h) = parse_resize("512x256").expect("resize parse");
        assert_eq!(w, 512);
        assert_eq!(h, 256);
    }

    #[test]
    fn parse_resize_rejects_invalid() {
        assert!(parse_resize("512").is_err());
        assert!(parse_resize("0x10").is_err());
    }

    #[test]
    fn infer_format_from_path() {
        let input = Path::new("in.webp");
        let fmt = infer_output_format(None, None, input).expect("format");
        assert_eq!(fmt, talu::file::OutputFormat::Png);
    }

    #[test]
    fn numbered_output_single_page() {
        let base = Path::new("/tmp/out.png");
        assert_eq!(
            numbered_output_path(base, 0, 1),
            PathBuf::from("/tmp/out.png")
        );
    }

    #[test]
    fn numbered_output_multi_page() {
        let base = Path::new("/tmp/pages.jpg");
        assert_eq!(
            numbered_output_path(base, 0, 3),
            PathBuf::from("/tmp/pages_1.jpg")
        );
        assert_eq!(
            numbered_output_path(base, 1, 3),
            PathBuf::from("/tmp/pages_2.jpg")
        );
        assert_eq!(
            numbered_output_path(base, 2, 3),
            PathBuf::from("/tmp/pages_3.jpg")
        );
    }
}
