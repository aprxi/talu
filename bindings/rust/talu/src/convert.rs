//! Safe wrappers for talu model conversion/quantization.

use crate::error::error_from_last_or;
use crate::Result;
use std::ffi::{c_void, CStr, CString};

/// Quantization scheme for model conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Scheme {
    Q40 = 0,
    Q4KM = 1,
    Q5K = 2,
    Q6K = 3,
    Q80 = 4,
    F16 = 5,
    Gaf432 = 10,
    Gaf464 = 11,
    Gaf4128 = 12,
    Gaf832 = 13,
    Gaf864 = 14,
    Gaf8128 = 15,
    Fp8E4m3 = 20,
    Fp8E5m2 = 21,
    Mxfp4 = 22,
    Nvfp4 = 23,
}

impl Scheme {
    /// Parses a scheme name string to a Scheme enum.
    pub fn parse(name: &str) -> Option<Self> {
        let Ok(c_str) = CString::new(name) else {
            return None;
        };
        // SAFETY: c_str is a valid null-terminated string.
        let result = unsafe { talu_sys::talu_convert_parse_scheme(c_str.as_ptr()) };
        if result < 0 {
            return None;
        }
        Some(match result as u32 {
            0 => Scheme::Q40,
            1 => Scheme::Q4KM,
            2 => Scheme::Q5K,
            3 => Scheme::Q6K,
            4 => Scheme::Q80,
            5 => Scheme::F16,
            10 => Scheme::Gaf432,
            11 => Scheme::Gaf464,
            12 => Scheme::Gaf4128,
            13 => Scheme::Gaf832,
            14 => Scheme::Gaf864,
            15 => Scheme::Gaf8128,
            20 => Scheme::Fp8E4m3,
            21 => Scheme::Fp8E5m2,
            22 => Scheme::Mxfp4,
            23 => Scheme::Nvfp4,
            _ => return None,
        })
    }
}

/// Options for model conversion.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    /// Quantization scheme.
    pub scheme: Option<Scheme>,
    /// Force overwrite existing output.
    pub force: bool,
    /// Return model ID (org/model-suffix) instead of filesystem path.
    pub return_model_id: bool,
}

/// Progress action type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressAction {
    /// Adding a new progress line.
    Add,
    /// Updating an existing progress line.
    Update,
    /// Completing/removing a progress line.
    Complete,
}

impl From<talu_sys::ProgressAction> for ProgressAction {
    fn from(action: talu_sys::ProgressAction) -> Self {
        match action {
            talu_sys::ProgressAction::Add => ProgressAction::Add,
            talu_sys::ProgressAction::Update => ProgressAction::Update,
            talu_sys::ProgressAction::Complete => ProgressAction::Complete,
        }
    }
}

/// Progress update during conversion.
#[derive(Debug, Clone)]
pub struct ConvertProgress {
    /// The action for this update.
    pub action: ProgressAction,
    /// Line ID for multi-line progress.
    pub line_id: u8,
    /// Name/label of the current operation.
    pub label: String,
    /// Progress message.
    pub message: String,
    /// Current progress value.
    pub current: u64,
    /// Total expected value.
    pub total: u64,
}

/// Callback type for conversion progress updates.
pub type ConvertProgressCallback = Box<dyn FnMut(ConvertProgress) + Send>;

/// Result of a successful conversion.
#[derive(Debug, Clone)]
pub struct ConvertResult {
    /// Path to the output model.
    pub output_path: String,
}

/// Callback wrapper for C API progress callback.
struct ProgressContext {
    callback: ConvertProgressCallback,
}

extern "C" fn progress_callback_wrapper(
    update: *const talu_sys::ProgressUpdate,
    user_data: *mut c_void,
) {
    if update.is_null() || user_data.is_null() {
        return;
    }

    // SAFETY: user_data is a valid pointer to ProgressContext created by convert.
    let ctx = unsafe { &mut *(user_data as *mut ProgressContext) };

    // SAFETY: update is a valid pointer passed from C.
    let update_ref = unsafe { &*update };

    let label = if update_ref.label.is_null() {
        String::new()
    } else {
        // SAFETY: label is a valid C string from the C API.
        unsafe { CStr::from_ptr(update_ref.label) }
            .to_string_lossy()
            .into_owned()
    };

    let message = if update_ref.message.is_null() {
        String::new()
    } else {
        // SAFETY: message is a valid C string from the C API.
        unsafe { CStr::from_ptr(update_ref.message) }
            .to_string_lossy()
            .into_owned()
    };

    let progress = ConvertProgress {
        action: ProgressAction::from(update_ref.action),
        line_id: update_ref.line_id,
        label,
        message,
        current: update_ref.current,
        total: update_ref.total,
    };

    (ctx.callback)(progress);
}

/// Converts a model to a quantized format.
pub fn convert(
    model_path: &str,
    output_dir: &str,
    options: ConvertOptions,
    callback: Option<ConvertProgressCallback>,
) -> Result<ConvertResult> {
    let c_model = CString::new(model_path)?;
    let c_output = CString::new(output_dir)?;

    let mut ctx = callback.map(|cb| Box::new(ProgressContext { callback: cb }));
    let ctx_ptr = ctx
        .as_mut()
        .map(|c| c.as_mut() as *mut ProgressContext as *mut c_void)
        .unwrap_or(std::ptr::null_mut());

    let progress_cb = if ctx.is_some() {
        progress_callback_wrapper as *mut c_void
    } else {
        std::ptr::null_mut()
    };

    let mut c_options = talu_sys::ConvertOptions::default();
    if let Some(scheme) = options.scheme {
        c_options.scheme = talu_sys::Scheme::from(scheme as u32);
    }
    c_options.force = options.force;
    c_options.return_model_id = options.return_model_id;
    c_options.progress_callback = progress_cb;
    c_options.progress_user_data = ctx_ptr;

    // SAFETY: All pointers are valid, c_model and c_output are null-terminated.
    let result = unsafe { talu_sys::talu_convert(c_model.as_ptr(), c_output.as_ptr(), &c_options) };

    if !result.success {
        let msg = if !result.error_msg.is_null() {
            // SAFETY: error_msg is a valid C string from the C API.
            let s = unsafe { CStr::from_ptr(result.error_msg) }
                .to_string_lossy()
                .into_owned();
            // SAFETY: error_msg was allocated by talu and must be freed.
            unsafe { talu_sys::talu_convert_free_string(result.error_msg) };
            s
        } else {
            error_from_last_or("Conversion failed").to_string()
        };

        // Free output_path if allocated (even on error, it might be set)
        if !result.output_path.is_null() {
            // SAFETY: output_path was allocated by talu and must be freed.
            unsafe { talu_sys::talu_convert_free_string(result.output_path) };
        }

        return Err(crate::error::Error::generic(msg));
    }

    let output_path = if !result.output_path.is_null() {
        // SAFETY: output_path is a valid C string from the C API.
        let s = unsafe { CStr::from_ptr(result.output_path) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: output_path was allocated by talu and must be freed.
        unsafe { talu_sys::talu_convert_free_string(result.output_path) };
        s
    } else {
        String::new()
    };

    // Free error_msg if set (even on success)
    if !result.error_msg.is_null() {
        // SAFETY: error_msg was allocated by talu and must be freed.
        unsafe { talu_sys::talu_convert_free_string(result.error_msg) };
    }

    Ok(ConvertResult { output_path })
}
