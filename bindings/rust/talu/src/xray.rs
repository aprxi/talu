//! XRay trace capture helpers.

use crate::error::error_from_last_or;
use crate::Result;
use std::os::raw::c_void;

/// A single trace record from the execution trace.
#[derive(Debug, Clone)]
pub struct TraceRecord {
    /// Trace point name (e.g., "embed", "layer_q", "logits")
    pub point: String,
    /// Layer index (0xFFFF = not applicable)
    pub layer: u16,
    /// Token index
    pub token: u32,
    /// Position in sequence
    pub position: u32,
    /// Backend id (0=cpu, 1=metal, 2=cuda)
    pub backend: u8,
    /// Tensor shape
    pub shape: Vec<u32>,
    /// Data type
    pub dtype: u8,
    /// Kernel name that produced this tensor (e.g., "matmul_q4_cpu_avx512")
    pub kernel_name: Option<String>,
    /// Runtime-provided exact work counters.
    pub work_flops: u64,
    pub work_bytes: u64,
    /// Tensor statistics
    pub stats: TraceStats,
    /// Nanosecond timestamp (monotonic clock)
    pub timestamp_ns: i64,
}

/// Statistics for a traced tensor.
#[derive(Debug, Clone, Default)]
pub struct TraceStats {
    pub count: u64,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub rms: f32,
    pub nan_count: u32,
    pub inf_count: u32,
}

impl TraceRecord {
    /// Format as a display name (e.g., "layer_0.layer_q" or "embed")
    pub fn display_name(&self) -> String {
        if self.layer == 0xFFFF {
            self.point.clone()
        } else {
            format!("layer_{}.{}", self.layer, self.point)
        }
    }

    /// Get dtype name.
    pub fn dtype_name(&self) -> &'static str {
        match self.dtype {
            0 => "f32",
            4 => "f16",
            5 => "bf16",
            25 => "q4",
            26 => "q8",
            _ => "other",
        }
    }

    /// Format shape as string.
    pub fn shape_str(&self) -> String {
        let dims: Vec<String> = self.shape.iter().map(|d| d.to_string()).collect();
        format!("[{}]", dims.join(", "))
    }

    /// Get backend name.
    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            0 => "cpu",
            1 => "metal",
            2 => "cuda",
            _ => "other",
        }
    }
}

/// Convert a trace point ID to its name.
fn xray_point_name(point: u8) -> &'static str {
    let ptr = unsafe { talu_sys::talu_xray_point_name(point) };
    if ptr.is_null() {
        "unknown"
    } else {
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_str()
            .unwrap_or("unknown")
    }
}

/// Handle that enables XRay capture for tracing.
pub struct XrayCaptureHandle {
    handle: *mut c_void,
}

impl XrayCaptureHandle {
    /// Create a new capture handle with all trace points enabled (stats mode).
    pub fn new() -> Result<Self> {
        // Mode 1 = stats only, sample_count = 0
        let handle = unsafe { talu_sys::talu_xray_capture_create_all(1, 0) };
        if handle.is_null() {
            return Err(error_from_last_or("xray capture create failed"));
        }
        Ok(Self { handle })
    }

    /// Create a low-overhead capture handle for timing/table views.
    pub fn new_timing() -> Result<Self> {
        // Mode 0 = metadata/timing only, sample_count = 0
        let handle = unsafe { talu_sys::talu_xray_capture_create_all(0, 0) };
        if handle.is_null() {
            return Err(error_from_last_or("xray capture create failed"));
        }
        Ok(Self { handle })
    }

    /// Enable capture (installs the trace handler).
    pub fn enable(&self) {
        unsafe { talu_sys::talu_xray_capture_enable(self.handle) };
    }

    /// Disable capture (stop receiving trace emissions).
    pub fn disable(&self) {
        unsafe { talu_sys::talu_xray_capture_disable() };
    }

    /// Clear all captured data (keep configuration).
    pub fn clear(&self) {
        unsafe { talu_sys::talu_xray_capture_clear(self.handle) };
    }

    /// Get number of captured trace records.
    pub fn count(&self) -> usize {
        unsafe { talu_sys::talu_xray_capture_count(self.handle) }
    }

    /// Get all captured trace records.
    pub fn get_trace(&self) -> Vec<TraceRecord> {
        let count = self.count();
        let mut records = Vec::with_capacity(count);

        for i in 0..count {
            let mut info = talu_sys::CapturedTensorInfo::default();
            if !unsafe {
                talu_sys::talu_xray_get(
                    self.handle,
                    i,
                    &mut info as *mut talu_sys::CapturedTensorInfo,
                )
            } {
                continue;
            }

            let point_name = xray_point_name(info.point);
            let shape: Vec<u32> = info.shape[..info.ndim as usize].to_vec();

            // Extract kernel_name (null-terminated string in [u8; 48])
            let kernel_name = {
                let name_len = info
                    .kernel_name
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(info.kernel_name.len());
                if name_len > 0 {
                    Some(String::from_utf8_lossy(&info.kernel_name[..name_len]).to_string())
                } else {
                    None
                }
            };

            records.push(TraceRecord {
                point: point_name.to_string(),
                layer: info.layer,
                token: info.token,
                position: info.position,
                backend: info.backend,
                shape,
                dtype: info.dtype,
                kernel_name,
                work_flops: info.work_flops,
                work_bytes: info.work_bytes,
                stats: TraceStats {
                    count: info.stats.count,
                    min: info.stats.min,
                    max: info.stats.max,
                    mean: info.stats.mean,
                    rms: info.stats.rms,
                    nan_count: info.stats.nan_count,
                    inf_count: info.stats.inf_count,
                },
                timestamp_ns: info.timestamp_ns,
            });
        }

        records
    }
}

impl Drop for XrayCaptureHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { talu_sys::talu_xray_capture_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

// =============================================================================
// Reference Recording & Verification System
// =============================================================================

/// Handle for reference recorder (recording phase).
pub struct ReferenceRecorderHandle {
    handle: *mut c_void,
}

impl ReferenceRecorderHandle {
    /// Create a new reference recorder for recording phase.
    pub fn new(model_name: &str, seed: u64, temperature: f32, max_tokens: u32) -> Result<Self> {
        let c_model_name = std::ffi::CString::new(model_name)?;
        let handle = unsafe {
            talu_sys::talu_xray_reference_recorder_create(
                c_model_name.as_ptr(),
                seed,
                temperature,
                max_tokens,
            )
        };
        if handle.is_null() {
            return Err(error_from_last_or("reference recorder create failed"));
        }
        Ok(Self { handle })
    }

    /// Record a sampled token.
    pub fn record_token(&self, token_id: u32) -> Result<()> {
        let ok = unsafe { talu_sys::talu_xray_reference_recorder_record_token(self.handle, token_id) };
        if !ok {
            return Err(error_from_last_or("failed to record token"));
        }
        Ok(())
    }

    /// Advance to next token position.
    pub fn next_token(&self) {
        unsafe { talu_sys::talu_xray_reference_recorder_next_token(self.handle) };
    }

    /// Finalize and return reference data. Consumes the recorder.
    pub fn finalize(mut self) -> Result<ReferenceDataHandle> {
        let data_handle = unsafe { talu_sys::talu_xray_reference_recorder_finalize(self.handle) };
        if data_handle.is_null() {
            return Err(error_from_last_or("failed to finalize recorder"));
        }
        // Prevent drop from destroying the already-consumed handle
        self.handle = std::ptr::null_mut();
        Ok(ReferenceDataHandle { handle: data_handle })
    }
}

impl Drop for ReferenceRecorderHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { talu_sys::talu_xray_reference_recorder_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Handle for reference data (can be saved/loaded from JSON).
pub struct ReferenceDataHandle {
    handle: *mut c_void,
}

impl ReferenceDataHandle {
    /// Load reference data from JSON file.
    pub fn load(path: &str) -> Result<Self> {
        let c_path = std::ffi::CString::new(path)?;
        let handle = unsafe { talu_sys::talu_xray_reference_data_load_json(c_path.as_ptr()) };
        if handle.is_null() {
            return Err(error_from_last_or("failed to load reference from JSON"));
        }
        Ok(Self { handle })
    }

    /// Save reference data to JSON file.
    pub fn save(&self, path: &str) -> Result<()> {
        let c_path = std::ffi::CString::new(path)?;
        let ok = unsafe { talu_sys::talu_xray_reference_data_save_json(self.handle, c_path.as_ptr()) };
        if !ok {
            return Err(error_from_last_or("failed to save reference to JSON"));
        }
        Ok(())
    }

    /// Get raw handle for passing to other FFI functions.
    pub fn as_ptr(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for ReferenceDataHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { talu_sys::talu_xray_reference_data_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Handle for reference verifier (verification phase).
pub struct ReferenceVerifierHandle {
    handle: *mut c_void,
}

impl ReferenceVerifierHandle {
    /// Create a new reference verifier for verification phase.
    pub fn new(reference: &ReferenceDataHandle, tolerance: f32) -> Result<Self> {
        let handle = unsafe {
            talu_sys::talu_xray_reference_verifier_create(reference.as_ptr(), tolerance)
        };
        if handle.is_null() {
            return Err(error_from_last_or("reference verifier create failed"));
        }
        Ok(Self { handle })
    }

    /// Get next forced token (for teacher forcing).
    /// Returns None if no more tokens available.
    pub fn get_next_token(&self) -> Option<u32> {
        let token = unsafe { talu_sys::talu_xray_reference_verifier_get_next_token(self.handle) };
        if token == 0xFFFFFFFF {
            None
        } else {
            Some(token)
        }
    }

    /// Advance to next token position.
    pub fn next_token(&self) {
        unsafe { talu_sys::talu_xray_reference_verifier_next_token(self.handle) };
    }

    /// Check if verification has detected divergence.
    pub fn has_diverged(&self) -> bool {
        unsafe { talu_sys::talu_xray_reference_verifier_has_diverged(self.handle) }
    }

    /// Get raw handle for passing to other FFI functions.
    pub fn as_ptr(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for ReferenceVerifierHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { talu_sys::talu_xray_reference_verifier_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Handle for verify capture (enhanced capture with recording/verification).
pub struct VerifyCaptureHandle {
    handle: *mut c_void,
}

impl VerifyCaptureHandle {
    /// Create verify capture in recording mode.
    pub fn new_recording(recorder: &ReferenceRecorderHandle) -> Result<Self> {
        let handle = unsafe {
            talu_sys::talu_xray_verify_capture_create_recording(recorder.handle)
        };
        if handle.is_null() {
            return Err(error_from_last_or("verify capture create (recording) failed"));
        }
        Ok(Self { handle })
    }

    /// Create verify capture in verification mode.
    /// panic_dump_dir can be None to disable panic dumps.
    pub fn new_verification(
        verifier: &ReferenceVerifierHandle,
        panic_dump_dir: Option<&str>,
    ) -> Result<Self> {
        let c_dir = panic_dump_dir.map(|s| std::ffi::CString::new(s).ok()).flatten();
        let dir_ptr = c_dir.as_ref().map(|cs| cs.as_ptr()).unwrap_or(std::ptr::null());

        let handle = unsafe {
            talu_sys::talu_xray_verify_capture_create_verification(verifier.as_ptr(), dir_ptr)
        };
        if handle.is_null() {
            return Err(error_from_last_or("verify capture create (verification) failed"));
        }
        Ok(Self { handle })
    }

    /// Enable verify capture (start receiving trace emissions).
    pub fn enable(&self) {
        unsafe { talu_sys::talu_xray_verify_capture_enable(self.handle) };
    }

    /// Disable verify capture (stop receiving trace emissions).
    pub fn disable() {
        unsafe { talu_sys::talu_xray_verify_capture_disable() };
    }
}

impl Drop for VerifyCaptureHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { talu_sys::talu_xray_verify_capture_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Teacher forcing control (global state).
pub struct TeacherForcing;

impl TeacherForcing {
    /// Enable teacher forcing with a verifier as the token source.
    pub fn enable_with_verifier(verifier: &ReferenceVerifierHandle) {
        unsafe { talu_sys::talu_xray_teacher_forcing_enable_with_verifier(verifier.as_ptr()) };
    }

    /// Disable teacher forcing (return to normal sampling).
    pub fn disable() {
        unsafe { talu_sys::talu_xray_teacher_forcing_disable() };
    }

    /// Check if teacher forcing is active.
    pub fn is_enabled() -> bool {
        unsafe { talu_sys::talu_xray_teacher_forcing_is_enabled() }
    }

    /// Get next forced token (for use by sampler).
    /// Returns None if teacher forcing is disabled or no more tokens.
    pub fn get_next_token() -> Option<u32> {
        let token = unsafe { talu_sys::talu_xray_teacher_forcing_get_next_token() };
        if token == 0xFFFFFFFF {
            None
        } else {
            Some(token)
        }
    }
}
