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
    /// Tensor shape
    pub shape: Vec<u32>,
    /// Data type
    pub dtype: u8,
    /// Kernel name that produced this tensor (e.g., "matmul_q4_cpu_avx512")
    pub kernel_name: Option<String>,
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
    /// Create a new capture handle with all trace points enabled.
    pub fn new() -> Result<Self> {
        // Mode 0 = stats only, sample_count = 0
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
                shape,
                dtype: info.dtype,
                kernel_name,
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
