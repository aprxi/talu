use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use talu::error::last_error_message;
use talu::{ChatHandle, XrayCaptureHandle};

use crate::provider::create_backend_for_model;

use super::repo::resolve_model_path;
use super::XrayArgs;

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn _exit(status: i32) -> !;
}

struct VerifyOverrideGuard {
    point_mask_overridden: bool,
    exact_emission_overridden: bool,
}

#[derive(Clone, Copy)]
struct ExactEmissionOverride {
    point: u8,
    layer: u16,
    position: u32,
}

impl VerifyOverrideGuard {
    fn new(
        teacher_forcing: bool,
        full_capture: bool,
        point_mask_override: Option<u64>,
        exact_emission_override: Option<ExactEmissionOverride>,
    ) -> Self {
        // Verify runner invariant:
        // Any temporary state configured here must remain transcript-level only.
        // Do not add toggles that alter backend execution behavior.
        talu::xray::VerifyCaptureHandle::set_ignore_token_parity(teacher_forcing);
        talu::xray::VerifyCaptureHandle::set_token_only(!teacher_forcing);
        talu::xray::VerifyCaptureHandle::set_full_capture(full_capture);
        if let Some(mask) = point_mask_override {
            talu::xray::VerifyCaptureHandle::set_point_mask(mask);
        }
        if let Some(target) = exact_emission_override {
            talu::xray::VerifyCaptureHandle::set_exact_emission_filter(
                target.point,
                target.layer,
                target.position,
            );
        }
        Self {
            point_mask_overridden: point_mask_override.is_some(),
            exact_emission_overridden: exact_emission_override.is_some(),
        }
    }
}

impl Drop for VerifyOverrideGuard {
    fn drop(&mut self) {
        talu::xray::VerifyCaptureHandle::clear_ignore_token_parity_override();
        talu::xray::VerifyCaptureHandle::clear_token_only_override();
        talu::xray::VerifyCaptureHandle::clear_full_capture_override();
        if self.point_mask_overridden {
            talu::xray::VerifyCaptureHandle::clear_point_mask_override();
        }
        if self.exact_emission_overridden {
            talu::xray::VerifyCaptureHandle::clear_exact_emission_filter();
        }
    }
}

struct ActiveVerifyRunGuard {
    teacher_forcing_enabled: bool,
}

impl ActiveVerifyRunGuard {
    fn activate(
        teacher_forcing: bool,
        verifier: &talu::xray::ReferenceVerifierHandle,
        verify_cap: &talu::xray::VerifyCaptureHandle,
    ) -> Self {
        verify_cap.enable();
        if teacher_forcing {
            talu::xray::TeacherForcing::enable_with_verifier(verifier);
        }
        Self {
            teacher_forcing_enabled: teacher_forcing,
        }
    }
}

impl Drop for ActiveVerifyRunGuard {
    fn drop(&mut self) {
        if self.teacher_forcing_enabled {
            talu::xray::TeacherForcing::disable();
        }
        talu::xray::VerifyCaptureHandle::disable();
    }
}

struct CacheRefreshLock {
    path: PathBuf,
    _file: File,
}

impl CacheRefreshLock {
    fn acquire(reference_path: &Path) -> Result<Self> {
        let path = reference_lock_path(reference_path);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|err| {
                anyhow!(
                    "xray CPU reference cache is already being refreshed: {} ({err})",
                    path.display()
                )
            })?;
        Ok(Self { path, _file: file })
    }
}

impl Drop for CacheRefreshLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

struct TempReferenceBundle {
    reference_path: PathBuf,
    sidecar_path: PathBuf,
    keep: bool,
}

impl TempReferenceBundle {
    fn new(final_reference_path: &Path) -> Self {
        let reference_path = temp_reference_path(final_reference_path, std::process::id());
        let sidecar_path = reference_sidecar_path(&reference_path);
        Self {
            reference_path,
            sidecar_path,
            keep: false,
        }
    }

    fn persist(&mut self, final_reference_path: &Path) -> Result<()> {
        if !self.reference_path.exists() {
            return Err(anyhow!(
                "CPU reference refresh did not produce JSON cache: {}",
                self.reference_path.display()
            ));
        }
        if !self.sidecar_path.exists() {
            return Err(anyhow!(
                "CPU reference refresh did not produce NPZ sidecar: {}",
                self.sidecar_path.display()
            ));
        }

        let final_sidecar_path = reference_sidecar_path(final_reference_path);
        replace_file_atomically(&self.reference_path, final_reference_path)?;
        replace_file_atomically(&self.sidecar_path, &final_sidecar_path)?;
        self.keep = true;
        Ok(())
    }
}

impl Drop for TempReferenceBundle {
    fn drop(&mut self) {
        if self.keep {
            return;
        }
        let _ = std::fs::remove_file(&self.reference_path);
        let _ = std::fs::remove_file(&self.sidecar_path);
    }
}

struct ScopedTempJson {
    path: PathBuf,
}

impl ScopedTempJson {
    fn create(prefix: &str) -> Result<(Self, File)> {
        let temp_dir = std::env::temp_dir();
        for attempt in 0..32u32 {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|err| anyhow!("system clock error while creating temp file: {err}"))?
                .as_nanos();
            let path = temp_dir.join(format!(
                "{}_{}_{}_{}.json",
                prefix,
                std::process::id(),
                nonce,
                attempt
            ));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => return Ok((Self { path }, file)),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => {
                    return Err(anyhow!(
                        "failed to create temp JSON file {}: {err}",
                        path.display()
                    ));
                }
            }
        }
        Err(anyhow!(
            "failed to allocate unique temp JSON path after repeated attempts"
        ))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScopedTempJson {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn synchronize_backend(backend: &talu::InferenceBackend) -> Result<()> {
    let rc = unsafe { talu_sys::talu_backend_synchronize(backend.as_ptr()) };
    if rc != 0 {
        let message =
            last_error_message().unwrap_or_else(|| "backend synchronize failed".to_string());
        return Err(anyhow!("Error: {} (code {})", message, rc));
    }
    Ok(())
}

fn maybe_hard_exit_after_verify(result: Result<()>) -> Result<()> {
    if should_process_scope_backend_teardown() {
        let _exit_code = if result.is_ok() { 0 } else { 1 };
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        #[cfg(target_os = "macos")]
        unsafe {
            _exit(_exit_code);
        }
        #[allow(unreachable_code)]
        return result;
    }
    result
}

/// Kernel usage derived from trace records.
struct KernelUsage {
    name: String,
    call_count: usize,
    total_us: u64,
}

struct KernelShapeUsageRow {
    logical_op: String,
    backend: String,
    kernel_name: String,
    shape: String,
    call_count: usize,
    total_us: u64,
    share_global_pct: f64,
    avg_us: u64,
    p50_us: u64,
    p90_us: u64,
    min_us: u64,
    max_us: u64,
    work_flops: u64,
    work_bytes: u64,
}

struct KernelShapeAccumulator {
    call_count: usize,
    total_us: u64,
    total_work_flops: u64,
    total_work_bytes: u64,
    samples_us: Vec<u64>,
    point_counts: std::collections::HashMap<String, usize>,
}

struct PointUsageRow {
    point: String,
    call_count: usize,
    total_us: u64,
}

struct PointAccumulator {
    call_count: usize,
    total_us: u64,
}

struct PointUsage {
    point: String,
    call_count: usize,
    total_us: u64,
}

struct TimelineSegmentRow {
    seq: usize,
    calls: usize,
    total_us: u64,
    avg_us: u64,
    first_ns: u64,
    last_ns: u64,
    backend: String,
    point: String,
    kernel_name: String,
    shape: String,
}

struct ParentSummaryRow {
    parent: String,
    calls: usize,
    total_us: u64,
    share_pct: f64,
    first_ns: u64,
}

struct EdgeTransitionRow {
    from_point: String,
    to_point: String,
    backend: String,
    to_kernel: String,
    to_shape: String,
    calls: usize,
    total_us: u64,
    avg_us: u64,
    p50_us: u64,
    p90_us: u64,
    min_us: u64,
    max_us: u64,
    work_flops: u64,
    work_bytes: u64,
    first_ns: u64,
}

struct KernelPointUsage {
    kernel_name: String,
    total_us: u64,
    points: Vec<PointUsage>,
}

#[derive(Debug, Deserialize)]
struct PerfHintsJson {
    bench_model: String,
}

fn truncate_with_ellipsis(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    if max_chars <= 2 {
        return ".".repeat(max_chars);
    }
    let keep = max_chars - 2;
    let prefix: String = s.chars().take(keep).collect();
    format!("{prefix}..")
}

fn display_kernel_name(kernel_name: &str) -> &str {
    if kernel_name.is_empty() || kernel_name == "other(non-kernel)" {
        "-"
    } else {
        kernel_name
    }
}

fn derive_timeline_segments(
    trace: &[talu::TraceRecord],
    record_delta_ns: &[i64],
) -> Vec<TimelineSegmentRow> {
    if trace.is_empty() {
        return Vec::new();
    }

    let trace_start_ns = trace.first().map(|r| r.timestamp_ns).unwrap_or(0);
    let mut rows: Vec<TimelineSegmentRow> = Vec::new();

    for (i, record) in trace.iter().enumerate() {
        let backend = record.backend_name().to_string();
        let kernel_name = match &record.kernel_name {
            Some(name) if !name.is_empty() => name.clone(),
            _ => "other(non-kernel)".to_string(),
        };
        let shape = record.shape_str();
        let point = record.point.clone();
        let rel_ns = record.timestamp_ns.saturating_sub(trace_start_ns) as u64;
        let delta_us = record_delta_ns[i].unsigned_abs() / 1_000;

        if let Some(last) = rows.last_mut() {
            if last.backend == backend
                && last.point == point
                && last.kernel_name == kernel_name
                && last.shape == shape
            {
                last.calls += 1;
                last.total_us += delta_us;
                last.avg_us = if last.calls > 0 {
                    last.total_us / last.calls as u64
                } else {
                    0
                };
                last.last_ns = rel_ns;
                continue;
            }
        }

        rows.push(TimelineSegmentRow {
            seq: rows.len(),
            calls: 1,
            total_us: delta_us,
            avg_us: delta_us,
            first_ns: rel_ns,
            last_ns: rel_ns,
            backend,
            point,
            kernel_name,
            shape,
        });
    }

    rows
}

fn point_parent(point: &str) -> String {
    if let Some((prefix, _)) = point.split_once('.') {
        return prefix.to_string();
    }
    if point.starts_with("layer_") {
        return "layer".to_string();
    }
    if point.starts_with("logits_") {
        return "logits".to_string();
    }
    if point.starts_with("embed_") {
        return "embed".to_string();
    }
    if point == "token_select" {
        return "token".to_string();
    }
    if point == "final_norm" {
        return "final".to_string();
    }
    point.to_string()
}

fn derive_parent_summary_rows(
    timeline_segments: &[TimelineSegmentRow],
    total_us: u64,
) -> Vec<ParentSummaryRow> {
    let mut by_parent: std::collections::HashMap<String, (usize, u64, u64)> =
        std::collections::HashMap::new();
    for seg in timeline_segments {
        let parent = point_parent(&seg.point);
        let entry = by_parent.entry(parent).or_insert((0, 0, seg.first_ns));
        entry.0 += seg.calls;
        entry.1 += seg.total_us;
        entry.2 = entry.2.min(seg.first_ns);
    }

    let mut rows: Vec<ParentSummaryRow> = by_parent
        .into_iter()
        .map(|(parent, (calls, total_us_parent, first_ns))| {
            let share_pct = if total_us > 0 {
                (total_us_parent as f64 / total_us as f64) * 100.0
            } else {
                0.0
            };
            ParentSummaryRow {
                parent,
                calls,
                total_us: total_us_parent,
                share_pct,
                first_ns,
            }
        })
        .collect();

    rows.sort_by(|a, b| {
        b.total_us
            .cmp(&a.total_us)
            .then_with(|| a.first_ns.cmp(&b.first_ns))
            .then_with(|| a.parent.cmp(&b.parent))
    });
    rows
}

fn derive_edge_transition_rows(
    trace: &[talu::TraceRecord],
    record_delta_ns: &[i64],
) -> Vec<EdgeTransitionRow> {
    if trace.len() < 2 {
        return Vec::new();
    }
    let trace_start_ns = trace.first().map(|r| r.timestamp_ns).unwrap_or(0);
    let mut by_edge: std::collections::HashMap<
        String,
        (
            String,
            String,
            String,
            String,
            String,
            usize,
            u64,
            u64,
            u64,
            Vec<u64>,
            u64,
        ),
    > = std::collections::HashMap::new();

    for i in 1..trace.len() {
        let prev = &trace[i - 1];
        let cur = &trace[i];
        let backend = cur.backend_name().to_string();
        let to_kernel = match &cur.kernel_name {
            Some(name) if !name.is_empty() => name.clone(),
            _ => "other(non-kernel)".to_string(),
        };
        let to_shape = cur.shape_str();
        let delta_us = record_delta_ns[i].unsigned_abs() / 1_000;
        let work_flops = cur.work_flops;
        let work_bytes = cur.work_bytes;
        let rel_ns = cur.timestamp_ns.saturating_sub(trace_start_ns) as u64;
        let key = format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
            prev.point, cur.point, backend, to_kernel, to_shape
        );
        let entry = by_edge.entry(key).or_insert_with(|| {
            (
                prev.point.clone(),
                cur.point.clone(),
                backend.clone(),
                to_kernel.clone(),
                to_shape.clone(),
                0,
                0,
                0,
                0,
                Vec::new(),
                rel_ns,
            )
        });
        entry.5 += 1;
        entry.6 += delta_us;
        entry.7 += work_flops;
        entry.8 += work_bytes;
        entry.9.push(delta_us);
        entry.10 = entry.10.min(rel_ns);
    }

    let mut rows: Vec<EdgeTransitionRow> = by_edge
        .into_values()
        .map(
            |(
                from_point,
                to_point,
                backend,
                to_kernel,
                to_shape,
                calls,
                total_us_edge,
                total_work_flops,
                total_work_bytes,
                samples,
                first_ns,
            )| {
                let avg_us = if calls > 0 {
                    total_us_edge / calls as u64
                } else {
                    0
                };
                EdgeTransitionRow {
                    from_point,
                    to_point,
                    backend,
                    to_kernel,
                    to_shape,
                    calls,
                    total_us: total_us_edge,
                    avg_us,
                    p50_us: percentile_us(&samples, 50),
                    p90_us: percentile_us(&samples, 90),
                    min_us: *samples.iter().min().unwrap_or(&0),
                    max_us: *samples.iter().max().unwrap_or(&0),
                    work_flops: total_work_flops,
                    work_bytes: total_work_bytes,
                    first_ns,
                }
            },
        )
        .collect();

    rows.sort_by(|a, b| {
        b.total_us
            .cmp(&a.total_us)
            .then_with(|| a.first_ns.cmp(&b.first_ns))
            .then_with(|| a.from_point.cmp(&b.from_point))
            .then_with(|| a.to_point.cmp(&b.to_point))
    });
    rows
}

/// Derive kernel usage from trace records (counts + timing).
/// Each record's delta is attributed to its kernel_name.
fn derive_kernel_usage(trace: &[talu::TraceRecord], record_delta_ns: &[i64]) -> Vec<KernelUsage> {
    let mut stats: std::collections::HashMap<String, (usize, u64)> =
        std::collections::HashMap::new();
    for (i, record) in trace.iter().enumerate() {
        if let Some(name) = &record.kernel_name {
            if !name.is_empty() {
                let delta_us = record_delta_ns[i].unsigned_abs() / 1_000;
                let entry = stats.entry(name.clone()).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += delta_us;
            }
        }
    }
    let mut usage: Vec<KernelUsage> = stats
        .into_iter()
        .map(|(name, (call_count, total_us))| KernelUsage {
            name,
            call_count,
            total_us,
        })
        .collect();
    usage.sort_by(|a, b| b.total_us.cmp(&a.total_us));
    usage
}

fn derive_kernel_usage_by_point(
    trace: &[talu::TraceRecord],
    record_delta_ns: &[i64],
) -> Vec<KernelPointUsage> {
    let mut kernel_point_stats: std::collections::HashMap<
        String,
        std::collections::HashMap<String, (usize, u64)>,
    > = std::collections::HashMap::new();
    let mut kernel_totals: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();

    for (i, record) in trace.iter().enumerate() {
        if let Some(kernel_name) = &record.kernel_name {
            if !kernel_name.is_empty() {
                let delta_us = record_delta_ns[i].unsigned_abs() / 1_000;
                *kernel_totals.entry(kernel_name.clone()).or_insert(0) += delta_us;
                let point_stats = kernel_point_stats.entry(kernel_name.clone()).or_default();
                let entry = point_stats.entry(record.point.clone()).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += delta_us;
            }
        }
    }

    let mut usage: Vec<KernelPointUsage> = kernel_point_stats
        .into_iter()
        .map(|(kernel_name, points)| {
            let mut point_usage: Vec<PointUsage> = points
                .into_iter()
                .map(|(point, (call_count, total_us))| PointUsage {
                    point,
                    call_count,
                    total_us,
                })
                .collect();
            point_usage.sort_by(|a, b| {
                b.total_us
                    .cmp(&a.total_us)
                    .then_with(|| a.point.cmp(&b.point))
            });
            KernelPointUsage {
                total_us: kernel_totals.get(&kernel_name).copied().unwrap_or(0),
                kernel_name,
                points: point_usage,
            }
        })
        .collect();
    usage.sort_by(|a, b| {
        b.total_us
            .cmp(&a.total_us)
            .then_with(|| a.kernel_name.cmp(&b.kernel_name))
    });
    usage
}

fn derive_other_usage(trace: &[talu::TraceRecord], record_delta_ns: &[i64]) -> Vec<PointUsage> {
    let mut stats: std::collections::HashMap<String, (usize, u64)> =
        std::collections::HashMap::new();
    for (i, record) in trace.iter().enumerate() {
        let has_kernel = record
            .kernel_name
            .as_ref()
            .map(|name| !name.is_empty())
            .unwrap_or(false);
        if !has_kernel {
            let delta_us = record_delta_ns[i].unsigned_abs() / 1_000;
            let entry = stats.entry(record.point.clone()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += delta_us;
        }
    }

    let mut usage: Vec<PointUsage> = stats
        .into_iter()
        .map(|(point, (call_count, total_us))| PointUsage {
            point,
            call_count,
            total_us,
        })
        .collect();
    usage.sort_by(|a, b| {
        b.total_us
            .cmp(&a.total_us)
            .then_with(|| a.point.cmp(&b.point))
    });
    usage
}

fn derive_record_delta_ns(trace: &[talu::TraceRecord]) -> Vec<i64> {
    (0..trace.len())
        .map(|i| {
            if i == 0 {
                0
            } else {
                trace[i].timestamp_ns - trace[i - 1].timestamp_ns
            }
        })
        .collect()
}

fn split_decode_trace(all_trace: Vec<talu::TraceRecord>) -> Vec<talu::TraceRecord> {
    // Decode window starts at the *last* layer-0 layer_input with seq_len=1.
    // Using last (not first) avoids including prefill for architectures that emit
    // per-token recurrent steps with shape[1] == 1 during prompt processing.
    if let Some(split_idx) = all_trace.iter().rposition(|r| {
        r.point == "layer_input" && r.layer == 0 && r.shape.len() >= 2 && r.shape[1] == 1
    }) {
        return all_trace.into_iter().skip(split_idx).collect();
    }

    // Fallback: last layer-0 layer_input regardless of shape.
    if let Some(split_idx) = all_trace
        .iter()
        .rposition(|r| r.point == "layer_input" && r.layer == 0)
    {
        return all_trace.into_iter().skip(split_idx).collect();
    }

    // Last-resort fallback: keep full trace.
    all_trace
}

fn detect_token_count(trace: &[talu::TraceRecord]) -> u64 {
    trace
        .iter()
        .find(|r| r.shape.len() >= 2 && r.shape[0] == 1)
        .map(|r| r.shape[1] as u64)
        .unwrap_or(1)
}

fn percentile_us(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let last = sorted.len() - 1;
    let idx = (last * percentile) / 100;
    sorted[idx]
}

fn tflops_from_work(flops: u64, total_us: u64) -> Option<f64> {
    if flops == 0 || total_us == 0 {
        None
    } else {
        Some(flops as f64 / (total_us as f64 * 1_000_000.0))
    }
}

fn gbps_from_work(bytes: u64, total_us: u64) -> Option<f64> {
    if bytes == 0 || total_us == 0 {
        None
    } else {
        Some(bytes as f64 / (total_us as f64 * 1_000.0))
    }
}

fn format_rate(rate: Option<f64>) -> String {
    match rate {
        Some(v) => format!("{v:.1}"),
        None => "-".to_string(),
    }
}

fn derive_point_table(trace: &[talu::TraceRecord], record_delta_ns: &[i64]) -> Vec<PointUsageRow> {
    let mut by_point: std::collections::HashMap<String, PointAccumulator> =
        std::collections::HashMap::new();
    for (i, record) in trace.iter().enumerate() {
        let delta_us = record_delta_ns[i].unsigned_abs() / 1_000;
        let entry = by_point
            .entry(record.point.clone())
            .or_insert_with(|| PointAccumulator {
                call_count: 0,
                total_us: 0,
            });
        entry.call_count += 1;
        entry.total_us += delta_us;
    }

    by_point
        .into_iter()
        .map(|(point, acc)| PointUsageRow {
            point,
            call_count: acc.call_count,
            total_us: acc.total_us,
        })
        .collect()
}

fn derive_kernel_shape_table(
    trace: &[talu::TraceRecord],
    record_delta_ns: &[i64],
    total_us: u64,
) -> Vec<KernelShapeUsageRow> {
    let mut by_key: std::collections::HashMap<
        String,
        (String, String, String, KernelShapeAccumulator),
    > = std::collections::HashMap::new();
    for (i, record) in trace.iter().enumerate() {
        let backend = record.backend_name().to_string();
        let kernel_name: String = match &record.kernel_name {
            Some(name) if !name.is_empty() => name.clone(),
            _ => "other(non-kernel)".to_string(),
        };
        let shape = record.shape_str();
        let key = format!("{backend}\u{1f}{kernel_name}\u{1f}{shape}");
        let delta_us = record_delta_ns[i].unsigned_abs() / 1_000;
        let entry = by_key.entry(key).or_insert_with(|| {
            (
                backend.clone(),
                kernel_name.clone(),
                shape.clone(),
                KernelShapeAccumulator {
                    call_count: 0,
                    total_us: 0,
                    total_work_flops: 0,
                    total_work_bytes: 0,
                    samples_us: Vec::new(),
                    point_counts: std::collections::HashMap::new(),
                },
            )
        });
        entry.3.call_count += 1;
        entry.3.total_us += delta_us;
        entry.3.total_work_flops += record.work_flops;
        entry.3.total_work_bytes += record.work_bytes;
        entry.3.samples_us.push(delta_us);
        *entry
            .3
            .point_counts
            .entry(record.point.clone())
            .or_insert(0) += 1;
    }

    let mut rows: Vec<KernelShapeUsageRow> = by_key
        .into_values()
        .map(|(backend, kernel_name, shape, acc)| {
            let logical_op = if acc.point_counts.len() == 1 {
                acc.point_counts
                    .keys()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string())
            } else if acc.point_counts.is_empty() {
                "unknown".to_string()
            } else {
                let mut points: Vec<(String, usize)> = acc.point_counts.into_iter().collect();
                points.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let top = &points[0].0;
                let extra = points.len() - 1;
                format!("{top}+{extra}")
            };
            let share_global_pct = if total_us > 0 {
                (acc.total_us as f64 / total_us as f64) * 100.0
            } else {
                0.0
            };
            let avg_us = if acc.call_count > 0 {
                acc.total_us / acc.call_count as u64
            } else {
                0
            };
            KernelShapeUsageRow {
                logical_op,
                backend,
                kernel_name,
                shape,
                call_count: acc.call_count,
                total_us: acc.total_us,
                share_global_pct,
                avg_us,
                p50_us: percentile_us(&acc.samples_us, 50),
                p90_us: percentile_us(&acc.samples_us, 90),
                min_us: *acc.samples_us.iter().min().unwrap_or(&0),
                max_us: *acc.samples_us.iter().max().unwrap_or(&0),
                work_flops: acc.total_work_flops,
                work_bytes: acc.total_work_bytes,
            }
        })
        .collect();

    rows.sort_by(|a, b| {
        b.total_us
            .cmp(&a.total_us)
            .then_with(|| a.backend.cmp(&b.backend))
            .then_with(|| a.kernel_name.cmp(&b.kernel_name))
            .then_with(|| a.shape.cmp(&b.shape))
    });
    rows
}

fn print_method_table(
    model_info: &talu::ModelInfo,
    mode_label: &str,
    mut kernel_rows: Vec<KernelShapeUsageRow>,
    mut point_rows: Vec<PointUsageRow>,
    edge_rows: Vec<EdgeTransitionRow>,
    timeline_segments: Vec<TimelineSegmentRow>,
    total_us: u64,
    token_count: u64,
    debug: bool,
    perf_hints: Option<PerfHintsJson>,
) {
    kernel_rows.sort_by(|a, b| {
        b.total_us
            .cmp(&a.total_us)
            .then_with(|| a.backend.cmp(&b.backend))
            .then_with(|| a.kernel_name.cmp(&b.kernel_name))
            .then_with(|| a.shape.cmp(&b.shape))
    });
    point_rows.sort_by(|a, b| {
        b.total_us
            .cmp(&a.total_us)
            .then_with(|| a.point.cmp(&b.point))
    });

    let model_name = model_info
        .architecture
        .split("For")
        .next()
        .unwrap_or(&model_info.model_type);
    println!(
        "{} ({}) \u{2014} {} [table]",
        model_name, model_info.model_type, mode_label
    );
    println!(
        "total={}  per_token={}  tokens={}",
        format_us(total_us, 1),
        format_us(
            if token_count > 0 {
                total_us / token_count
            } else {
                total_us
            },
            1
        ),
        token_count,
    );
    println!();

    let kernels_total_us: u64 = kernel_rows.iter().map(|r| r.total_us).sum();
    let kernels_total_calls: usize = kernel_rows.iter().map(|r| r.call_count).sum();
    let kernels_total_work_flops: u64 = kernel_rows.iter().map(|r| r.work_flops).sum();
    let kernels_total_work_bytes: u64 = kernel_rows.iter().map(|r| r.work_bytes).sum();
    let total_tok_s = if total_us > 0 {
        (token_count as f64 * 1_000_000.0) / total_us as f64
    } else {
        0.0
    };

    let parent_rows = derive_parent_summary_rows(&timeline_segments, total_us);
    println!();
    println!("Stage Totals (Auto)");
    println!(
        "{:<12} {:>7} {:>10} {:>6}",
        "source", "calls", "total_us", "share",
    );
    println!("{}", "-".repeat(40));
    for row in &parent_rows {
        println!(
            "{:<12} {:>7} {:>10} {:>5.1}%",
            row.parent, row.calls, row.total_us, row.share_pct,
        );
    }
    println!("{}", "-".repeat(40));
    println!(
        "{:<12} {:>7} {:>10} {:>6}",
        "sum",
        parent_rows.iter().map(|r| r.calls).sum::<usize>(),
        parent_rows.iter().map(|r| r.total_us).sum::<u64>(),
        "100.0%",
    );
    println!("  derived throughput: {:.1} tok/s", total_tok_s);

    println!();
    println!("Flow Edges (Zoom)");
    let flow_header = format!(
        "{:<6} {:<6} {:<24} {:<16} {:<10} {:>7} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "group",
        "level",
        "item",
        "source",
        "shape",
        "calls",
        "total_us",
        "global%",
        "split%",
        "avg_us",
        "p50_us",
        "p90_us",
        "min_us",
        "max_us",
        "TF/s",
        "GB/s",
    );
    println!("{}", flow_header);
    println!("{}", "-".repeat(flow_header.chars().count()));

    let mut edges_by_bucket: std::collections::HashMap<String, Vec<&EdgeTransitionRow>> =
        std::collections::HashMap::new();
    for edge in &edge_rows {
        let key = format!(
            "{}\u{1f}{}\u{1f}{}",
            edge.backend, edge.to_kernel, edge.to_shape
        );
        edges_by_bucket.entry(key).or_default().push(edge);
    }
    for rows in edges_by_bucket.values_mut() {
        rows.sort_by(|a, b| {
            b.total_us
                .cmp(&a.total_us)
                .then_with(|| a.from_point.cmp(&b.from_point))
                .then_with(|| a.to_point.cmp(&b.to_point))
        });
    }

    let mut folded_edge_rows: usize = 0;
    let mut backend_group_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for bucket in &kernel_rows {
        let backend_group_idx = backend_group_counts
            .entry(bucket.backend.clone())
            .and_modify(|n| *n += 1)
            .or_insert(1);
        let group_id = format!("{}{}", bucket.backend, backend_group_idx);
        let bucket_key = format!(
            "{}\u{1f}{}\u{1f}{}",
            bucket.backend, bucket.kernel_name, bucket.shape
        );
        let bucket_edges = edges_by_bucket.get(&bucket_key);
        let single_exact_edge = bucket_edges.and_then(|edges| {
            if edges.len() == 1
                && edges[0].calls == bucket.call_count
                && edges[0].total_us == bucket.total_us
            {
                Some(edges[0])
            } else {
                None
            }
        });
        let bucket_item = if let Some(edge) = single_exact_edge {
            folded_edge_rows += 1;
            format!(
                "{} [{} -> {}]",
                bucket.logical_op, edge.from_point, edge.to_point
            )
        } else {
            bucket.logical_op.clone()
        };
        let bucket_global = format!("{:.1}%", bucket.share_global_pct);
        let bucket_tf = format_rate(tflops_from_work(bucket.work_flops, bucket.total_us));
        let bucket_gb = format_rate(gbps_from_work(bucket.work_bytes, bucket.total_us));
        println!(
            "{:<6} {:<6} {:<24} {:<16} {:<10} {:>7} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            group_id,
            "bucket",
            truncate_with_ellipsis(&bucket_item, 24),
            truncate_with_ellipsis(display_kernel_name(&bucket.kernel_name), 16),
            truncate_with_ellipsis(&bucket.shape, 10),
            bucket.call_count,
            bucket.total_us,
            bucket_global,
            "-",
            bucket.avg_us,
            bucket.p50_us,
            bucket.p90_us,
            bucket.min_us,
            bucket.max_us,
            bucket_tf,
            bucket_gb,
        );

        if single_exact_edge.is_none() {
            if let Some(edges) = bucket_edges {
                for edge in edges {
                    let edge_item = format!("{} -> {}", edge.from_point, edge.to_point);
                    let edge_item = format!("|- {}", edge_item);
                    let edge_split_pct = if bucket.total_us > 0 {
                        (edge.total_us as f64 / bucket.total_us as f64) * 100.0
                    } else {
                        0.0
                    };
                    let edge_split = format!("{:.1}%", edge_split_pct);
                    let edge_tf = format_rate(tflops_from_work(edge.work_flops, edge.total_us));
                    let edge_gb = format_rate(gbps_from_work(edge.work_bytes, edge.total_us));
                    println!(
                    "{:<6} {:<6} {:<24} {:<16} {:<10} {:>7} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
                    group_id,
                    "edge",
                    truncate_with_ellipsis(&edge_item, 24),
                    truncate_with_ellipsis(display_kernel_name(&edge.to_kernel), 16),
                    truncate_with_ellipsis(&edge.to_shape, 10),
                    edge.calls,
                    edge.total_us,
                    "-",
                    edge_split,
                    edge.avg_us,
                    edge.p50_us,
                    edge.p90_us,
                    edge.min_us,
                    edge.max_us,
                    edge_tf,
                    edge_gb,
                );
                }
            }
        }
    }

    println!("{}", "-".repeat(flow_header.chars().count()));
    let total_tf = format_rate(tflops_from_work(kernels_total_work_flops, kernels_total_us));
    let total_gb = format_rate(gbps_from_work(kernels_total_work_bytes, kernels_total_us));
    println!(
        "{:<6} {:<6} {:<24} {:<16} {:<10} {:>7} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "-",
        "-",
        "TOTAL(bucket)",
        "",
        "",
        kernels_total_calls,
        kernels_total_us,
        "100.0%",
        "",
        "",
        "",
        "",
        "",
        "",
        total_tf,
        total_gb,
    );
    println!("  derived throughput: {:.1} tok/s", total_tok_s);
    if folded_edge_rows > 0 {
        println!(
            "  folded {} exact 1:1 edge rows into bucket rows",
            folded_edge_rows
        );
    }

    if debug {
        println!();
        println!("Timeline (Exact, Contiguous Segments)");
        println!(
            "{:>4} {:>6} {:>10} {:>9} {:<7} {:<22} {:<24} {:<20}",
            "seq", "calls", "total_us", "avg_us", "backend", "logical_op", "kernel", "shape",
        );
        println!("{}", "-".repeat(112));
        for row in &timeline_segments {
            println!(
                "{:>4} {:>6} {:>10} {:>9} {:<7} {:<22} {:<24} {:<20}",
                row.seq,
                row.calls,
                row.total_us,
                row.avg_us,
                row.backend,
                truncate_with_ellipsis(&row.point, 22),
                truncate_with_ellipsis(&row.kernel_name, 24),
                row.shape,
            );
        }
        println!("{}", "-".repeat(112));
        println!(
            "{:>4} {:>6} {:>10}",
            "-",
            timeline_segments.iter().map(|r| r.calls).sum::<usize>(),
            timeline_segments.iter().map(|r| r.total_us).sum::<u64>(),
        );
    }

    if let Some(perf_hints) = perf_hints {
        println!();
        println!("Bench helper:");
        println!(
            "  make -C core/bench/compute/cpu model={}",
            perf_hints.bench_model
        );
    }

    print_checkpoint_consistency_warnings(&point_rows);
}

fn print_checkpoint_consistency_warnings(point_rows: &[PointUsageRow]) {
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for row in point_rows {
        counts.insert(row.point.as_str(), row.call_count);
    }

    let gdelta_points = [
        "gdelta.in_proj",
        "gdelta.conv",
        "gdelta.ssm",
        "gdelta.norm",
        "gdelta.out",
    ];
    let mut present_counts: Vec<(&str, usize)> = Vec::new();
    for p in gdelta_points {
        if let Some(c) = counts.get(p).copied() {
            present_counts.push((p, c));
        }
    }
    if present_counts.len() > 1 {
        let base = present_counts[0].1;
        let mismatch = present_counts.iter().any(|(_, c)| *c != base);
        if mismatch {
            println!();
            println!("Checkpoint Consistency Warnings");
            println!("  gated-delta stage counts mismatch:");
            for (name, c) in &present_counts {
                println!("    {}: {}", name, c);
            }
        }
    }
}

/// Recording mode: generate tokens and save reference stats to JSON
fn cmd_xray_record(model: &str, ref_path: &str, args: &XrayArgs) -> Result<()> {
    use talu::xray::{ReferenceRecorderHandle, VerifyCaptureHandle};

    let prompt_text = if args.prompt.is_empty() {
        "xray".to_string()
    } else {
        args.prompt.join(" ")
    };

    println!("Recording reference for model: {}", model);
    println!("  Tokens: {}", args.tokens);
    println!("  Seed: {}", args.seed);
    println!("  Prompt: \"{}\"", prompt_text);

    let max_tokens = args.tokens;

    // Create reference recorder (use seed from args)
    let recorder = ReferenceRecorderHandle::new(model, args.seed, 1.0, max_tokens)?;

    // Create verify capture in recording mode
    let verify_cap = VerifyCaptureHandle::new_recording(&recorder)?;

    // Create backend and chat
    let backend = create_backend_for_model(model, None)?;
    let chat = ChatHandle::new(None)?;

    // Enable capture AFTER backend initialization to skip warmup passes
    verify_cap.enable();

    let content = vec![talu::router::ContentPart::Text(prompt_text)];

    let cfg = talu::router::GenerateConfig {
        temperature: 1.0,
        max_tokens: max_tokens as usize,
        seed: args.seed,
        ..Default::default()
    };

    let result = talu::router::generate(&chat, &content, &backend, &cfg);

    VerifyCaptureHandle::disable();
    synchronize_backend(&backend)?;

    let result = result?;
    if result.error_code() != 0 {
        let message = last_error_message().unwrap_or_else(|| "generation failed".to_string());
        return Err(anyhow!("Error: {} (code {})", message, result.error_code()));
    }

    if let Some(_text) = result.text() {
        println!("\nGenerated {} tokens.", result.completion_tokens());
    } else {
        println!("Warning: No text generated, reference will have empty token transcript");
    }

    // Finalize and save
    println!("Finalizing reference...");
    let full_dump_path = format!("{}.layers_full.npz", ref_path);
    verify_cap.save_full_npz(&full_dump_path)?;
    let reference = recorder.finalize()?;
    reference.save(ref_path)?;

    let _ = ref_path;
    let _ = full_dump_path;
    println!("✓ Reference cache updated");
    Ok(())
}

fn run_verify_pass_with_backend(
    backend: &talu::InferenceBackend,
    prompt_text: &str,
    seed: u64,
    tolerance: f32,
    reference_path: &str,
    max_tokens: usize,
    teacher_forcing: bool,
    full_npz_path: Option<&Path>,
    point_mask_override: Option<u64>,
    exact_emission_override: Option<ExactEmissionOverride>,
) -> Result<(bool, Option<String>)> {
    use talu::xray::{ReferenceDataHandle, ReferenceVerifierHandle, VerifyCaptureHandle};

    let _override_guard = VerifyOverrideGuard::new(
        teacher_forcing,
        full_npz_path.is_some(),
        point_mask_override,
        exact_emission_override,
    );

    let cfg = talu::router::GenerateConfig {
        temperature: 1.0,
        max_tokens,
        seed,
        ..Default::default()
    };
    let reference = ReferenceDataHandle::load(reference_path)?;
    let verifier = ReferenceVerifierHandle::new(&reference, tolerance)?;
    let verify_cap = VerifyCaptureHandle::new_verification(&verifier, Some("/tmp/panic_dumps"))?;
    let chat = ChatHandle::new(None)?;
    let content = vec![talu::router::ContentPart::Text(prompt_text.to_string())];
    let active_run_guard = ActiveVerifyRunGuard::activate(teacher_forcing, &verifier, &verify_cap);

    let pass_outcome = {
        let generation = talu::router::generate(&chat, &content, &backend, &cfg);

        drop(active_run_guard);
        let sync_result = synchronize_backend(&backend);
        let result = generation?;
        sync_result?;
        if let Some(path) = full_npz_path {
            if let Some(path_str) = path.to_str() {
                verify_cap.save_full_npz(path_str)?;
            }
        }

        let outcome = if result.error_code() != 0 {
            if verifier.has_diverged() {
                let msg = verifier.finish().err().map(|err| err.to_string());
                Ok((true, msg))
            } else {
                let message = last_error_message().unwrap_or_else(|| "generation failed".to_string());
                Err(anyhow!("Error: {} (code {})", message, result.error_code()))
            }
        } else if !teacher_forcing {
            if verifier.has_diverged() {
                let msg = verifier.finish().err().map(|err| err.to_string());
                Ok((true, msg))
            } else if verifier.get_next_token().is_some() {
                Ok((
                    true,
                    Some(
                        "Token transcript mismatch: generation ended before consuming reference transcript"
                            .to_string(),
                    ),
                ))
            } else {
                Ok((false, None))
            }
        } else {
            match verifier.finish() {
                Ok(()) => Ok((verifier.has_diverged(), None)),
                Err(err) => {
                    if verifier.has_diverged() {
                        Ok((true, Some(err.to_string())))
                    } else {
                        Err(err.into())
                    }
                }
            }
        };
        drop(result);
        outcome
    };

    // XRAY teardown contract:
    // 1. Disable capture / teacher forcing.
    // 2. Synchronize backend work.
    // 3. Finalize and drop xray verifier/capture state.
    // 4. Drop router state.
    drop(verify_cap);
    drop(verifier);
    drop(reference);
    drop(content);
    drop(chat);

    pass_outcome
}

fn run_verify_pass(
    model: &str,
    prompt_text: &str,
    seed: u64,
    tolerance: f32,
    reference_path: &str,
    max_tokens: usize,
    teacher_forcing: bool,
    full_npz_path: Option<&Path>,
    point_mask_override: Option<u64>,
    exact_emission_override: Option<ExactEmissionOverride>,
) -> Result<(bool, Option<String>)> {
    // Each verify pass must start from a clean inference backend. Reusing the
    // same backend across free-run and teacher-forced phases leaks backend
    // state between logically independent inferences and can create false
    // phase-2 divergences or crashes.
    let backend = create_backend_for_model(model, None)?;
    let pass_outcome = run_verify_pass_with_backend(
        &backend,
        prompt_text,
        seed,
        tolerance,
        reference_path,
        max_tokens,
        teacher_forcing,
        full_npz_path,
        point_mask_override,
        exact_emission_override,
    );

    if should_process_scope_backend_teardown() {
        std::mem::forget(backend);
    } else {
        drop(backend);
    }

    pass_outcome
}

/// Verification mode: load reference and verify generation matches.
///
/// XRAY VERIFY PROCEDURE (contract):
/// 1) Phase 1 free-run token parity:
///    - Run generation normally and compare selected token IDs.
///    - This is the production-behavior tripwire.
/// 2) Phase 2 teacher-forced numeric localization:
///    - Replay on the reference transcript and compare captured checkpoints.
///    - This localizes first numeric divergence without sequence drift.
///
/// XRAY VERIFY RULES (non-negotiable):
/// - Verify is observability-only; it MUST NOT change backend route selection.
/// - Verify is observability-only; it MUST NOT change fusion policy.
/// - Verify is observability-only; it MUST NOT change kernel selection/arithmetic.
/// - Teacher forcing may control token progression only; it must not alter math paths.
fn cmd_xray_verify(model: &str, ref_path: &str, tolerance: f32, args: &XrayArgs) -> Result<()> {
    let reference_json_path = ref_path.to_string();
    let golden_full_npz = PathBuf::from(format!("{}.layers_full.npz", reference_json_path));

    let prompt_text = if args.prompt.is_empty() {
        "xray".to_string()
    } else {
        args.prompt.join(" ")
    };

    println!("Verifying model {} against cached CPU reference", model);
    println!("  Tokens: {}", args.tokens);
    println!("  Seed: {}", args.seed);
    println!("  Tolerance: {}", tolerance);
    println!("  Prompt: \"{}\"", prompt_text);

    if args.verify_phase1_only {
        return run_phase1_only_in_current_process(
            model,
            &prompt_text,
            tolerance,
            &reference_json_path,
            args,
        );
    }

    if args.verify_phase2_only {
        let phase2_max_tokens = args
            .verify_phase2_max_tokens
            .ok_or_else(|| anyhow!("missing internal --verify-phase2-max-tokens"))?
            as usize;
        return run_phase2_only_in_current_process(
            model,
            &prompt_text,
            tolerance,
            &reference_json_path,
            args,
            phase2_max_tokens,
        );
    }

    let checkpoint_target = args
        .verify_checkpoint
        .as_deref()
        .map(parse_verify_checkpoint_target)
        .transpose()?;
    if let Some(target) = checkpoint_target {
        if args.verify_checkpoint_stats_only {
            let max_tokens = (target.token as usize).saturating_add(1).max(1);
            let target_point_mask = point_mask_for_name(&target.point)?;
            let filtered_reference = write_filtered_reference_json(&reference_json_path, &target)?;
            let filtered_reference_path = filtered_reference
                .path()
                .to_str()
                .ok_or_else(|| anyhow!("filtered reference path is not valid UTF-8"))?;
            let exact_emission_override = exact_emission_override_for_target(&target)?;
            println!(
                "Targeted checkpoint stats verification ({}, max_tokens={}):",
                args.verify_checkpoint
                    .as_deref()
                    .unwrap_or("<parsed checkpoint>"),
                max_tokens
            );
            let (diverged, msg) = run_verify_pass(
                model,
                &prompt_text,
                args.seed,
                tolerance,
                filtered_reference_path,
                max_tokens,
                true,
                None,
                Some(target_point_mask),
                exact_emission_override,
            )?;
            if diverged {
                println!("✗ Verification FAILED");
                if let Some(msg) = msg {
                    print_first_divergence_report(&msg);
                } else {
                    println!("token=?, layer=?, point=? -> FAILED");
                    println!();
                    println!("details:");
                    println!("  divergence detected but no structured message was provided");
                }
                return Err(anyhow!(
                    "Verification failed: checkpoint divergence detected"
                ));
            }

            let layer_label = match target.layer {
                VerifyLayerTarget::Global => "global".to_string(),
                VerifyLayerTarget::Layer(layer) => layer.to_string(),
            };
            println!(
                "token={}, layer={}, point={}, pos={} -> PASSED",
                target.token,
                layer_label,
                target.point,
                target.position.unwrap_or(0)
            );
            println!();
            println!("details:");
            println!("  kind=stats");
            println!("  checkpoint matched within tolerance");
            return Ok(());
        }

        println!(
            "Targeted checkpoint verification ({}, max_tokens={}):",
            args.verify_checkpoint
                .as_deref()
                .unwrap_or("<parsed checkpoint>"),
            (target.token as usize).saturating_add(1).max(1)
        );
        let golden_tensors = load_npz_f32(&golden_full_npz)?;
        let result = run_targeted_checkpoint_compare(
            model,
            &prompt_text,
            args.seed,
            tolerance,
            &reference_json_path,
            &golden_full_npz,
            &golden_tensors,
            &target,
        )?;
        print_targeted_checkpoint_report(model, &prompt_text, &golden_full_npz, &result);
        if result.diverged {
            return Err(anyhow!(
                "Verification failed: checkpoint divergence at {} (rel_rms={:.6})",
                result.selected.key,
                result.rel_rms
            ));
        }
        let _ = std::fs::remove_file(&result.candidate_sidecar);
        return Ok(());
    }

    run_verify_in_current_process(model, &prompt_text, tolerance, &reference_json_path, args)
}

struct Phase1ProcessResult {
    report: String,
    passed: bool,
}

fn should_process_scope_backend_teardown() -> bool {
    cfg!(target_os = "macos")
}

fn run_phase1_only_in_current_process(
    model: &str,
    prompt_text: &str,
    tolerance: f32,
    reference_json_path: &str,
    args: &XrayArgs,
) -> Result<()> {
    let (diverged, divergence_msg) = run_verify_pass(
        model,
        prompt_text,
        args.seed,
        tolerance,
        reference_json_path,
        args.tokens as usize,
        false,
        None,
        None,
        None,
    )?;

    if diverged {
        if let Some(msg) = &divergence_msg {
            print_first_divergence_report(msg);
        } else {
            println!("token=?, layer=?, point=? -> FAILED");
            println!();
            println!("details:");
            println!("  divergence detected but no structured message was provided");
        }
        return Err(anyhow!("Verification failed: divergence detected"));
    }

    println!("token=all, layer=all, point=all -> PASSED");
    println!();
    println!("details:");
    println!(
        "  verified tokens={} seed={} tolerance={}",
        args.tokens, args.seed, tolerance
    );
    Ok(())
}

fn run_phase1_process(
    exe_path: &Path,
    model: &str,
    prompt_text: &str,
    reference_json_path: &str,
    tolerance: f32,
    args: &XrayArgs,
) -> Result<Phase1ProcessResult> {
    let mut command = Command::new(exe_path);
    apply_verify_child_environment(&mut command);
    command
        .arg("xray")
        .arg(model)
        .arg("--verify")
        .arg("--verify-phase1-only")
        .arg("--verify-reference-path")
        .arg(reference_json_path)
        .arg("--tokens")
        .arg(args.tokens.to_string())
        .arg("--seed")
        .arg(args.seed.to_string())
        .arg("--tolerance")
        .arg(tolerance.to_string())
        .arg(prompt_text);
    let (status, stdout, stderr) =
        run_command_capture_to_temp_files(&mut command, "phase 1 verify child process")?;
    let combined = if stderr.trim().is_empty() {
        stdout
    } else if stdout.trim().is_empty() {
        stderr
    } else {
        format!("{stdout}\n{stderr}")
    };
    let report = extract_verify_report(&combined).unwrap_or_default();

    if status.success() {
        return Ok(Phase1ProcessResult {
            report,
            passed: true,
        });
    }

    if !report.is_empty() {
        return Ok(Phase1ProcessResult {
            report,
            passed: false,
        });
    }

    Err(anyhow!(
        "phase 1 verify child process exited without a structured report: {}",
        status
    ))
}

fn run_verify_in_current_process(
    model: &str,
    prompt_text: &str,
    tolerance: f32,
    reference_json_path: &str,
    args: &XrayArgs,
) -> Result<()> {
    println!(
        "Phase 1/2: Free-run token parity ({} tokens, first divergence only)...",
        args.tokens
    );
    let exe_path = std::env::current_exe()
        .map_err(|err| anyhow!("failed to resolve current executable for phase 1: {err}"))?;
    let phase1 = run_phase1_process(
        &exe_path,
        model,
        prompt_text,
        reference_json_path,
        tolerance,
        args,
    )?;

    if !phase1.passed {
        println!("✗ Phase 1 FAILED");
        if phase1.report.is_empty() {
            println!("token=?, layer=?, point=? -> FAILED");
            println!();
            println!("details:");
            println!("  divergence detected but no structured message was provided");
        } else {
            println!("{}", phase1.report);
        }
        let phase2_max_tokens = parse_u32_after(&phase1.report, "token=")
            .map(|token_idx| token_idx.saturating_add(1) as usize)
            .map(|count| count.max(1))
            .unwrap_or(args.tokens as usize)
            .min(args.tokens as usize)
            .max(1);
        return run_phase2_localization_with_single_process_child(
            model,
            prompt_text,
            tolerance,
            reference_json_path,
            args,
            phase2_max_tokens,
        );
    }

    if phase1.report.is_empty() {
        println!("token=all, layer=all, point=all -> PASSED");
        println!();
        println!("details:");
        println!(
            "  verified tokens={} seed={} tolerance={}",
            args.tokens, args.seed, tolerance
        );
    } else {
        println!("{}", phase1.report);
    }
    Ok(())
}

fn run_phase2_only_in_current_process(
    model: &str,
    prompt_text: &str,
    tolerance: f32,
    reference_json_path: &str,
    args: &XrayArgs,
    phase2_max_tokens: usize,
) -> Result<()> {
    let filtered_reference = write_phase2_reference_json(reference_json_path, phase2_max_tokens)?;
    let filtered_reference_path = filtered_reference
        .path()
        .to_str()
        .ok_or_else(|| anyhow!("filtered reference path is not valid UTF-8"))?;
    println!(
        "Phase 2/2: Teacher-forced numeric localization from token=0 (max_tokens={})...",
        phase2_max_tokens
    );

    let backend = create_backend_for_model(model, None)?;
    let (diverged, msg) = match run_verify_pass_with_backend(
        &backend,
        prompt_text,
        args.seed,
        tolerance,
        filtered_reference_path,
        phase2_max_tokens,
        true,
        None,
        None,
        None,
    ) {
        Ok(result) => result,
        Err(err) => {
            if should_process_scope_backend_teardown() {
                std::mem::forget(backend);
            } else {
                drop(backend);
            }
            println!("token=?, layer=?, point=? -> FAILED");
            println!();
            println!("details:");
            println!("  kind=internal_error");
            println!("  detail={err}");
            return Err(err);
        }
    };

    if should_process_scope_backend_teardown() {
        std::mem::forget(backend);
    } else {
        drop(backend);
    }

    if diverged {
        if let Some(msg) = msg {
            println!("{}", format_first_divergence_report(&msg));
        } else {
            println!("token=?, layer=?, point=? -> FAILED");
            println!();
            println!("details:");
            println!("  divergence detected but no structured message was provided");
        }
        return Err(anyhow!("Verification failed: divergence detected"));
    }

    println!("token=?, layer=?, point=? -> FAILED");
    println!();
    println!("details:");
    println!("  divergence detected but no structured message was provided");
    println!(
        "  localization=filtered_phase2_replay_found_no_numeric_divergence checked_default_points=true max_token={}",
        phase2_max_tokens.saturating_sub(1)
    );
    Err(anyhow!(
        "Verification failed: free-run token mismatch without numeric localization"
    ))
}

fn run_phase2_localization_with_single_process_child(
    model: &str,
    prompt_text: &str,
    tolerance: f32,
    reference_json_path: &str,
    args: &XrayArgs,
    phase2_max_tokens: usize,
) -> Result<()> {
    println!(
        "Phase 2/2: Teacher-forced numeric localization from token=0 (max_tokens={})...",
        phase2_max_tokens
    );

    let exe_path = std::env::current_exe()
        .map_err(|err| anyhow!("failed to resolve current executable for phase 2: {err}"))?;
    let mut command = Command::new(exe_path);
    apply_verify_child_environment(&mut command);
    command
        .arg("xray")
        .arg(model)
        .arg("--verify")
        .arg("--verify-phase2-only")
        .arg("--verify-phase2-max-tokens")
        .arg(phase2_max_tokens.to_string())
        .arg("--verify-reference-path")
        .arg(reference_json_path)
        .arg("--tokens")
        .arg(phase2_max_tokens.to_string())
        .arg("--seed")
        .arg(args.seed.to_string())
        .arg("--tolerance")
        .arg(tolerance.to_string())
        .arg(prompt_text);

    let (status, stdout, stderr) =
        run_command_capture_to_temp_files(&mut command, "phase 2 verify child process")?;
    let combined = if stderr.trim().is_empty() {
        stdout
    } else if stdout.trim().is_empty() {
        stderr
    } else {
        format!("{stdout}\n{stderr}")
    };
    let report = extract_verify_report(&combined).unwrap_or_default();

    if !status.success() && !report.is_empty() {
        println!("✗ Verification FAILED");
        println!("{}", report);
        println!("  Check panic dumps in /tmp/panic_dumps/");
        return Err(anyhow!("Verification failed: divergence detected"));
    }

    Err(anyhow!(
        "phase 2 verify child process exited without a structured report: {}",
        status
    ))
}

fn sanitize_model_for_filename(model: &str) -> String {
    model
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '.' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn effective_prompt_text(args: &XrayArgs) -> String {
    if args.prompt.is_empty() {
        "xray".to_string()
    } else {
        args.prompt.join(" ")
    }
}

fn reference_cache_key(model: &str, prompt: &str, seed: u64, tokens: u32) -> String {
    // Cache contract marker:
    // - ties default verify cache to CPU-recorded references only
    // - invalidates stale cache bundles created before backend-scoped keys
    const CACHE_SCHEMA_VERSION: u32 = 3;
    const REFERENCE_BACKEND: &str = "cpu";
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    CACHE_SCHEMA_VERSION.hash(&mut hasher);
    REFERENCE_BACKEND.hash(&mut hasher);
    model.hash(&mut hasher);
    prompt.hash(&mut hasher);
    seed.hash(&mut hasher);
    tokens.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn reference_sidecar_path(reference_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.layers_full.npz", reference_path.display()))
}

fn reference_lock_path(reference_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.lock", reference_path.display()))
}

fn ensure_reference_cache_unlocked(reference_path: &Path) -> Result<()> {
    let lock_path = reference_lock_path(reference_path);
    if lock_path.exists() {
        return Err(anyhow!(
            "xray CPU reference cache refresh is in progress: {}",
            lock_path.display()
        ));
    }
    Ok(())
}

fn temp_reference_path(reference_path: &Path, pid: u32) -> PathBuf {
    PathBuf::from(format!("{}.tmp.{}", reference_path.display(), pid))
}

fn replace_file_atomically(src: &Path, dst: &Path) -> Result<()> {
    #[cfg(target_family = "unix")]
    {
        std::fs::rename(src, dst).map_err(|err| {
            anyhow!(
                "failed to replace cache file {} with {}: {err}",
                dst.display(),
                src.display()
            )
        })?;
    }

    #[cfg(not(target_family = "unix"))]
    {
        if dst.exists() {
            std::fs::remove_file(dst).map_err(|err| {
                anyhow!(
                    "failed to remove existing cache file {}: {err}",
                    dst.display()
                )
            })?;
        }
        std::fs::rename(src, dst).map_err(|err| {
            anyhow!(
                "failed to replace cache file {} with {}: {err}",
                dst.display(),
                src.display()
            )
        })?;
    }

    Ok(())
}

fn refresh_cpu_reference_cache(model: &str, reference_path: &Path, args: &XrayArgs) -> Result<()> {
    // Refresh must never leave a half-written cache bundle behind. Record into a
    // temp bundle in the same directory, then replace the final bundle only
    // after both JSON and NPZ are present.
    let _lock = CacheRefreshLock::acquire(reference_path)?;
    let mut temp_bundle = TempReferenceBundle::new(reference_path);

    let exe_path = std::env::current_exe().map_err(|err| {
        anyhow!("failed to resolve current executable for CPU cache refresh: {err}")
    })?;

    let mut cmd = Command::new(&exe_path);
    cmd.env("BACKEND", "cpu");
    cmd.arg("xray")
        .arg(model)
        .arg("--record-reference")
        .arg(
            temp_bundle
                .reference_path
                .to_str()
                .ok_or_else(|| anyhow!("reference path is not valid UTF-8"))?,
        )
        .arg("--tokens")
        .arg(args.tokens.to_string())
        .arg("--seed")
        .arg(args.seed.to_string());

    if !args.prompt.is_empty() {
        for prompt_part in &args.prompt {
            cmd.arg(prompt_part);
        }
    }

    let output = cmd
        .output()
        .map_err(|err| anyhow!("failed to spawn CPU reference recorder: {err}"))?;
    if output.status.success() {
        temp_bundle.persist(reference_path)?;
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    Err(anyhow!(
        "CPU reference refresh failed (status={}):\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout.trim(),
        stderr.trim()
    ))
}

fn default_dev_reference_path(model: &str, args: &XrayArgs) -> Result<PathBuf> {
    let home = dirs::home_dir()
        .ok_or_else(|| anyhow!("failed to resolve home directory for xray dev cache path"))?;
    let dir = home.join(".cache").join("talu").join("dev");
    std::fs::create_dir_all(&dir)?;
    let prompt_text = effective_prompt_text(args);
    let key = reference_cache_key(model, &prompt_text, args.seed, args.tokens);
    Ok(dir.join(format!(
        "xray_{}_{}.json",
        sanitize_model_for_filename(model),
        key
    )))
}

fn cmd_xray_verify_default(model: &str, args: &XrayArgs) -> Result<()> {
    let reference_path = if let Some(path) = &args.verify_reference_path {
        PathBuf::from(path)
    } else {
        default_dev_reference_path(model, args)?
    };
    let sidecar_path = reference_sidecar_path(&reference_path);
    println!("Using cache bundle: {}", sidecar_path.display());
    let refresh_cache = args.no_cache || !reference_path.exists() || !sidecar_path.exists();
    if refresh_cache {
        if args.no_cache {
            println!("Refreshing CPU reference cache (--no-cache)...");
        } else {
            println!("CPU reference cache missing, generating it first...");
        }
        refresh_cpu_reference_cache(model, &reference_path, args)?;
    } else {
        ensure_reference_cache_unlocked(&reference_path)?;
    }

    cmd_xray_verify(
        model,
        reference_path
            .to_str()
            .ok_or_else(|| anyhow!("reference path is not valid UTF-8"))?,
        args.tolerance,
        args,
    )
}

fn parse_u32_after(msg: &str, marker: &str) -> Option<u32> {
    let start = msg.find(marker)? + marker.len();
    let end = msg[start..]
        .find(|ch: char| !ch.is_ascii_digit())
        .map(|offset| start + offset)
        .unwrap_or(msg.len());
    msg[start..end].parse::<u32>().ok()
}

fn parse_f32_after(msg: &str, marker: &str) -> Option<f32> {
    let start = msg.find(marker)? + marker.len();
    let end = msg[start..]
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == '+'))
        .map(|offset| start + offset)
        .unwrap_or(msg.len());
    msg[start..end].parse::<f32>().ok()
}

fn parse_point_after(msg: &str, marker: &str) -> Option<String> {
    let start = msg.find(marker)? + marker.len();
    let end = msg[start..]
        .find(|ch: char| ch == ':' || ch == ';' || ch.is_whitespace())
        .map(|offset| start + offset)
        .unwrap_or(msg.len());
    Some(msg[start..end].to_string())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum VerifyLayerTarget {
    Global,
    Layer(u16),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VerifyCheckpointTarget {
    token: u32,
    layer: VerifyLayerTarget,
    point: String,
    position: Option<u32>,
}

#[derive(Debug, Clone)]
struct ParsedCheckpointKey {
    key: String,
    token: u32,
    position: u32,
    scope: String,
    layer_index: u16,
    point: String,
}

#[derive(Debug)]
struct TargetedCheckpointResult {
    selected: ParsedCheckpointKey,
    diverged: bool,
    expected_len: usize,
    actual_len: usize,
    expected_rms: f64,
    actual_rms: f64,
    abs_rms_diff: f64,
    rel_rms: f64,
    max_abs: f64,
    detail: Option<String>,
    candidate_sidecar: PathBuf,
}

fn parse_verify_checkpoint_target(raw: &str) -> Result<VerifyCheckpointTarget> {
    let parts: Vec<&str> = raw.split(':').collect();
    if parts.len() != 3 && parts.len() != 4 {
        return Err(anyhow!(
            "invalid --verify-checkpoint '{}': expected '<token>:<layer|global>:<point>[:<pos>]'",
            raw
        ));
    }
    let token = parts[0].parse::<u32>().map_err(|_| {
        anyhow!(
            "invalid token '{}' in --verify-checkpoint '{}'",
            parts[0],
            raw
        )
    })?;
    let layer = if parts[1].eq_ignore_ascii_case("global") {
        VerifyLayerTarget::Global
    } else {
        VerifyLayerTarget::Layer(parts[1].parse::<u16>().map_err(|_| {
            anyhow!(
                "invalid layer '{}' in --verify-checkpoint '{}': use layer index or 'global'",
                parts[1],
                raw
            )
        })?)
    };
    let point = parts[2].trim();
    if point.is_empty() {
        return Err(anyhow!(
            "invalid --verify-checkpoint '{}': point must not be empty",
            raw
        ));
    }
    let position = if parts.len() == 4 {
        Some(parts[3].parse::<u32>().map_err(|_| {
            anyhow!(
                "invalid position '{}' in --verify-checkpoint '{}'",
                parts[3],
                raw
            )
        })?)
    } else {
        None
    };
    Ok(VerifyCheckpointTarget {
        token,
        layer,
        point: point.to_string(),
        position,
    })
}

fn point_index_for_name(point: &str) -> Result<u8> {
    let point_idx = match point {
        "embed_tokens" => 0u32,
        "embed_pos" => 1,
        "layer_input" => 2,
        "layer_attn_norm" => 3,
        "attn.q" => 4,
        "attn.k" => 5,
        "attn.v" => 6,
        "attn.qk" => 7,
        "attn.weights" => 8,
        "attn.out" => 9,
        "layer_ffn_norm" => 10,
        "ffn.gate" => 11,
        "ffn.up" => 12,
        "ffn.act" => 13,
        "ffn.down" => 14,
        "block.out" => 15,
        "mamba.out" => 16,
        "conv.in_proj" => 17,
        "conv.conv" => 18,
        "conv.out_proj" => 19,
        "final_norm" => 20,
        "lm_head" => 21,
        "logits_scaled" => 22,
        "logits_ready" => 23,
        "token_select" => 24,
        "ffn.act.map" => 25,
        "ffn.act.mix" => 26,
        "gdelta.in_proj" => 27,
        "gdelta.conv" => 28,
        "gdelta.ssm" => 29,
        "gdelta.norm" => 30,
        "gdelta.out" => 31,
        _ => {
            return Err(anyhow!(
                "unsupported checkpoint point '{}': targeted verify currently supports built-in xray points only",
                point
            ));
        }
    };
    Ok(point_idx as u8)
}

fn point_mask_for_name(point: &str) -> Result<u64> {
    let point_idx = point_index_for_name(point)? as u32;
    Ok(1u64 << point_idx)
}

fn parse_checkpoint_key(key: &str) -> Option<ParsedCheckpointKey> {
    let key = normalize_npz_entry_key(key);
    let (token, position, scope, layer_index, point) = {
        let rest = key.strip_prefix("tok")?;
        let pos_marker = rest.find("_pos")?;
        let token = rest[..pos_marker].parse::<u32>().ok()?;
        let rest = &rest[pos_marker + "_pos".len()..];
        let scope_marker = rest.find('_')?;
        let position = rest[..scope_marker].parse::<u32>().ok()?;
        let rest = &rest[scope_marker + 1..];
        let layer_marker = rest.find('_')?;
        let scope = &rest[..layer_marker];
        if scope != "layer" && scope != "global" {
            return None;
        }
        let rest = &rest[layer_marker + 1..];
        let point_marker = rest.find('_')?;
        let layer_index = rest[..point_marker].parse::<u16>().ok()?;
        let point = &rest[point_marker + 1..];
        if point.is_empty() {
            return None;
        }
        (
            token,
            position,
            scope.to_string(),
            layer_index,
            point.to_string(),
        )
    };
    Some(ParsedCheckpointKey {
        key,
        token,
        position,
        scope,
        layer_index,
        point,
    })
}

fn select_checkpoint_key(
    tensors: &BTreeMap<String, Vec<f32>>,
    target: &VerifyCheckpointTarget,
    candidate: Option<&BTreeMap<String, Vec<f32>>>,
) -> Result<ParsedCheckpointKey> {
    let mut matches: Vec<ParsedCheckpointKey> = tensors
        .keys()
        .filter_map(|key| parse_checkpoint_key(key))
        .filter(|parsed| {
            if parsed.token != target.token {
                return false;
            }
            if parsed.point != target.point {
                return false;
            }
            match target.layer {
                VerifyLayerTarget::Global => {
                    if parsed.scope != "global" {
                        return false;
                    }
                }
                VerifyLayerTarget::Layer(layer_idx) => {
                    if parsed.scope != "layer" || parsed.layer_index != layer_idx {
                        return false;
                    }
                }
            }
            if let Some(position) = target.position {
                if parsed.position != position {
                    return false;
                }
            }
            true
        })
        .collect();

    matches.sort_by_key(|entry| entry.position);
    if matches.is_empty() {
        let layer_label = match target.layer {
            VerifyLayerTarget::Global => "global".to_string(),
            VerifyLayerTarget::Layer(idx) => idx.to_string(),
        };
        return Err(anyhow!(
            "no checkpoint found for token={} layer={} point={}{}",
            target.token,
            layer_label,
            target.point,
            target
                .position
                .map(|pos| format!(" pos={}", pos))
                .unwrap_or_default()
        ));
    }
    if target.position.is_none() {
        if let Some(candidate_tensors) = candidate {
            if let Some(common) = matches
                .iter()
                .find(|entry| candidate_tensors.contains_key(&entry.key))
            {
                return Ok(common.clone());
            }
        }
    }

    Ok(matches[0].clone())
}

fn is_default_phase2_point_name(point: &str) -> bool {
    // Keep this list in lock-step with
    // core/src/xray/verify.zig:VerifyCapture.defaultVerificationPointSet().
    matches!(
        point,
        "layer_attn_norm" | "block.out" | "lm_head" | "logits_ready" | "token_select" | "gdelta.out"
    )
}

fn reference_record_matches_target(
    record: &serde_json::Value,
    target: &VerifyCheckpointTarget,
) -> bool {
    let token_idx = match record.get("token_idx").and_then(|value| value.as_u64()) {
        Some(value) => value as u32,
        None => return false,
    };
    if token_idx != target.token {
        return false;
    }

    let layer = match record.get("layer").and_then(|value| value.as_u64()) {
        Some(value) => value as u16,
        None => return false,
    };
    match target.layer {
        VerifyLayerTarget::Global => {
            if layer != u16::MAX {
                return false;
            }
        }
        VerifyLayerTarget::Layer(layer_idx) => {
            if layer != layer_idx {
                return false;
            }
        }
    }

    let point = match record.get("point").and_then(|value| value.as_str()) {
        Some(value) => value,
        None => return false,
    };
    if point != target.point {
        return false;
    }

    if let Some(expected_position) = target.position {
        let position = match record.get("position").and_then(|value| value.as_u64()) {
            Some(value) => value as u32,
            None => return false,
        };
        if position != expected_position {
            return false;
        }
    }

    true
}

fn write_filtered_reference_json(
    reference_json_path: &str,
    target: &VerifyCheckpointTarget,
) -> Result<ScopedTempJson> {
    let file = File::open(reference_json_path)?;
    let mut reference: serde_json::Value = serde_json::from_reader(file)?;
    let stats = reference
        .get_mut("stats")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| anyhow!("reference JSON missing 'stats' array"))?;

    let filtered_stats: Vec<serde_json::Value> = stats
        .iter()
        .filter(|record| reference_record_matches_target(record, target))
        .cloned()
        .collect();
    if filtered_stats.is_empty() {
        return Err(anyhow!(
            "checkpoint {} not found in reference JSON {}",
            format_verify_checkpoint_arg(target),
            reference_json_path
        ));
    }

    *stats = filtered_stats;

    let (temp_json, mut temp_file) = ScopedTempJson::create("talu_xray_filtered_ref")?;
    serde_json::to_writer(&mut temp_file, &reference)?;
    temp_file.flush()?;
    Ok(temp_json)
}

fn write_phase2_reference_json(
    reference_json_path: &str,
    max_tokens: usize,
) -> Result<ScopedTempJson> {
    let file = File::open(reference_json_path)?;
    let mut reference: serde_json::Value = serde_json::from_reader(file)?;

    if let Some(metadata) = reference.get_mut("metadata").and_then(|value| value.as_object_mut()) {
        metadata.insert("max_tokens".to_string(), serde_json::Value::from(max_tokens as u64));
    }

    if let Some(tokens) = reference.get_mut("tokens").and_then(|value| value.as_array_mut()) {
        tokens.truncate(max_tokens);
    }

    let stats = reference
        .get_mut("stats")
        .and_then(|value| value.as_array_mut())
        .ok_or_else(|| anyhow!("reference JSON missing 'stats' array"))?;

    let filtered_stats: Vec<serde_json::Value> = stats
        .iter()
        .filter(|record| {
            let token_idx = match record.get("token_idx").and_then(|value| value.as_u64()) {
                Some(value) => value as usize,
                None => return false,
            };
            if token_idx >= max_tokens {
                return false;
            }

            let point = match record.get("point").and_then(|value| value.as_str()) {
                Some(value) => value,
                None => return false,
            };
            is_default_phase2_point_name(point)
        })
        .cloned()
        .collect();

    if filtered_stats.is_empty() {
        return Err(anyhow!(
            "reference JSON {} does not contain any default Phase 2 checkpoints within max_tokens={}",
            reference_json_path,
            max_tokens
        ));
    }

    *stats = filtered_stats;

    let (temp_json, mut temp_file) = ScopedTempJson::create("talu_xray_phase2_ref")?;
    serde_json::to_writer(&mut temp_file, &reference)?;
    temp_file.flush()?;
    Ok(temp_json)
}

fn run_targeted_checkpoint_compare(
    model: &str,
    prompt_text: &str,
    seed: u64,
    tolerance: f32,
    reference_json_path: &str,
    golden_full_npz: &Path,
    golden_tensors: &BTreeMap<String, Vec<f32>>,
    target: &VerifyCheckpointTarget,
) -> Result<TargetedCheckpointResult> {
    let max_tokens = (target.token as usize).saturating_add(1).max(1);
    let target_point_mask = point_mask_for_name(&target.point)?;
    let exact_emission_override = exact_emission_override_for_target(target)?;
    let candidate_npz = std::env::temp_dir().join(format!(
        "talu_xray_verify_candidate_{}_{}_{}_{}.layers_full.npz",
        std::process::id(),
        target.token,
        match target.layer {
            VerifyLayerTarget::Global => "global".to_string(),
            VerifyLayerTarget::Layer(layer) => layer.to_string(),
        },
        target.point.replace('.', "_"),
    ));

    let _ = run_verify_pass(
        model,
        prompt_text,
        seed,
        tolerance,
        reference_json_path,
        max_tokens,
        true,
        Some(&candidate_npz),
        Some(target_point_mask),
        exact_emission_override,
    )?;

    if !golden_full_npz.exists() {
        return Err(anyhow!(
            "golden full sidecar missing: {}. Re-record CPU reference first",
            golden_full_npz.display()
        ));
    }

    let candidate_tensors = load_npz_f32(&candidate_npz)?;
    let selected = select_checkpoint_key(golden_tensors, target, Some(&candidate_tensors))?;
    let expected = golden_tensors.get(&selected.key).ok_or_else(|| {
        anyhow!(
            "internal error: selected golden checkpoint '{}' not found",
            selected.key
        )
    })?;

    let actual = match candidate_tensors.get(&selected.key) {
        Some(values) => values,
        None => {
            return Ok(TargetedCheckpointResult {
                selected,
                diverged: true,
                expected_len: expected.len(),
                actual_len: 0,
                expected_rms: rms(expected),
                actual_rms: 0.0,
                abs_rms_diff: rms(expected),
                rel_rms: f64::INFINITY,
                max_abs: 0.0,
                detail: Some("checkpoint missing from candidate sidecar".to_string()),
                candidate_sidecar: candidate_npz,
            });
        }
    };

    let expected_rms = rms(expected);
    let actual_rms = rms(actual);
    let abs_rms_diff = (actual_rms - expected_rms).abs();
    let (rel_rms, max_abs) = rel_rms_and_max_abs(expected, actual);
    let len_mismatch = expected.len() != actual.len();
    let diverged = len_mismatch || exceeds_numeric_tolerance(rel_rms, abs_rms_diff, tolerance);

    Ok(TargetedCheckpointResult {
        selected,
        diverged,
        expected_len: expected.len(),
        actual_len: actual.len(),
        expected_rms,
        actual_rms,
        abs_rms_diff,
        rel_rms,
        max_abs: max_abs.into(),
        detail: if len_mismatch {
            Some("length mismatch".to_string())
        } else {
            None
        },
        candidate_sidecar: candidate_npz,
    })
}

fn format_verify_checkpoint_arg(target: &VerifyCheckpointTarget) -> String {
    match target.layer {
        VerifyLayerTarget::Global => format!(
            "{}:global:{}:{}",
            target.token,
            target.point,
            target.position.unwrap_or(0)
        ),
        VerifyLayerTarget::Layer(layer) => format!(
            "{}:{}:{}:{}",
            target.token,
            layer,
            target.point,
            target.position.unwrap_or(0)
        ),
    }
}

fn exact_emission_override_for_target(
    target: &VerifyCheckpointTarget,
) -> Result<Option<ExactEmissionOverride>> {
    let position = match target.position {
        Some(value) => value,
        None => return Ok(None),
    };
    let layer = match target.layer {
        VerifyLayerTarget::Global => u16::MAX,
        VerifyLayerTarget::Layer(layer_idx) => layer_idx,
    };
    Ok(Some(ExactEmissionOverride {
        point: point_index_for_name(&target.point)?,
        layer,
        position,
    }))
}

fn extract_verify_report(output: &str) -> Option<String> {
    let lines: Vec<&str> = output.lines().collect();
    let start = lines
        .iter()
        .position(|line| line.starts_with("token="))
        .or_else(|| lines.iter().position(|line| line.starts_with("details:")))?;
    Some(lines[start..].join("\n"))
}

fn run_command_capture_to_temp_files(
    command: &mut Command,
    context: &str,
) -> Result<(std::process::ExitStatus, String, String)> {
    let stdout_path = std::env::temp_dir().join(format!(
        "talu_xray_{}_{}_stdout.log",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let stderr_path = std::env::temp_dir().join(format!(
        "talu_xray_{}_{}_stderr.log",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let stdout_writer = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stdout_path)
        .map_err(|err| anyhow!("failed to create stdout temp file for {context}: {err}"))?;
    let stderr_writer = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stderr_path)
        .map_err(|err| anyhow!("failed to create stderr temp file for {context}: {err}"))?;

    let status = command
        .stdout(stdout_writer)
        .stderr(stderr_writer)
        .status()
        .map_err(|err| anyhow!("failed to run {context}: {err}"))?;

    let stdout = std::fs::read(&stdout_path)
        .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
        .map_err(|err| anyhow!("failed to read stdout temp file for {context}: {err}"))?;
    let stderr = std::fs::read(&stderr_path)
        .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
        .map_err(|err| anyhow!("failed to read stderr temp file for {context}: {err}"))?;
    let _ = std::fs::remove_file(&stdout_path);
    let _ = std::fs::remove_file(&stderr_path);

    Ok((status, stdout, stderr))
}

fn apply_verify_child_environment(command: &mut Command) {
    if let Ok(backend) = std::env::var("BACKEND") {
        command.env("BACKEND", backend);
    }
}

fn print_targeted_checkpoint_report(
    model: &str,
    prompt_text: &str,
    golden_full_npz: &Path,
    result: &TargetedCheckpointResult,
) {
    let layer_label = if result.selected.scope == "global" {
        "global".to_string()
    } else {
        result.selected.layer_index.to_string()
    };

    println!(
        "token={}, layer={}, point={}, pos={} -> {}",
        result.selected.token,
        layer_label,
        result.selected.point,
        result.selected.position,
        if result.diverged { "FAILED" } else { "PASSED" }
    );
    println!();
    println!("details:");
    println!("  kind=checkpoint");
    println!("  checkpoint={}", result.selected.key);
    println!(
        "  expected_len={} actual_len={}",
        result.expected_len, result.actual_len
    );
    println!(
        "  metric=rms expected={:.6} actual={:.6} abs_diff={:.6} rel_diff={:.6}",
        result.expected_rms, result.actual_rms, result.abs_rms_diff, result.rel_rms
    );
    println!("  metric=max_abs_diff value={:.6}", result.max_abs);
    if let Some(detail) = &result.detail {
        println!("  detail={detail}");
    }
    if result.diverged {
        println!("  golden_sidecar={}", golden_full_npz.display());
        println!("  candidate_sidecar={}", result.candidate_sidecar.display());
        print_targeted_checkpoint_hint(model, prompt_text, &result.selected);
    }
}

fn format_first_divergence_report(msg: &str) -> String {
    if msg.starts_with("Stats divergence at token=") {
        let token = parse_u32_after(msg, "token=");
        let layer = parse_u32_after(msg, "layer=");
        let point = parse_point_after(msg, "point=");
        let position = parse_u32_after(msg, "pos=");
        let expected = parse_f32_after(msg, "RMS expected=");
        let actual = parse_f32_after(msg, "actual=");
        let mut lines = vec![format!(
            "token={}, layer={}, point={}, pos={} -> FAILED",
            token
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_string()),
            layer
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_string()),
            point.clone().unwrap_or_else(|| "?".to_string()),
            position
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_string())
        )];
        lines.push(String::new());
        lines.push("details:".to_string());
        lines.push("  kind=stats".to_string());
        if let Some(position) = position {
            lines.push(format!("  position={}", position));
        }
        match (expected, actual) {
            (Some(exp), Some(act)) => {
                let abs = (act - exp).abs();
                let rel = if exp.abs() > 0.0 {
                    abs / exp.abs()
                } else {
                    abs
                };
                lines.push(format!(
                    "  metric=rms expected={:.6} actual={:.6} abs_diff={:.6} rel_diff={:.6}",
                    exp, act, abs, rel
                ));
            }
            _ => lines.push(format!("  detail={}", msg)),
        }
        return lines.join("\n");
    }

    if msg.starts_with("Token divergence at token=") {
        let token = parse_u32_after(msg, "token=");
        let expected = parse_u32_after(msg, "expected=");
        let actual = parse_u32_after(msg, "actual=");
        let mut lines = vec![format!(
            "token={}, layer=global, point=token_select -> FAILED",
            token
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_string())
        )];
        lines.push(String::new());
        lines.push("details:".to_string());
        lines.push("  kind=token".to_string());
        if let Some(token) = token {
            lines.push(format!("  token={}", token));
        }
        if let Some(expected) = expected {
            lines.push(format!("  expected_token={}", expected));
        }
        if let Some(actual) = actual {
            lines.push(format!("  actual_token={}", actual));
        }
        return lines.join("\n");
    }

    if msg.starts_with("Missing expected stats for token=") {
        let token = parse_u32_after(msg, "token=");
        let layer = parse_u32_after(msg, "layer=");
        let point = parse_point_after(msg, "point=");
        let position = parse_u32_after(msg, "pos=");
        let mut lines = vec![format!(
            "token={}, layer={}, point={}, pos={} -> FAILED",
            token
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_string()),
            layer
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_string()),
            point.clone().unwrap_or_else(|| "?".to_string()),
            position
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_string())
        )];
        lines.push(String::new());
        lines.push("details:".to_string());
        lines.push("  kind=coverage".to_string());
        if let Some(token) = token {
            lines.push(format!("  token={}", token));
        }
        if let Some(layer) = layer {
            lines.push(format!("  layer={}", layer));
        }
        if let Some(point) = point {
            lines.push(format!("  point={}", point));
        }
        if let Some(position) = position {
            lines.push(format!("  position={}", position));
        }
        lines.push("  detail=expected checkpoint missing from backend trace".to_string());
        return lines.join("\n");
    }

    format!("token=?, layer=?, point=? -> FAILED\n\ndetails:\n  detail={}", msg)
}

fn print_first_divergence_report(msg: &str) {
    println!("{}", format_first_divergence_report(msg));
}

#[derive(Debug, Clone)]
#[cfg(test)]
enum FirstCheckpointDiffKind {
    Missing,
    LengthMismatch,
    Numeric,
}

#[derive(Debug, Clone)]
#[cfg(test)]
struct FirstCheckpointDiff {
    parsed: ParsedCheckpointKey,
    kind: FirstCheckpointDiffKind,
}

fn normalize_npz_entry_key(name: &str) -> String {
    let stem = name.strip_suffix(".npy").unwrap_or(name);
    if let Some((prefix, rest)) = stem.split_once('_') {
        if !prefix.is_empty() && prefix.chars().all(|ch| ch.is_ascii_digit()) {
            return rest.to_string();
        }
    }
    stem.to_string()
}

fn parse_npy_f32(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        return Err(anyhow!("invalid npy header"));
    }
    let major = data[6];
    let (header_len, data_offset): (usize, usize) = match major {
        1 => {
            let len = u16::from_le_bytes([data[8], data[9]]) as usize;
            (len, 10)
        }
        2 | 3 => {
            if data.len() < 12 {
                return Err(anyhow!("invalid npy header length"));
            }
            let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            (len, 12)
        }
        _ => return Err(anyhow!("unsupported npy major version {}", major)),
    };
    let header_end = data_offset + header_len;
    if header_end > data.len() {
        return Err(anyhow!("truncated npy header"));
    }
    let header = std::str::from_utf8(&data[data_offset..header_end])?;
    if !header.contains("<f4") {
        return Err(anyhow!("unsupported npy dtype (expected <f4): {}", header));
    }
    let payload = &data[header_end..];
    if payload.len() % 4 != 0 {
        return Err(anyhow!("npy payload byte-size is not divisible by 4"));
    }
    let mut values = Vec::with_capacity(payload.len() / 4);
    for chunk in payload.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}

fn load_npz_f32(path: &Path) -> Result<BTreeMap<String, Vec<f32>>> {
    let file = File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut tensors: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for idx in 0..archive.len() {
        let mut entry = archive.by_index(idx)?;
        let name = entry.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        let key = normalize_npz_entry_key(&name);
        let values = parse_npy_f32(&bytes)?;
        tensors.insert(key, values);
    }
    Ok(tensors)
}

fn rel_rms_and_max_abs(expected: &[f32], actual: &[f32]) -> (f64, f32) {
    let count = expected.len().min(actual.len());
    if count == 0 {
        return (0.0, 0.0);
    }
    let mut sum_sq_diff = 0.0f64;
    let mut sum_sq_expected = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..count {
        let exp = expected[i];
        let act = actual[i];
        let diff = (act - exp) as f64;
        sum_sq_diff += diff * diff;
        let exp64 = exp as f64;
        sum_sq_expected += exp64 * exp64;
        let abs = (act - exp).abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    let rms_diff = (sum_sq_diff / count as f64).sqrt();
    let rms_expected = (sum_sq_expected / count as f64).sqrt();
    let rel_rms = if rms_expected > 0.0 {
        rms_diff / rms_expected
    } else {
        rms_diff
    };
    (rel_rms, max_abs)
}

fn exceeds_numeric_tolerance(rel_rms: f64, abs_rms_diff: f64, tolerance: f32) -> bool {
    let tol = tolerance as f64;
    rel_rms > tol && abs_rms_diff > tol
}

fn rms(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sq = values
        .iter()
        .map(|value| {
            let v = *value as f64;
            v * v
        })
        .sum::<f64>();
    (sum_sq / values.len() as f64).sqrt()
}

#[cfg(test)]
fn find_first_checkpoint_diff(
    expected: &BTreeMap<String, Vec<f32>>,
    actual: &BTreeMap<String, Vec<f32>>,
    tolerance: f32,
) -> Option<FirstCheckpointDiff> {
    let mut ordered: Vec<ParsedCheckpointKey> = expected
        .keys()
        .filter_map(|key| parse_checkpoint_key(key))
        .collect();
    ordered.sort_by(|a, b| {
        let scope_rank = |scope: &str| match scope {
            "layer" => 0u8,
            "global" => 1u8,
            _ => 2u8,
        };
        (
            a.token,
            scope_rank(&a.scope),
            a.position,
            a.layer_index,
            a.point.as_str(),
            a.key.as_str(),
        )
            .cmp(&(
                b.token,
                scope_rank(&b.scope),
                b.position,
                b.layer_index,
                b.point.as_str(),
                b.key.as_str(),
            ))
    });

    for parsed in ordered {
        let expected_tensor = expected.get(&parsed.key)?;
        let expected_len = expected_tensor.len();
        let actual_tensor = match actual.get(&parsed.key) {
            Some(values) => values,
            None => {
                return Some(FirstCheckpointDiff {
                    parsed,
                    kind: FirstCheckpointDiffKind::Missing,
                });
            }
        };
        if actual_tensor.len() != expected_len {
            return Some(FirstCheckpointDiff {
                parsed,
                kind: FirstCheckpointDiffKind::LengthMismatch,
            });
        }
        let (rel_rms, max_abs) = rel_rms_and_max_abs(expected_tensor, actual_tensor);
        let expected_rms = rms(expected_tensor);
        let actual_rms = rms(actual_tensor);
        let abs_rms_diff = (actual_rms - expected_rms).abs();
        if exceeds_numeric_tolerance(rel_rms, abs_rms_diff, tolerance) {
            let _ = (max_abs, expected_rms, actual_rms);
            return Some(FirstCheckpointDiff {
                parsed,
                kind: FirstCheckpointDiffKind::Numeric,
            });
        }
    }
    None
}

fn print_targeted_checkpoint_hint(model: &str, prompt_text: &str, parsed: &ParsedCheckpointKey) {
    let layer_arg = if parsed.scope == "global" {
        "global".to_string()
    } else {
        parsed.layer_index.to_string()
    };
    let prompt_escaped = prompt_text.replace('\\', "\\\\").replace('"', "\\\"");
    println!();
    println!("hint:");
    println!("  rerun this checkpoint only:");
    println!(
        "  ./zig-out/bin/talu xray {} --verify --verify-checkpoint {}:{}:{}:{} \"{}\"",
        model, parsed.token, layer_arg, parsed.point, parsed.position, prompt_escaped
    );
}

#[cfg(test)]
mod tests {
    use super::{
        ensure_reference_cache_unlocked, find_first_checkpoint_diff, normalize_npz_entry_key,
        parse_checkpoint_key, parse_npy_f32, parse_verify_checkpoint_target, reference_cache_key,
        reference_lock_path, reference_sidecar_path, replace_file_atomically,
        select_checkpoint_key, temp_reference_path, write_filtered_reference_json,
        write_phase2_reference_json,
        FirstCheckpointDiffKind, VerifyCheckpointTarget, VerifyLayerTarget,
    };
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn normalize_npz_entry_key_strips_index_prefix() {
        let key = normalize_npz_entry_key("12_tok1_pos3_layer_0_layer_ffn_norm.npy");
        assert_eq!(key, "tok1_pos3_layer_0_layer_ffn_norm");
    }

    #[test]
    fn parse_npy_f32_reads_small_array() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY");
        bytes.push(1);
        bytes.push(0);
        let header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }           \n";
        let header_len = header.len() as u16;
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes.extend_from_slice(&(-2.25f32).to_le_bytes());

        let parsed = parse_npy_f32(&bytes).expect("npy parse should succeed");
        assert_eq!(parsed.len(), 2);
        assert!((parsed[0] - 1.5).abs() < 1e-6);
        assert!((parsed[1] + 2.25).abs() < 1e-6);
    }

    #[test]
    fn parse_verify_checkpoint_target_accepts_layer_and_position() {
        let parsed = parse_verify_checkpoint_target("1:0:layer_ffn_norm:16")
            .expect("checkpoint parse should succeed");
        assert_eq!(
            parsed,
            VerifyCheckpointTarget {
                token: 1,
                layer: VerifyLayerTarget::Layer(0),
                point: "layer_ffn_norm".to_string(),
                position: Some(16),
            }
        );
    }

    #[test]
    fn parse_verify_checkpoint_target_accepts_global_without_position() {
        let parsed = parse_verify_checkpoint_target("0:global:lm_head")
            .expect("checkpoint parse should succeed");
        assert_eq!(
            parsed,
            VerifyCheckpointTarget {
                token: 0,
                layer: VerifyLayerTarget::Global,
                point: "lm_head".to_string(),
                position: None,
            }
        );
    }

    #[test]
    fn write_filtered_reference_json_keeps_only_exact_target_checkpoint() {
        let mut source = tempfile::NamedTempFile::new().expect("temp reference");
        serde_json::to_writer(
            source.as_file_mut(),
            &serde_json::json!({
                "metadata": {
                    "model_name": "test",
                    "seed": 42,
                    "temperature": 1.0,
                    "max_tokens": 4
                },
                "tokens": [11, 22, 33, 44],
                "stats": [
                    {
                        "token_idx": 0,
                        "layer": 0,
                        "point": "gdelta.out",
                        "position": 15,
                        "stats": { "count": 4, "min": 0.0, "max": 1.0, "sum": 1.0, "sum_sq": 1.0, "nan_count": 0, "inf_count": 0 }
                    },
                    {
                        "token_idx": 0,
                        "layer": 22,
                        "point": "gdelta.out",
                        "position": 10,
                        "stats": { "count": 4, "min": 0.0, "max": 1.0, "sum": 1.0, "sum_sq": 1.0, "nan_count": 0, "inf_count": 0 }
                    },
                    {
                        "token_idx": 1,
                        "layer": 65535,
                        "point": "lm_head",
                        "position": 0,
                        "stats": { "count": 4, "min": 0.0, "max": 1.0, "sum": 1.0, "sum_sq": 1.0, "nan_count": 0, "inf_count": 0 }
                    }
                ]
            }),
        )
        .expect("write reference json");
        source.as_file_mut().flush().expect("flush reference json");

        let filtered = write_filtered_reference_json(
            source.path().to_str().expect("utf8 temp path"),
            &VerifyCheckpointTarget {
                token: 0,
                layer: VerifyLayerTarget::Layer(22),
                point: "gdelta.out".to_string(),
                position: Some(10),
            },
        )
        .expect("filtered reference");

        let filtered_json: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(filtered.path()).expect("open filtered"))
                .expect("parse filtered json");
        let stats = filtered_json["stats"]
            .as_array()
            .expect("filtered stats should be array");
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0]["token_idx"].as_u64(), Some(0));
        assert_eq!(stats[0]["layer"].as_u64(), Some(22));
        assert_eq!(stats[0]["point"].as_str(), Some("gdelta.out"));
        assert_eq!(stats[0]["position"].as_u64(), Some(10));
        assert_eq!(
            filtered_json["tokens"].as_array().expect("tokens array").len(),
            4
        );
    }

    #[test]
    fn write_phase2_reference_json_keeps_only_default_points_up_to_max_token() {
        let mut source = tempfile::NamedTempFile::new().expect("temp reference");
        serde_json::to_writer(
            source.as_file_mut(),
            &serde_json::json!({
                "metadata": {
                    "model_name": "test",
                    "seed": 42,
                    "temperature": 1.0,
                    "max_tokens": 6
                },
                "tokens": [11, 22, 33, 44, 55, 66],
                "stats": [
                    {
                        "token_idx": 0,
                        "layer": 0,
                        "point": "layer_attn_norm",
                        "position": 16,
                        "stats": { "count": 4, "min": 0.0, "max": 1.0, "sum": 1.0, "sum_sq": 1.0, "nan_count": 0, "inf_count": 0 }
                    },
                    {
                        "token_idx": 0,
                        "layer": 0,
                        "point": "ffn.act",
                        "position": 16,
                        "stats": { "count": 4, "min": 0.0, "max": 1.0, "sum": 1.0, "sum_sq": 1.0, "nan_count": 0, "inf_count": 0 }
                    },
                    {
                        "token_idx": 1,
                        "layer": 22,
                        "point": "gdelta.out",
                        "position": 10,
                        "stats": { "count": 4, "min": 0.0, "max": 1.0, "sum": 1.0, "sum_sq": 1.0, "nan_count": 0, "inf_count": 0 }
                    },
                    {
                        "token_idx": 2,
                        "layer": 65535,
                        "point": "lm_head",
                        "position": 0,
                        "stats": { "count": 4, "min": 0.0, "max": 1.0, "sum": 1.0, "sum_sq": 1.0, "nan_count": 0, "inf_count": 0 }
                    },
                    {
                        "token_idx": 3,
                        "layer": 65535,
                        "point": "token_select",
                        "position": 0,
                        "stats": { "count": 4, "min": 0.0, "max": 1.0, "sum": 1.0, "sum_sq": 1.0, "nan_count": 0, "inf_count": 0 }
                    }
                ]
            }),
        )
        .expect("write reference json");
        source.as_file_mut().flush().expect("flush reference json");

        let filtered = write_phase2_reference_json(
            source.path().to_str().expect("utf8 temp path"),
            2,
        )
        .expect("filtered phase2 reference");

        let filtered_json: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(filtered.path()).expect("open filtered"))
                .expect("parse filtered json");
        let stats = filtered_json["stats"]
            .as_array()
            .expect("filtered stats should be array");
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0]["point"].as_str(), Some("layer_attn_norm"));
        assert_eq!(stats[1]["point"].as_str(), Some("gdelta.out"));
        assert_eq!(stats[1]["token_idx"].as_u64(), Some(1));
        assert_eq!(
            filtered_json["tokens"].as_array().expect("tokens array").len(),
            2
        );
        assert_eq!(filtered_json["metadata"]["max_tokens"].as_u64(), Some(2));
    }

    #[test]
    fn parse_checkpoint_key_extracts_metadata() {
        let parsed = parse_checkpoint_key("42_tok1_pos16_layer_0_layer_ffn_norm.npy")
            .expect("checkpoint key should parse");
        assert_eq!(parsed.token, 1);
        assert_eq!(parsed.position, 16);
        assert_eq!(parsed.scope, "layer");
        assert_eq!(parsed.layer_index, 0);
        assert_eq!(parsed.point, "layer_ffn_norm");
        assert_eq!(parsed.key, "tok1_pos16_layer_0_layer_ffn_norm");
    }

    #[test]
    fn select_checkpoint_key_picks_lowest_position_when_unspecified() {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "tok1_pos18_layer_0_layer_ffn_norm".to_string(),
            vec![0.0f32],
        );
        tensors.insert(
            "tok1_pos16_layer_0_layer_ffn_norm".to_string(),
            vec![0.0f32],
        );
        tensors.insert(
            "tok1_pos16_layer_1_layer_ffn_norm".to_string(),
            vec![0.0f32],
        );

        let target = VerifyCheckpointTarget {
            token: 1,
            layer: VerifyLayerTarget::Layer(0),
            point: "layer_ffn_norm".to_string(),
            position: None,
        };
        let selected =
            select_checkpoint_key(&tensors, &target, None).expect("selection should succeed");
        assert_eq!(selected.key, "tok1_pos16_layer_0_layer_ffn_norm");
    }

    #[test]
    fn select_checkpoint_key_prefers_common_key_when_position_unspecified() {
        let mut golden = BTreeMap::new();
        golden.insert(
            "tok0_pos1_layer_0_layer_attn_norm".to_string(),
            vec![0.0f32],
        );
        golden.insert(
            "tok0_pos16_layer_0_layer_attn_norm".to_string(),
            vec![0.0f32],
        );
        let mut candidate = BTreeMap::new();
        candidate.insert(
            "tok0_pos16_layer_0_layer_attn_norm".to_string(),
            vec![0.0f32],
        );

        let target = VerifyCheckpointTarget {
            token: 0,
            layer: VerifyLayerTarget::Layer(0),
            point: "layer_attn_norm".to_string(),
            position: None,
        };
        let selected = select_checkpoint_key(&golden, &target, Some(&candidate))
            .expect("selection should succeed");
        assert_eq!(selected.key, "tok0_pos16_layer_0_layer_attn_norm");
    }

    #[test]
    fn find_first_checkpoint_diff_prefers_layer_before_global() {
        let mut expected = BTreeMap::new();
        expected.insert("tok0_pos0_global_0_lm_head".to_string(), vec![1.0f32]);
        expected.insert(
            "tok0_pos16_layer_0_layer_attn_norm".to_string(),
            vec![1.0f32],
        );

        let mut actual = BTreeMap::new();
        actual.insert("tok0_pos0_global_0_lm_head".to_string(), vec![1.0f32]);

        let diff = find_first_checkpoint_diff(&expected, &actual, 1e-3)
            .expect("a missing layer checkpoint should be reported");
        assert_eq!(diff.parsed.key, "tok0_pos16_layer_0_layer_attn_norm");
        assert!(matches!(diff.kind, FirstCheckpointDiffKind::Missing));
    }

    #[test]
    fn reference_cache_key_changes_when_prompt_changes() {
        let a = reference_cache_key("Qwen/Qwen3.5-0.8B-GAF4", "tell me a story", 42, 100);
        let b = reference_cache_key("Qwen/Qwen3.5-0.8B-GAF4", "tell me a poem", 42, 100);
        assert_ne!(a, b);
    }

    #[test]
    fn reference_cache_key_changes_when_seed_changes() {
        let a = reference_cache_key("Qwen/Qwen3.5-0.8B-GAF4", "tell me a story", 42, 100);
        let b = reference_cache_key("Qwen/Qwen3.5-0.8B-GAF4", "tell me a story", 43, 100);
        assert_ne!(a, b);
    }

    #[test]
    fn reference_sidecar_path_appends_npz_suffix() {
        let path = reference_sidecar_path(Path::new("/tmp/xray_qwen.json"));
        assert_eq!(
            path,
            Path::new("/tmp/xray_qwen.json.layers_full.npz").to_path_buf()
        );
    }

    #[test]
    fn reference_lock_path_appends_lock_suffix() {
        let path = reference_lock_path(Path::new("/tmp/xray_qwen.json"));
        assert_eq!(path, Path::new("/tmp/xray_qwen.json.lock").to_path_buf());
    }

    #[test]
    fn ensure_reference_cache_unlocked_rejects_live_lock() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be monotonic enough for test naming")
            .as_nanos();
        let reference_path =
            std::env::temp_dir().join(format!("talu_xray_lock_test_{}.json", nonce));
        let lock_path = reference_lock_path(&reference_path);
        std::fs::write(&lock_path, "locked").expect("write lock");

        let err = ensure_reference_cache_unlocked(&reference_path).expect_err("lock should fail");
        assert!(
            err.to_string().contains("refresh is in progress"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_file(&lock_path);
    }

    #[test]
    fn temp_reference_path_is_derived_from_reference_path() {
        let path = temp_reference_path(Path::new("/tmp/xray_qwen.json"), 1234);
        assert_eq!(
            path,
            Path::new("/tmp/xray_qwen.json.tmp.1234").to_path_buf()
        );
    }

    #[test]
    fn replace_file_atomically_replaces_destination_contents() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock should be monotonic enough for test naming")
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!(
            "talu_xray_replace_file_atomically_test_{}_{}",
            std::process::id(),
            nonce
        ));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");
        let src = temp_dir.join("src.txt");
        let dst = temp_dir.join("dst.txt");
        std::fs::write(&src, "new").expect("write src");
        std::fs::write(&dst, "old").expect("write dst");

        replace_file_atomically(&src, &dst).expect("replace should succeed");

        assert_eq!(std::fs::read_to_string(&dst).expect("read dst"), "new");
        assert!(!src.exists());
        let _ = std::fs::remove_file(&dst);
        let _ = std::fs::remove_dir(&temp_dir);
    }
}

pub(super) fn cmd_xray(args: XrayArgs) -> Result<()> {
    let model = &args.model;

    if let Some(ref_path) = &args.record_reference {
        return cmd_xray_record(model, ref_path, &args);
    }

    // Check for verification mode
    if args.verify {
        return maybe_hard_exit_after_verify(cmd_xray_verify_default(model, &args));
    }

    let decode_mode = args.output;

    let capture = if args.json {
        XrayCaptureHandle::new()?
    } else {
        XrayCaptureHandle::new_timing()?
    };
    let backend = create_backend_for_model(model, None)?;
    capture.enable();
    // Drop backend init / warmup emissions so xray reflects route execution only.
    capture.clear();
    let chat = ChatHandle::new(None)?;
    let prompt = if args.prompt.is_empty() {
        "xray".to_string()
    } else {
        args.prompt.join(" ")
    };
    let content = vec![talu::router::ContentPart::Text(prompt)];
    // --output needs 2 tokens so we get a full decode step after prefill.
    // --input (default) needs 1 token (prefill only).
    let cfg = talu::router::GenerateConfig {
        temperature: 0.0,
        max_tokens: if decode_mode { 2 } else { 1 },
        ..Default::default()
    };

    let result = talu::router::generate(&chat, &content, &backend, &cfg);
    // Stop capture before reading records so no backend thread can append
    // while we iterate the captured trace buffer.
    capture.disable();
    let result = result?;
    if result.error_code() != 0 {
        let message = last_error_message().unwrap_or_else(|| "generation failed".to_string());
        return Err(anyhow!("Error: {} (code {})", message, result.error_code()));
    }

    let all_trace = capture.get_trace();

    // Split trace into prefill/decode records.
    // For --output, keep only the final decode forward window.
    // For --input, keep full trace.
    let trace: Vec<talu::TraceRecord> = if decode_mode {
        split_decode_trace(all_trace)
    } else {
        all_trace
    };

    let record_delta_ns = derive_record_delta_ns(&trace);

    // Derive kernel usage from trace records
    let usage = derive_kernel_usage(&trace, &record_delta_ns);
    let kernel_point_usage = derive_kernel_usage_by_point(&trace, &record_delta_ns);
    let other_usage = derive_other_usage(&trace, &record_delta_ns);

    if args.json {
        let trace_records: Vec<_> = trace
            .iter()
            .map(|r| {
                json!({
                    "point": r.point,
                    "layer": if r.layer == 0xFFFF { None } else { Some(r.layer) },
                    "name": r.display_name(),
                    "shape": r.shape,
                    "dtype": r.dtype_name(),
                    "work_flops": r.work_flops,
                    "work_bytes": r.work_bytes,
                    "timestamp_ns": r.timestamp_ns,
                    "stats": {
                        "min": r.stats.min,
                        "max": r.stats.max,
                        "mean": r.stats.mean,
                        "rms": r.stats.rms,
                        "nan_count": r.stats.nan_count,
                        "inf_count": r.stats.inf_count,
                    }
                })
            })
            .collect();
        let kernels: Vec<_> = usage
            .iter()
            .map(|k| {
                json!({
                    "name": k.name,
                    "call_count": k.call_count,
                })
            })
            .collect();
        let payload = json!({
            "trace": trace_records,
            "kernels": kernels
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    // Get model info for tree view
    let model_path = resolve_model_path(model)?;
    let model_info = talu::model::describe(&model_path)?;
    let mode_label = if decode_mode {
        "output (decode)"
    } else {
        "input (prefill)"
    };
    let total_us: u64 = record_delta_ns
        .iter()
        .map(|ns| ns.unsigned_abs() / 1_000)
        .sum();
    let token_count = detect_token_count(&trace);
    let perf_hints = load_perf_hints(&model_info)?;
    if args.table {
        let kernel_shape_rows = derive_kernel_shape_table(&trace, &record_delta_ns, total_us);
        let point_rows = derive_point_table(&trace, &record_delta_ns);
        let edge_rows = derive_edge_transition_rows(&trace, &record_delta_ns);
        let timeline_segments = derive_timeline_segments(&trace, &record_delta_ns);
        print_method_table(
            &model_info,
            mode_label,
            kernel_shape_rows,
            point_rows,
            edge_rows,
            timeline_segments,
            total_us,
            token_count,
            args.debug,
            perf_hints,
        );
        return Ok(());
    }
    print_architecture_tree(
        &model_info,
        &trace,
        &record_delta_ns,
        &usage,
        &kernel_point_usage,
        &other_usage,
        perf_hints,
        mode_label,
    );
    Ok(())
}

fn load_perf_hints(model_info: &talu::ModelInfo) -> Result<Option<PerfHintsJson>> {
    if let Some(raw) = talu::model::performance_hints_json(&model_info.model_type)? {
        let hints: PerfHintsJson = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("invalid performance hints JSON: {e}"))?;
        return Ok(Some(hints));
    }
    if !model_info.architecture.is_empty() {
        if let Some(raw) = talu::model::performance_hints_json(&model_info.architecture)? {
            let hints: PerfHintsJson = serde_json::from_str(&raw)
                .map_err(|e| anyhow!("invalid performance hints JSON: {e}"))?;
            return Ok(Some(hints));
        }
    }
    Ok(None)
}

/// Format microseconds right-aligned into a fixed-width column.
fn format_us(us: u64, width: usize) -> String {
    format!("{:>w$}\u{00b5}s", us, w = width)
}

/// A row in the profile table for a single trace point.
struct ProfileRow {
    point: String,
    shape: String,
    dtype: &'static str,
    avg_us: u64,
    kernel: String,
}

fn print_architecture_tree(
    model_info: &talu::ModelInfo,
    trace: &[talu::TraceRecord],
    record_delta_ns: &[i64],
    kernels: &[KernelUsage],
    kernel_point_usage: &[KernelPointUsage],
    other_usage: &[PointUsage],
    perf_hints: Option<PerfHintsJson>,
    mode_label: &str,
) {
    // ── Separate non-layer records and layer records ──
    let mut non_layer: Vec<(usize, &talu::TraceRecord)> = Vec::new(); // (index in trace)
                                                                      // layer_idx -> Vec<(index in trace, record)>
    let mut layer_traces: std::collections::BTreeMap<u16, Vec<(usize, &talu::TraceRecord)>> =
        std::collections::BTreeMap::new();
    for (i, record) in trace.iter().enumerate() {
        if record.layer == 0xFFFF {
            non_layer.push((i, record));
        } else {
            layer_traces
                .entry(record.layer)
                .or_default()
                .push((i, record));
        }
    }

    let num_layers = model_info.num_layers as u16;

    // ── Identify groups of consecutive layers with the same signature ──
    struct LayerGroup {
        start: u16,
        end: u16, // exclusive
        signature: Vec<String>,
    }

    let layer_signature = |idx: u16| -> Vec<String> {
        layer_traces
            .get(&idx)
            .map(|records| records.iter().map(|(_, r)| r.point.clone()).collect())
            .unwrap_or_default()
    };

    let mut groups: Vec<LayerGroup> = Vec::new();
    let mut idx = 0u16;
    while idx < num_layers {
        let sig = layer_signature(idx);
        let mut end = idx + 1;
        while end < num_layers && layer_signature(end) == sig {
            end += 1;
        }
        groups.push(LayerGroup {
            start: idx,
            end,
            signature: sig,
        });
        idx = end;
    }

    // ── Build profile rows for each layer group (averaged across layers) ──
    // For each group: for each trace point position, average the delta across all layers.
    let mut group_rows: Vec<(
        /* group_idx */ usize,
        Vec<ProfileRow>,
        /* layer_avg_us */ u64,
    )> = Vec::new();

    for (gi, group) in groups.iter().enumerate() {
        let n_layers = (group.end - group.start) as usize;
        let n_points = group.signature.len();
        if n_points == 0 {
            group_rows.push((gi, Vec::new(), 0));
            continue;
        }

        // For each point position, accumulate delta_us across all layers in group
        let mut sum_us: Vec<u64> = vec![0; n_points];
        // Use first layer in group for shape/dtype/kernel (representative)
        let representative = layer_traces.get(&group.start).unwrap();

        for layer_idx in group.start..group.end {
            if let Some(records) = layer_traces.get(&layer_idx) {
                for (pos, (trace_idx, _)) in records.iter().enumerate() {
                    if pos < n_points {
                        let delta = record_delta_ns[*trace_idx];
                        sum_us[pos] += delta.unsigned_abs() / 1_000;
                    }
                }
            }
        }

        let mut rows: Vec<ProfileRow> = Vec::with_capacity(n_points);
        let mut layer_total_us: u64 = 0;
        for (pos, (_, r)) in representative.iter().enumerate() {
            let avg = sum_us[pos] / n_layers as u64;
            layer_total_us += avg;
            let kernel = r.kernel_name.as_deref().unwrap_or("");
            rows.push(ProfileRow {
                point: r.point.clone(),
                shape: r.shape_str(),
                dtype: r.dtype_name(),
                avg_us: avg,
                kernel: kernel.to_string(),
            });
        }

        group_rows.push((gi, rows, layer_total_us));
    }

    // ── Build non-layer rows (embed, final_norm, lm_head, etc.) ──
    let mut pre_layer_rows: Vec<ProfileRow> = Vec::new();
    let mut post_layer_rows: Vec<ProfileRow> = Vec::new();
    let first_layer_trace_idx = layer_traces
        .values()
        .next()
        .and_then(|v| v.first())
        .map(|(i, _)| *i)
        .unwrap_or(usize::MAX);

    for (trace_idx, r) in &non_layer {
        let delta_us = record_delta_ns[*trace_idx].unsigned_abs() / 1_000;
        let kernel = r.kernel_name.as_deref().unwrap_or("");
        let row = ProfileRow {
            point: r.point.clone(),
            shape: r.shape_str(),
            dtype: r.dtype_name(),
            avg_us: delta_us,
            kernel: kernel.to_string(),
        };
        if *trace_idx < first_layer_trace_idx {
            pre_layer_rows.push(row);
        } else {
            post_layer_rows.push(row);
        }
    }

    // ── Compute column widths ──
    let all_rows = pre_layer_rows
        .iter()
        .chain(group_rows.iter().flat_map(|(_, rows, _)| rows.iter()))
        .chain(post_layer_rows.iter());

    let mut max_point_len: usize = 0;
    let mut max_shape_len: usize = 0;
    let mut max_us: u64 = 0;
    let mut _max_kernel_len: usize = 0;
    for row in all_rows {
        max_point_len = std::cmp::max(max_point_len, row.point.len());
        max_shape_len = std::cmp::max(max_shape_len, row.shape.len());
        max_us = std::cmp::max(max_us, row.avg_us);
        _max_kernel_len = std::cmp::max(_max_kernel_len, row.kernel.len());
    }
    // Also include group totals in us_width
    let mut grand_total_us: u64 = pre_layer_rows.iter().map(|r| r.avg_us).sum::<u64>();
    for (gi, _, layer_avg) in &group_rows {
        let n_layers = (groups[*gi].end - groups[*gi].start) as u64;
        grand_total_us += layer_avg * n_layers;
    }
    grand_total_us += post_layer_rows.iter().map(|r| r.avg_us).sum::<u64>();
    max_us = std::cmp::max(max_us, grand_total_us);
    let us_width = std::cmp::max(1, format!("{}", max_us).len());

    // ── Print ──
    let model_name = model_info
        .architecture
        .split("For")
        .next()
        .unwrap_or(&model_info.model_type);
    println!(
        "{} ({}) \u{2014} {}",
        model_name, model_info.model_type, mode_label
    );
    println!();

    let print_row = |row: &ProfileRow, indent: &str| {
        let kernel_str = if row.kernel.is_empty() {
            String::new()
        } else {
            format!("  {}", row.kernel)
        };
        println!(
            "{}{:<pw$}  {:>sw$} {:<4} {}{}",
            indent,
            row.point,
            row.shape,
            row.dtype,
            format_us(row.avg_us, us_width),
            kernel_str,
            pw = max_point_len,
            sw = max_shape_len,
        );
    };

    // Pre-layer (embed, etc.)
    for row in &pre_layer_rows {
        print_row(row, "  ");
    }
    if !pre_layer_rows.is_empty() {
        println!();
    }

    // Layer groups
    for (gi, rows, layer_avg_us) in &group_rows {
        let group = &groups[*gi];
        let n_layers = group.end - group.start;
        let group_total_us = layer_avg_us * n_layers as u64;

        if n_layers == 1 {
            println!("  layer[{}]", group.start);
        } else {
            println!(
                "  layers[{}..{}] (avg of {} layers)",
                group.start, group.end, n_layers
            );
        }

        for row in rows {
            print_row(row, "    ");
        }

        // Summary line
        let separator_width = max_point_len + 2 + max_shape_len + 5 + us_width + 2;
        println!(
            "    {:>w$}",
            "\u{2500}".repeat(us_width + 2),
            w = separator_width
        );
        println!(
            "    {:>w$}",
            format!("{} avg/layer", format_us(*layer_avg_us, us_width)),
            w = separator_width
        );
        if n_layers > 1 {
            println!(
                "    {:>w$}",
                format!(
                    "{} x{} total",
                    format_us(group_total_us, us_width),
                    n_layers
                ),
                w = separator_width
            );
        }
        println!();
    }

    // Post-layer (final_norm, lm_head, etc.)
    for row in &post_layer_rows {
        print_row(row, "  ");
    }

    // Grand total
    let separator_width = 2 + max_point_len + 2 + max_shape_len + 5 + us_width + 2;
    println!(
        "  {:>w$}",
        "\u{2500}".repeat(us_width + 2),
        w = separator_width
    );
    println!(
        "  {:>w$}",
        format!("{} total", format_us(grand_total_us, us_width)),
        w = separator_width
    );

    // Detect token count from first 3D shape with seq_len > 1 (prefill), or seq_len == 1 (decode)
    let n_tokens = detect_token_count(trace);
    if n_tokens > 1 {
        let per_token_us = grand_total_us / n_tokens;
        println!(
            "  {:>w$}",
            format!(
                "{} per token ({} tokens)",
                format_us(per_token_us, us_width),
                n_tokens
            ),
            w = separator_width
        );
    }

    // Kernel usage report with timing
    if !kernels.is_empty() {
        let kernel_total_us: u64 = kernels.iter().map(|k| k.total_us).sum();
        let other_us = grand_total_us.saturating_sub(kernel_total_us);

        // Column width for kernel time values
        let max_kernel_us = std::cmp::max(
            kernels.iter().map(|k| k.total_us).max().unwrap_or(0),
            other_us,
        );
        let kw = std::cmp::max(1, format!("{}", max_kernel_us).len());

        println!();
        println!("Kernel Usage:");
        for kernel in kernels {
            let pct = if grand_total_us > 0 {
                kernel.total_us as f64 / grand_total_us as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  {:40} {:6} calls  {} {:5.1}%",
                format!("{}(..)", kernel.name),
                kernel.call_count,
                format_us(kernel.total_us, kw),
                pct,
            );
        }
        if other_us > 0 {
            let other_calls: usize = trace
                .iter()
                .filter(|r| {
                    !r.kernel_name
                        .as_ref()
                        .map(|k| !k.is_empty())
                        .unwrap_or(false)
                })
                .count();
            let pct = if grand_total_us > 0 {
                other_us as f64 / grand_total_us as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  {:40} {:6} calls  {} {:5.1}%",
                "(other)",
                other_calls,
                format_us(other_us, kw),
                pct,
            );
        }
    }

    if !kernel_point_usage.is_empty() {
        println!();
        println!("Kernel Usage By Point:");
        for kernel in kernel_point_usage {
            println!("  {}(..)", kernel.kernel_name);
            for point in &kernel.points {
                let pct = if kernel.total_us > 0 {
                    point.total_us as f64 / kernel.total_us as f64 * 100.0
                } else {
                    0.0
                };
                println!(
                    "    {:20} {:6} calls  {} {:5.1}%",
                    point.point,
                    point.call_count,
                    format_us(point.total_us, 1),
                    pct,
                );
            }
            println!();
        }
    }

    if !other_usage.is_empty() {
        let other_total_us: u64 = other_usage.iter().map(|point| point.total_us).sum();
        println!("Other Usage:");
        for point in other_usage {
            let pct = if other_total_us > 0 {
                point.total_us as f64 / other_total_us as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  {:22} {:6} calls  {} {:5.1}%",
                point.point,
                point.call_count,
                format_us(point.total_us, 1),
                pct,
            );
        }
    }

    if let Some(perf_hints) = perf_hints {
        println!();
        println!("Bench helper:");
        println!(
            "  make -C core/bench/compute/cpu model={}",
            perf_hints.bench_model
        );
    }
}
