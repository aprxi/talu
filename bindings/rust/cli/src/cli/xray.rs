use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::json;

use talu::error::last_error_message;
use talu::{ChatHandle, XrayCaptureHandle};

use crate::provider::create_backend_for_model;

use super::repo::resolve_model_path;
use super::XrayArgs;

/// Kernel usage derived from trace records.
struct KernelUsage {
    name: String,
    call_count: usize,
    total_us: u64,
}

struct KernelShapeUsageRow {
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
    us_per_token: u64,
}

struct KernelShapeAccumulator {
    call_count: usize,
    total_us: u64,
    samples_us: Vec<u64>,
}

struct PointUsageRow {
    point: String,
    backend: String,
    call_count: usize,
    total_us: u64,
    share_pct: f64,
    avg_us: u64,
    p50_us: u64,
    p90_us: u64,
    min_us: u64,
    max_us: u64,
    us_per_token: u64,
    shape_count: usize,
    kernel_count: usize,
}

struct PointAccumulator {
    call_count: usize,
    total_us: u64,
    samples_us: Vec<u64>,
    shape_signatures: std::collections::HashSet<String>,
    kernels: std::collections::HashSet<String>,
    backends: std::collections::HashSet<String>,
}

struct PointUsage {
    point: String,
    call_count: usize,
    total_us: u64,
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

fn derive_point_table(
    trace: &[talu::TraceRecord],
    record_delta_ns: &[i64],
    total_us: u64,
    token_count: u64,
) -> Vec<PointUsageRow> {
    let mut by_point: std::collections::HashMap<String, PointAccumulator> =
        std::collections::HashMap::new();
    for (i, record) in trace.iter().enumerate() {
        let delta_us = record_delta_ns[i].unsigned_abs() / 1_000;
        let entry = by_point
            .entry(record.point.clone())
            .or_insert_with(|| PointAccumulator {
                call_count: 0,
                total_us: 0,
                samples_us: Vec::new(),
                shape_signatures: std::collections::HashSet::new(),
                kernels: std::collections::HashSet::new(),
                backends: std::collections::HashSet::new(),
            });
        entry.call_count += 1;
        entry.total_us += delta_us;
        entry.samples_us.push(delta_us);
        let signature = format!("{} {}", record.shape_str(), record.dtype_name());
        entry.shape_signatures.insert(signature);
        entry.backends.insert(record.backend_name().to_string());
        if let Some(kernel_name) = &record.kernel_name {
            if !kernel_name.is_empty() {
                entry.kernels.insert(kernel_name.clone());
            }
        }
    }

    by_point
        .into_iter()
        .map(|(point, acc)| {
            let avg_us = if acc.call_count > 0 {
                acc.total_us / acc.call_count as u64
            } else {
                0
            };
            let share_pct = if total_us > 0 {
                (acc.total_us as f64 / total_us as f64) * 100.0
            } else {
                0.0
            };
            let us_per_token = if token_count > 0 {
                acc.total_us / token_count
            } else {
                acc.total_us
            };
            let backend = if acc.backends.len() == 1 {
                acc.backends
                    .iter()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| "-".to_string())
            } else {
                "mixed".to_string()
            };
            PointUsageRow {
                point,
                backend,
                call_count: acc.call_count,
                total_us: acc.total_us,
                share_pct,
                avg_us,
                p50_us: percentile_us(&acc.samples_us, 50),
                p90_us: percentile_us(&acc.samples_us, 90),
                min_us: *acc.samples_us.iter().min().unwrap_or(&0),
                max_us: *acc.samples_us.iter().max().unwrap_or(&0),
                us_per_token,
                shape_count: acc.shape_signatures.len(),
                kernel_count: acc.kernels.len(),
            }
        })
        .collect()
}

fn derive_kernel_shape_table(
    trace: &[talu::TraceRecord],
    record_delta_ns: &[i64],
    total_us: u64,
    token_count: u64,
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
                    samples_us: Vec::new(),
                },
            )
        });
        entry.3.call_count += 1;
        entry.3.total_us += delta_us;
        entry.3.samples_us.push(delta_us);
    }

    let mut rows: Vec<KernelShapeUsageRow> = by_key
        .into_values()
        .map(|(backend, kernel_name, shape, acc)| {
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
            let us_per_token = if token_count > 0 {
                acc.total_us / token_count
            } else {
                acc.total_us
            };
            KernelShapeUsageRow {
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
                us_per_token,
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
    total_us: u64,
    token_count: u64,
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

    println!("Kernel Methods");
    println!(
        "{:<7} {:<24} {:<20} {:>7} {:>10} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "backend",
        "kernel",
        "shape",
        "calls",
        "total_us",
        "share",
        "avg_us",
        "p50_us",
        "p90_us",
        "min_us",
        "max_us",
        "us/token"
    );
    println!("{}", "-".repeat(184));

    for row in &kernel_rows {
        let kernel_name = truncate_with_ellipsis(&row.kernel_name, 24);
        println!(
            "{:<7} {:<24} {:<20} {:>7} {:>10} {:>6.1}% {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}",
            row.backend,
            kernel_name,
            row.shape,
            row.call_count,
            row.total_us,
            row.share_global_pct,
            row.avg_us,
            row.p50_us,
            row.p90_us,
            row.min_us,
            row.max_us,
            row.us_per_token,
        );
    }

    let kernels_total_us: u64 = kernel_rows.iter().map(|r| r.total_us).sum();
    let kernels_total_calls: usize = kernel_rows.iter().map(|r| r.call_count).sum();
    let kernels_share = if total_us > 0 {
        (kernels_total_us as f64 / total_us as f64) * 100.0
    } else {
        0.0
    };
    let total_tok_s = if total_us > 0 {
        (token_count as f64 * 1_000_000.0) / total_us as f64
    } else {
        0.0
    };
    println!("{}", "-".repeat(184));
    println!(
        "{:<7} {:<24} {:<20} {:>7} {:>10} {:>6.1}% ({:.1} tok/s)",
        "-",
        "TOTAL(kernels)",
        "",
        kernels_total_calls,
        kernels_total_us,
        kernels_share,
        total_tok_s
    );
    println!();

    println!("Logical Op Totals");
    println!(
        "{:<22} {:<7} {:>7} {:>10} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>8} {:>8}",
        "logical_op",
        "backend",
        "calls",
        "total_us",
        "share",
        "avg_us",
        "p50_us",
        "p90_us",
        "min_us",
        "max_us",
        "us/token",
        "shapes",
        "kernels"
    );
    println!("{}", "-".repeat(145));
    for row in &point_rows {
        println!(
            "{:<22} {:<7} {:>7} {:>10} {:>6.1}% {:>9} {:>9} {:>9} {:>9} {:>9} {:>10} {:>8} {:>8}",
            row.point,
            row.backend,
            row.call_count,
            row.total_us,
            row.share_pct,
            row.avg_us,
            row.p50_us,
            row.p90_us,
            row.min_us,
            row.max_us,
            row.us_per_token,
            row.shape_count,
            row.kernel_count,
        );
    }
    let points_total_us: u64 = point_rows.iter().map(|r| r.total_us).sum();
    let points_total_calls: usize = point_rows.iter().map(|r| r.call_count).sum();
    let points_share = if total_us > 0 {
        (points_total_us as f64 / total_us as f64) * 100.0
    } else {
        0.0
    };
    println!("{}", "-".repeat(145));
    println!(
        "{:<22} {:<7} {:>7} {:>10} {:>6.1}%",
        "TOTAL(logical_ops)", "-", points_total_calls, points_total_us, points_share
    );

    if let Some(perf_hints) = perf_hints {
        println!();
        println!("Bench helper:");
        println!(
            "  make -C core/bench/compute/cpu model={}",
            perf_hints.bench_model
        );
    }
}

pub(super) fn cmd_xray(args: XrayArgs) -> Result<()> {
    let model = &args.model;

    let decode_mode = args.output;

    let capture = XrayCaptureHandle::new()?;
    capture.enable();

    let backend = create_backend_for_model(model, None)?;
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
    // For --input, keep full trace. CUDA prefill can emit prefix-growing shapes
    // (seq_len=1,2,3,...) which would otherwise be incorrectly split to empty.
    let trace: Vec<talu::TraceRecord> = if decode_mode {
        let split_idx = all_trace
            .iter()
            .position(|r| r.layer != 0xFFFF && r.shape.len() >= 2 && r.shape[1] == 1)
            .unwrap_or(all_trace.len());
        all_trace.into_iter().skip(split_idx).collect()
    } else {
        all_trace
    };

    // Compute per-record deltas (ns from previous record)
    let record_delta_ns: Vec<i64> = (0..trace.len())
        .map(|i| {
            if i == 0 {
                0
            } else {
                trace[i].timestamp_ns - trace[i - 1].timestamp_ns
            }
        })
        .collect();

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
        let kernel_shape_rows =
            derive_kernel_shape_table(&trace, &record_delta_ns, total_us, token_count);
        let point_rows = derive_point_table(&trace, &record_delta_ns, total_us, token_count);
        print_method_table(
            &model_info,
            mode_label,
            kernel_shape_rows,
            point_rows,
            total_us,
            token_count,
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
