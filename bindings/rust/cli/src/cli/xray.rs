use anyhow::{anyhow, Result};
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

pub(super) fn cmd_xray(args: XrayArgs) -> Result<()> {
    let model = &args.model;

    let decode_mode = args.output;

    let capture = XrayCaptureHandle::new()?;
    capture.enable();

    let backend = create_backend_for_model(model)?;
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

    let result = talu::router::generate(&chat, &content, &backend, &cfg)?;
    if result.error_code() != 0 {
        let message = last_error_message().unwrap_or_else(|| "generation failed".to_string());
        return Err(anyhow!("Error: {} (code {})", message, result.error_code()));
    }

    let all_trace = capture.get_trace();

    // Split trace into prefill and decode records.
    // Prefill records have multi-token shapes (shape[1] > 1 for 3D tensors).
    // Decode records have single-token shapes (shape[1] == 1 for 3D, or come after them).
    // The split point is where layer records transition from seq_len>1 to seq_len==1.
    let split_idx = all_trace
        .iter()
        .position(|r| r.layer != 0xFFFF && r.shape.len() >= 2 && r.shape[1] == 1)
        .unwrap_or(all_trace.len());

    let trace: Vec<talu::TraceRecord> = if decode_mode {
        all_trace.into_iter().skip(split_idx).collect()
    } else {
        all_trace.into_iter().take(split_idx).collect()
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
    print_architecture_tree(&model_info, &trace, &record_delta_ns, &usage, mode_label);
    Ok(())
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
    let n_tokens = trace
        .iter()
        .find(|r| r.shape.len() >= 2 && r.shape[0] == 1)
        .map(|r| r.shape[1] as u64)
        .unwrap_or(1);
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
}
