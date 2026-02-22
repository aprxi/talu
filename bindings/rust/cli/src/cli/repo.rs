use std::env;
use std::io::{self, BufRead, IsTerminal};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde_json::Value;

use talu::error::last_error_message;
use talu::RepoProgressAction;
use talu::{ChatHandle, InferenceBackend};

use crate::pin_store::PinStore;

use super::util::{format_size, truncate_str};
use super::GetArgs;
use super::RmArgs;
use super::SampleArgs;

struct SyncProgressBars {
    model: ProgressBar,
    file: ProgressBar,
}

struct PinSampleResult {
    text: String,
    load_ns: u64,
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_ns: u64,
    generation_ns: u64,
}

const DEFAULT_PIN_SAMPLE_TOKENS: usize = 20;
const DEFAULT_PIN_SAMPLE_PROMPT: &str = "Reply with one short sentence.";

// =============================================================================
// Unified Progress UI
// =============================================================================

/// Context for unified progress callbacks.
/// Renders progress updates from core using indicatif.
/// Supports multiple progress lines that can be added, updated, and completed.
pub(super) struct UnifiedProgressCtx {
    multi: MultiProgress,
    /// Map of line_id -> ProgressBar
    bars: std::collections::HashMap<u8, ProgressBar>,
}

impl UnifiedProgressCtx {
    pub(super) fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            bars: std::collections::HashMap::new(),
        }
    }

    /// Handle a progress update from repo download.
    pub(super) fn on_download_update(&mut self, update: &talu::DownloadProgress) {
        self.on_update(
            update.action,
            update.line_id,
            &update.label,
            &update.message,
            update.current,
            update.total,
        );
    }

    /// Handle a progress update from convert.
    pub(super) fn on_convert_update(&mut self, update: &talu::ConvertProgress) {
        let action = match update.action {
            talu::ConvertProgressAction::Add => RepoProgressAction::Add,
            talu::ConvertProgressAction::Update => RepoProgressAction::Update,
            talu::ConvertProgressAction::Complete => RepoProgressAction::Complete,
        };
        self.on_update(
            action,
            update.line_id,
            &update.label,
            &update.message,
            update.current,
            update.total,
        );
    }

    /// Handle a progress update.
    pub(super) fn on_update(
        &mut self,
        action: RepoProgressAction,
        line_id: u8,
        label: &str,
        message: &str,
        current: u64,
        total: u64,
    ) {
        match action {
            RepoProgressAction::Add => {
                // Finish any existing bar with this ID
                if let Some(old_bar) = self.bars.remove(&line_id) {
                    old_bar.finish_and_clear();
                }

                // Create new progress bar
                let bar = if total > 0 {
                    let bar = self.multi.add(ProgressBar::new(total));
                    bar.set_style(
                        ProgressStyle::default_bar()
                            .template("{spinner:.cyan} {prefix} [{bar:40.cyan/bright.black}] {pos}/{len} {msg}")
                            .unwrap()
                            .progress_chars("#-"),
                    );
                    bar.set_prefix(label.to_string());
                    bar.set_message(message.to_string());
                    bar.enable_steady_tick(std::time::Duration::from_millis(100));
                    bar
                } else {
                    // Indeterminate (spinner)
                    let bar = self.multi.add(ProgressBar::new_spinner());
                    bar.set_style(
                        ProgressStyle::default_spinner()
                            .template("{spinner:.cyan} {prefix} {msg}")
                            .unwrap(),
                    );
                    bar.set_prefix(label.to_string());
                    bar.set_message(message.to_string());
                    bar.enable_steady_tick(std::time::Duration::from_millis(100));
                    bar.tick(); // Force immediate first render
                    bar
                };

                self.bars.insert(line_id, bar);
            }
            RepoProgressAction::Update => {
                if let Some(bar) = self.bars.get(&line_id) {
                    // Update total if provided (for byte-level progress where total isn't known at start)
                    // Also switch from spinner style to bar style when total becomes known
                    if total > 0 && bar.length() != Some(total) {
                        bar.set_length(total);
                        // Switch to bar style for byte display
                        bar.set_style(
                            ProgressStyle::default_bar()
                                .template("{spinner:.cyan} {prefix} [{bar:40.cyan/bright.black}] {bytes}/{total_bytes}")
                                .unwrap()
                                .progress_chars("#-"),
                        );
                    }

                    // Update position
                    if current > 0 {
                        bar.set_position(current);
                    }

                    // Update message if provided
                    if !message.is_empty() {
                        bar.set_message(message.to_string());
                    }
                }
            }
            RepoProgressAction::Complete => {
                if let Some(bar) = self.bars.remove(&line_id) {
                    bar.finish();
                }
            }
        }
    }

    /// Finalize all progress bars (keep visible).
    pub(super) fn finish(&mut self) {
        for (_, bar) in self.bars.drain() {
            bar.finish();
        }
    }
}

pub(super) fn cmd_get(args: GetArgs) -> Result<()> {
    let pin_bucket = if let Some(explicit) = args.bucket.clone() {
        explicit
    } else {
        crate::config::resolve_and_ensure_bucket(&args.profile)?
    };
    let pin_db_path = pin_bucket.join("meta.sqlite");
    let sync_pins = args.sync_pins;
    let add_pin = args.add_pin.as_deref();
    let remove_pin = args.remove_pin.as_deref();
    let dry_run = !args.no_dry_run;
    let pin_mode = sync_pins || add_pin.is_some() || remove_pin.is_some();

    if pin_mode {
        if args.target.is_some() {
            bail!("Error: pin modes do not accept a model target.");
        }
        if args.model_uri_only {
            bail!("Error: pin modes cannot be combined with --model-uri.");
        }
    }

    if pin_mode && args.force {
        bail!("Error: pin modes do not support --force.");
    }

    let pin_store = if pin_mode {
        Some(PinStore::open(&pin_db_path)?)
    } else {
        None
    };

    if let Some(model_uri) = add_pin {
        let store = pin_store
            .as_ref()
            .ok_or_else(|| anyhow!("internal error: pin store unavailable"))?;
        return cmd_add_pin(store, model_uri);
    }

    if let Some(model_uri) = remove_pin {
        let store = pin_store
            .as_ref()
            .ok_or_else(|| anyhow!("internal error: pin store unavailable"))?;
        return cmd_remove_pin(store, model_uri);
    }

    if sync_pins {
        let store = pin_store
            .as_ref()
            .ok_or_else(|| anyhow!("internal error: pin store unavailable"))?;
        return cmd_sync_pins(
            store,
            args.endpoint_url.as_deref(),
            dry_run,
            args.no_weights,
        );
    }

    let target = match args.target {
        Some(t) => t,
        None => {
            if args.model_uri_only {
                bail!("Error: --model-uri requires a model target (Org/Model).");
            }
            // No target specified — launch interactive HF search TUI
            match crate::hf::run_hf_search(&pin_db_path)? {
                Some(model_id) => model_id,
                None => return Ok(()), // user cancelled
            }
        }
    };

    let force = args.force;
    let endpoint_url = args.endpoint_url;

    if args.model_uri_only {
        let model_id = strip_hf_prefix(&target);
        if !is_model_id(&model_id) {
            bail!("Error: Invalid model URI. Expected format: Org/Model");
        }
        let _path = repo_fetch_no_progress(&model_id, force, endpoint_url.as_deref())?;
        println!("{}", model_id);
        return Ok(());
    }

    if !force && repo_get_cached_path(&target).is_some() {
        println!("Model {} is already cached.", target);
        println!("Use --force to re-download.");
        return Ok(());
    }

    println!("Downloading {}...", target);
    let path = repo_fetch_with_progress(&target, force, endpoint_url.as_deref())?;
    println!("\nDone! Model cached at:\n  {}", path);
    Ok(())
}

pub(super) fn cmd_sample(args: SampleArgs) -> Result<()> {
    let pin_bucket = if let Some(explicit) = args.bucket {
        explicit
    } else {
        crate::config::resolve_and_ensure_bucket(&args.profile)?
    };
    let pin_db_path = pin_bucket.join("meta.sqlite");
    let pin_store = PinStore::open(&pin_db_path)?;
    cmd_sample_pins(&pin_store, args.max_models)
}

fn cmd_add_pin(pin_store: &PinStore, model_uri: &str) -> Result<()> {
    let model_id = strip_hf_prefix(model_uri);
    if !is_model_id(&model_id) {
        bail!("Error: Invalid model URI. Expected format: Org/Model");
    }

    let inserted = pin_store.pin(&model_id)?;
    if let CachePathResult::Cached(_) = repo_get_cached_path_result(&model_id) {
        let local_size = talu::repo::repo_size(&model_id);
        if local_size > 0 {
            pin_store.upsert_size_bytes(&model_id, local_size)?;
        }
    }

    if inserted {
        println!("Pinned {}", model_id);
    } else {
        println!("Already pinned {}", model_id);
    }
    Ok(())
}

fn cmd_remove_pin(pin_store: &PinStore, model_uri: &str) -> Result<()> {
    let model_id = strip_hf_prefix(model_uri);
    if !is_model_id(&model_id) {
        bail!("Error: Invalid model URI. Expected format: Org/Model");
    }

    if pin_store.unpin(&model_id)? {
        println!("Removed pin {}", model_id);
    } else {
        println!("Not pinned {}", model_id);
    }
    Ok(())
}

fn cmd_sync_pins(
    pin_store: &PinStore,
    endpoint_url: Option<&str>,
    dry_run: bool,
    no_weights: bool,
) -> Result<()> {
    let entries = pin_store.list_pinned_entries()?;
    if entries.is_empty() {
        println!("No pinned models for this profile.");
        return Ok(());
    }

    let mut cached: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    let mut invalid: Vec<String> = Vec::new();
    let mut cache_errors: Vec<(String, String)> = Vec::new();
    let mut size_by_model = std::collections::HashMap::<String, u64>::new();

    for entry in &entries {
        let model_id = strip_hf_prefix(&entry.model_uri);
        if !is_model_id(&model_id) {
            invalid.push(entry.model_uri.clone());
            continue;
        }

        if let Some(size) = entry.size_bytes {
            if size > 0 {
                size_by_model.insert(model_id.clone(), size);
            } else {
                let _ = pin_store.clear_size_bytes(&entry.model_uri);
            }
        }

        match repo_get_cached_path_result_ex(&model_id, !no_weights) {
            CachePathResult::Cached(_) => {
                cached.push(model_id.clone());
                let local_size = talu::repo::repo_size(&model_id);
                if local_size > 0 {
                    size_by_model.insert(model_id.clone(), local_size);
                    let _ = pin_store.upsert_size_bytes(&entry.model_uri, local_size);
                }
            }
            CachePathResult::NotCached => missing.push(model_id),
            CachePathResult::Error(err) => cache_errors.push((model_id, err)),
        }
    }

    // Skip size hydration when --no-weights: metadata sizes are negligible.
    let size_errors = if no_weights {
        Vec::new()
    } else {
        let missing_size: Vec<String> = missing
            .iter()
            .filter(|model_id| !size_by_model.contains_key(*model_id))
            .cloned()
            .collect();
        let (_, errors) = hydrate_pin_sizes(
            pin_store,
            &missing_size,
            endpoint_url,
            "Resolving missing pin sizes",
            true,
            false,
            &mut size_by_model,
        );
        errors
    };

    let total = cached.len() + missing.len() + invalid.len() + cache_errors.len();
    let cached_known_size: u64 = cached
        .iter()
        .filter_map(|model_id| size_by_model.get(model_id).copied())
        .sum();

    if no_weights {
        println!("Pinned models: {} (metadata only, skipping weights)", total);
    } else {
        println!("Pinned models: {}", total);
    }
    println!(
        "Already cached: {} ({})",
        cached.len(),
        format_size(cached_known_size)
    );
    if no_weights {
        println!("Need download: {} (metadata only)", missing.len());
    } else {
        let download_known_size: u64 = missing
            .iter()
            .filter_map(|model_id| size_by_model.get(model_id).copied())
            .sum();
        let download_unknown_count = missing
            .iter()
            .filter(|model_id| !size_by_model.contains_key(*model_id))
            .count();
        if download_unknown_count == 0 {
            println!(
                "Need download: {} ({})",
                missing.len(),
                format_size(download_known_size)
            );
        } else {
            println!(
                "Need download: {} ({} + {} unknown)",
                missing.len(),
                format_size(download_known_size),
                download_unknown_count
            );
        }
    }
    if !invalid.is_empty() {
        println!("Invalid pins: {}", invalid.len());
    }
    if !cache_errors.is_empty() {
        println!("Cache check errors: {}", cache_errors.len());
    }
    if !size_errors.is_empty() {
        println!("Size lookup errors: {}", size_errors.len());
    }

    if dry_run {
        println!("\nDry run: no downloads were started.");
        if !missing.is_empty() {
            if no_weights {
                println!("Would download (metadata only, no weights):");
            } else {
                println!("Would download:");
            }
            for model_id in &missing {
                if no_weights {
                    println!("          -  {}", model_id);
                } else {
                    let size_text = format_size_cell(size_by_model.get(model_id).copied());
                    println!("  {:>10}  {}", size_text, model_id);
                }
            }
        }
        if !invalid.is_empty() {
            println!("\nInvalid pinned model URIs:");
            for model_uri in &invalid {
                println!("  {}", model_uri);
            }
        }
        if !cache_errors.is_empty() {
            println!("\nCache check failures:");
            for (model_id, err) in &cache_errors {
                println!("  {}: {}", model_id, err);
            }
        }
        if !size_errors.is_empty() {
            println!("\nSize lookup failures:");
            for (model_id, err) in &size_errors {
                println!("  {}: {}", model_id, err);
            }
        }
        if no_weights {
            println!("\nRun with --no-dry-run to download metadata for missing pinned models.");
        } else {
            println!("\nRun with --no-dry-run to download missing pinned models.");
        }
        return Ok(());
    }

    if missing.is_empty() {
        println!("\nNo missing pinned models to download.");
        if invalid.is_empty() && cache_errors.is_empty() {
            return Ok(());
        }
        bail!(
            "Sync incomplete: {} invalid pins, {} cache check errors.",
            invalid.len(),
            cache_errors.len()
        );
    }

    let mut downloaded = 0usize;
    let mut downloaded_size = 0u64;
    let mut failed_downloads: Vec<(String, String)> = Vec::new();
    let use_tty = io::stdout().is_terminal();

    let sync_bars: Option<SyncProgressBars> = if use_tty {
        let multi = MultiProgress::new();
        let model_bar = multi.add(ProgressBar::new(missing.len() as u64));
        model_bar.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.cyan} Sync pins [{bar:30.cyan/bright.black}] {pos}/{len} {elapsed_precise} {msg}",
                )
                .unwrap()
                .progress_chars("#-"),
        );
        model_bar.enable_steady_tick(Duration::from_millis(120));

        let file_bar = multi.add(ProgressBar::new_spinner());
        file_bar.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} Current file: {msg}")
                .unwrap(),
        );
        file_bar.enable_steady_tick(Duration::from_millis(120));
        file_bar.set_message("waiting");

        Some(SyncProgressBars {
            model: model_bar,
            file: file_bar,
        })
    } else {
        None
    };

    for (idx, model_id) in missing.iter().enumerate() {
        let started = Instant::now();
        let item_size = size_by_model.get(model_id).copied();
        let item_size_text = format_size_cell(item_size);
        if let Some(bars) = &sync_bars {
            bars.model.println(format!(
                "Current [{}/{}]: {} ({})",
                idx + 1,
                missing.len(),
                model_id,
                item_size_text
            ));
            bars.model.set_message(truncate_str(
                &format!("{} {}", model_id, item_size_text),
                44,
            ));
            bars.file.set_message("starting");
        } else {
            println!(
                "[{}/{}] Downloading {} ({})",
                idx + 1,
                missing.len(),
                model_id,
                item_size_text
            );
        }

        match repo_fetch_with_sync_progress(
            model_id,
            false,
            endpoint_url,
            sync_bars.as_ref().map(|bars| &bars.file),
            no_weights,
        ) {
            Ok(_) => {
                downloaded += 1;
                let elapsed = format_elapsed(started.elapsed());
                let local_size = talu::repo::repo_size(model_id);
                if local_size > 0 {
                    downloaded_size = downloaded_size.saturating_add(local_size);
                    size_by_model.insert(model_id.clone(), local_size);
                    let _ = pin_store.upsert_size_bytes(model_id, local_size);
                }
                if let Some(bars) = &sync_bars {
                    bars.model.println(format!(
                        "  [ok] {} ({}, {})",
                        model_id,
                        elapsed,
                        format_size_cell(if local_size > 0 {
                            Some(local_size)
                        } else {
                            item_size
                        })
                    ));
                } else {
                    println!(
                        "  [ok] {} ({}, {})",
                        model_id,
                        elapsed,
                        format_size_cell(if local_size > 0 {
                            Some(local_size)
                        } else {
                            item_size
                        })
                    );
                }
            }
            Err(err) => {
                failed_downloads.push((model_id.clone(), err.to_string()));
                let elapsed = format_elapsed(started.elapsed());
                if let Some(bars) = &sync_bars {
                    bars.model
                        .println(format!("  [failed] {} ({})", model_id, elapsed));
                } else {
                    println!("  [failed] {} ({})", model_id, elapsed);
                }
            }
        }

        if let Some(bars) = &sync_bars {
            bars.model.inc(1);
            bars.file.set_message("waiting");
        }
    }

    if let Some(bars) = sync_bars {
        bars.file.finish_and_clear();
        bars.model.finish_with_message("sync complete");
    }

    println!("\nSync summary:");
    println!(
        "Downloaded: {} ({})",
        downloaded,
        format_size(downloaded_size)
    );
    println!(
        "Already cached: {} ({})",
        cached.len(),
        format_size(cached_known_size)
    );
    if !failed_downloads.is_empty() {
        println!("Failed downloads: {}", failed_downloads.len());
        for (model_id, err) in &failed_downloads {
            println!("  {}: {}", model_id, err);
        }
    }
    if !invalid.is_empty() {
        println!("Invalid pins: {}", invalid.len());
        for model_uri in &invalid {
            println!("  {}", model_uri);
        }
    }
    if !cache_errors.is_empty() {
        println!("Cache check errors: {}", cache_errors.len());
        for (model_id, err) in &cache_errors {
            println!("  {}: {}", model_id, err);
        }
    }
    if !size_errors.is_empty() {
        println!("Size lookup errors: {}", size_errors.len());
        for (model_id, err) in &size_errors {
            println!("  {}: {}", model_id, err);
        }
    }

    if failed_downloads.is_empty() && invalid.is_empty() && cache_errors.is_empty() {
        return Ok(());
    }

    bail!(
        "Sync incomplete: {} failed downloads, {} invalid pins, {} cache check errors.",
        failed_downloads.len(),
        invalid.len(),
        cache_errors.len()
    );
}

fn cmd_sample_pins(pin_store: &PinStore, max_models: Option<usize>) -> Result<()> {
    let entries = pin_store.list_pinned_entries()?;
    if entries.is_empty() {
        println!("No pinned models for this profile.");
        return Ok(());
    }

    let mut cached: Vec<(String, String)> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let mut invalid: Vec<String> = Vec::new();
    let mut cache_errors: Vec<(String, String)> = Vec::new();

    for entry in &entries {
        let model_id = strip_hf_prefix(&entry.model_uri);
        if !is_model_id(&model_id) {
            invalid.push(entry.model_uri.clone());
            continue;
        }

        match repo_get_cached_path_result(&model_id) {
            CachePathResult::Cached(path) => cached.push((model_id, path)),
            CachePathResult::NotCached => skipped.push(model_id),
            CachePathResult::Error(err) => cache_errors.push((model_id, err)),
        }
    }

    if let Some(limit) = max_models {
        if limit == 0 {
            bail!("Error: --max-models must be >= 1.");
        }
        if cached.len() > limit {
            cached.truncate(limit);
        }
    }

    println!("Pinned models: {}", entries.len());
    println!("Cached for sampling: {}", cached.len());
    println!("Skipped (not cached): {}", skipped.len());
    if !invalid.is_empty() {
        println!("Invalid pins: {}", invalid.len());
    }
    if !cache_errors.is_empty() {
        println!("Cache check errors: {}", cache_errors.len());
    }

    for model_id in &skipped {
        println!("[skip] {} (not cached)", model_id);
    }

    if cached.is_empty() {
        if invalid.is_empty() && cache_errors.is_empty() {
            println!("No cached pinned models to sample.");
            return Ok(());
        }
        bail!(
            "Sampling incomplete: {} invalid pins, {} cache check errors.",
            invalid.len(),
            cache_errors.len()
        );
    }

    let max_tokens = env::var("TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PIN_SAMPLE_TOKENS);
    if max_tokens == 0 {
        bail!("Error: TOKENS must be > 0 for 'talu sample'.");
    }
    let sample_prompt = env::var("TALU_SAMPLE_PROMPT")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_PIN_SAMPLE_PROMPT.to_string());

    for (idx, (model_id, model_path)) in cached.iter().enumerate() {
        println!("\n[{}/{}] {}", idx + 1, cached.len(), model_id);
        let sample = sample_model_raw_no_stream(model_path, &sample_prompt, max_tokens)
            .map_err(|e| anyhow!("inference failed for {}: {}", model_id, e))?;
        print!("{}", sample.text);
        if !sample.text.ends_with('\n') {
            println!();
        }
        println!(
            "[timing] load {} | prefill {} ({} tok @ {:.1} t/s) | generation {} ({} tok @ {:.1} t/s)",
            format_duration_ns(sample.load_ns),
            format_duration_ns(sample.prefill_ns),
            sample.prompt_tokens,
            tokens_per_second(sample.prompt_tokens, sample.prefill_ns),
            format_duration_ns(sample.generation_ns),
            sample.completion_tokens,
            tokens_per_second(sample.completion_tokens, sample.generation_ns)
        );
    }

    if !invalid.is_empty() || !cache_errors.is_empty() {
        bail!(
            "Sampling incomplete: {} invalid pins, {} cache check errors.",
            invalid.len(),
            cache_errors.len()
        );
    }

    Ok(())
}

fn sample_model_raw_no_stream(
    model_path: &str,
    prompt: &str,
    max_tokens: usize,
) -> Result<PinSampleResult> {
    let gen_cfg = generation_config(model_path)?;
    let temperature_from_env = env::var("TEMPERATURE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let temperature = temperature_from_env.unwrap_or(gen_cfg.temperature);
    let top_k = env::var("TOP_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(gen_cfg.top_k);
    let top_p = env::var("TOP_P")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(gen_cfg.top_p);

    if temperature < 0.0 {
        bail!("Error: TEMPERATURE must be >= 0, got {}", temperature);
    }
    if !(0.0..=1.0).contains(&top_p) {
        bail!("Error: TOP_P must be in range [0.0, 1.0], got {}", top_p);
    }

    let mut cfg = talu::router::GenerateConfig {
        max_tokens,
        raw_output: true,
        ..Default::default()
    };
    if (gen_cfg.do_sample || temperature_from_env.is_some()) && temperature > 0.0 {
        cfg.temperature = temperature;
        cfg.top_k = top_k;
        cfg.top_p = top_p;
    } else {
        cfg.temperature = 0.0;
    }

    let chat = ChatHandle::new(Some("You are a helpful assistant."))?;
    let content = vec![talu::router::ContentPart::Text(prompt.to_string())];
    let load_start = Instant::now();
    let backend = InferenceBackend::new(model_path)?;
    let load_ns = duration_to_nanos_u64(load_start.elapsed());
    let result = talu::router::generate(&chat, &content, &backend, &cfg)?;
    if result.error_code() != 0 {
        let code = result.error_code();
        let message = last_error_message().unwrap_or_else(|| "generation failed".to_string());
        bail!("Error: {} (code {})", message, code);
    }

    Ok(PinSampleResult {
        text: result.text().unwrap_or_default(),
        load_ns,
        prompt_tokens: result.prompt_tokens(),
        completion_tokens: result.completion_tokens(),
        prefill_ns: result.prefill_ns(),
        generation_ns: result.generation_ns(),
    })
}

fn duration_to_nanos_u64(d: Duration) -> u64 {
    match u64::try_from(d.as_nanos()) {
        Ok(v) => v,
        Err(_) => u64::MAX,
    }
}

fn format_duration_ns(ns: u64) -> String {
    format!("{:.3}s", ns as f64 / 1_000_000_000.0)
}

fn tokens_per_second(tokens: usize, ns: u64) -> f64 {
    if ns == 0 {
        0.0
    } else {
        (tokens as f64) / (ns as f64 / 1_000_000_000.0)
    }
}

fn format_size_cell(size_bytes: Option<u64>) -> String {
    match size_bytes {
        Some(size) => format_size(size),
        None => "-".to_string(),
    }
}

fn hydrate_pin_sizes(
    pin_store: &PinStore,
    model_ids: &[String],
    endpoint_url: Option<&str>,
    activity: &str,
    show_status: bool,
    skip_weights: bool,
    size_by_model: &mut std::collections::HashMap<String, u64>,
) -> (usize, Vec<(String, String)>) {
    let mut deduped: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::<String>::new();
    for model_id in model_ids {
        if seen.insert(model_id.clone()) {
            deduped.push(model_id.clone());
        }
    }
    if deduped.is_empty() {
        return (0, Vec::new());
    }

    if show_status {
        println!("{} ({})...", activity, deduped.len());
    }

    let endpoint = effective_hf_endpoint(endpoint_url);
    let token = env::var("HF_TOKEN").ok();
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(err) => {
            return (
                0,
                deduped
                    .into_iter()
                    .map(|model_id| {
                        (
                            model_id,
                            format!("failed to create async runtime for size lookup: {}", err),
                        )
                    })
                    .collect(),
            );
        }
    };

    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent(format!("talu-cli/{}", env!("TALU_VERSION")))
        .build()
    {
        Ok(client) => Arc::new(client),
        Err(err) => {
            return (
                0,
                deduped
                    .into_iter()
                    .map(|model_id| {
                        (
                            model_id,
                            format!("failed to build HTTP client for size lookup: {}", err),
                        )
                    })
                    .collect(),
            );
        }
    };

    let use_tty = io::stdout().is_terminal();
    let progress = if use_tty && show_status {
        let bar = ProgressBar::new(deduped.len() as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.cyan} {msg} [{bar:30.cyan/bright.black}] {pos}/{len}")
                .unwrap()
                .progress_chars("#-"),
        );
        bar.set_message(activity.to_string());
        Some(bar)
    } else {
        None
    };

    let mut updated_count = 0usize;
    let mut errors: Vec<(String, String)> = Vec::new();
    for model_id in deduped {
        let fetch_result = runtime.block_on(fetch_hf_model_size(
            &client,
            &endpoint,
            &model_id,
            token.as_deref(),
            skip_weights,
        ));
        match fetch_result {
            Ok(size_bytes) if size_bytes > 0 => {
                size_by_model.insert(model_id.clone(), size_bytes);
                if let Err(err) = pin_store.upsert_size_bytes(&model_id, size_bytes) {
                    errors.push((model_id, format!("failed to persist size: {}", err)));
                } else {
                    updated_count += 1;
                }
            }
            Ok(_) => errors.push((model_id, "size metadata unavailable".to_string())),
            Err(err) => errors.push((model_id, err.to_string())),
        }

        if let Some(bar) = &progress {
            bar.inc(1);
        }
    }

    if let Some(bar) = progress {
        bar.finish_and_clear();
    }

    (updated_count, errors)
}

fn effective_hf_endpoint(endpoint_url: Option<&str>) -> String {
    let endpoint = endpoint_url
        .map(str::to_string)
        .or_else(|| env::var("HF_ENDPOINT").ok())
        .unwrap_or_else(|| "https://huggingface.co".to_string());
    endpoint.trim_end_matches('/').to_string()
}

async fn fetch_hf_model_size(
    client: &reqwest::Client,
    endpoint: &str,
    model_id: &str,
    token: Option<&str>,
    skip_weights: bool,
) -> Result<u64> {
    let info_url = format!(
        "{}/api/models/{}?blobs=true&files_metadata=true",
        endpoint, model_id
    );
    if let Ok(body) = hf_get_text(client, &info_url, token).await {
        if let Ok(size) = parse_hf_model_size_bytes(&body, skip_weights) {
            return Ok(size);
        }
    }

    let tree_url = format!("{}/api/models/{}/tree/main?recursive=0", endpoint, model_id);
    let body = hf_get_text(client, &tree_url, token).await?;
    parse_hf_tree_size_bytes(&body, skip_weights)
}

async fn hf_get_text(client: &reqwest::Client, url: &str, token: Option<&str>) -> Result<String> {
    let mut req = client.get(url);
    if let Some(t) = token {
        if !t.trim().is_empty() {
            req = req.bearer_auth(t);
        }
    }

    let response = req
        .send()
        .await
        .map_err(|e| anyhow!("HTTP request failed: {}", e))?;
    if !response.status().is_success() {
        return Err(anyhow!("HTTP {}", response.status()));
    }
    response
        .text()
        .await
        .map_err(|e| anyhow!("failed to read response body: {}", e))
}

fn parse_hf_model_size_bytes(body: &str, skip_weights: bool) -> Result<u64> {
    let value: Value =
        serde_json::from_str(body).map_err(|e| anyhow!("invalid model metadata JSON: {}", e))?;
    let siblings = value
        .get("siblings")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing siblings in model metadata"))?;

    let mut total_size = 0u64;
    for sibling in siblings {
        let filename = sibling
            .get("rfilename")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if filename.starts_with('.') || filename.contains('/') {
            continue;
        }
        if skip_weights && is_weight_filename(filename) {
            continue;
        }
        let size = sibling
            .get("size")
            .and_then(value_as_u64)
            .or_else(|| {
                sibling
                    .get("lfs")
                    .and_then(|v| v.get("size"))
                    .and_then(value_as_u64)
            })
            .unwrap_or(0);
        total_size = total_size.saturating_add(size);
    }

    if total_size == 0 {
        Err(anyhow!("size metadata unavailable"))
    } else {
        Ok(total_size)
    }
}

fn parse_hf_tree_size_bytes(body: &str, skip_weights: bool) -> Result<u64> {
    let value: Value =
        serde_json::from_str(body).map_err(|e| anyhow!("invalid tree metadata JSON: {}", e))?;
    let entries = value
        .as_array()
        .ok_or_else(|| anyhow!("invalid tree response shape"))?;

    let mut total_size = 0u64;
    for entry in entries {
        let filename = entry
            .get("path")
            .or_else(|| entry.get("rfilename"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if filename.is_empty() || filename.starts_with('.') || filename.contains('/') {
            continue;
        }
        if let Some(entry_type) = entry.get("type").and_then(Value::as_str) {
            if entry_type != "file" {
                continue;
            }
        }
        if skip_weights && is_weight_filename(filename) {
            continue;
        }
        let size = entry
            .get("size")
            .and_then(value_as_u64)
            .or_else(|| {
                entry
                    .get("lfs")
                    .and_then(|v| v.get("size"))
                    .and_then(value_as_u64)
            })
            .unwrap_or(0);
        total_size = total_size.saturating_add(size);
    }

    if total_size == 0 {
        Err(anyhow!("size metadata unavailable"))
    } else {
        Ok(total_size)
    }
}

fn is_weight_filename(filename: &str) -> bool {
    filename.ends_with(".safetensors") || filename.ends_with(".safetensors.index.json")
}

fn value_as_u64(value: &Value) -> Option<u64> {
    if let Some(v) = value.as_u64() {
        return Some(v);
    }
    if let Some(v) = value.as_i64() {
        return u64::try_from(v).ok();
    }
    value.as_str().and_then(|s| s.parse::<u64>().ok())
}

fn format_elapsed(d: Duration) -> String {
    let secs = d.as_secs();
    let mins = secs / 60;
    let rem = secs % 60;
    if mins > 0 {
        format!("{}m{}s", mins, rem)
    } else {
        format!("{}s", rem)
    }
}

fn format_sync_file_activity(
    update: &talu::DownloadProgress,
    fallback_file: Option<&str>,
) -> String {
    let file = if !update.label.is_empty() {
        truncate_str(&update.label, 56)
    } else if !update.message.is_empty() {
        truncate_str(&update.message, 56)
    } else if let Some(name) = fallback_file {
        truncate_str(name, 56)
    } else {
        "working".to_string()
    };

    if update.total > 0 {
        format!(
            "{} {}/{}",
            file,
            format_size(update.current),
            format_size(update.total)
        )
    } else if update.current > 0 {
        format!("{} {}", file, format_size(update.current))
    } else {
        file
    }
}

pub(super) fn cmd_rm(args: RmArgs) -> Result<()> {
    for target in &args.targets {
        let model_id = strip_hf_prefix(target);
        if !is_model_id(&model_id) {
            bail!("Error: Invalid model URI '{}'. Expected format: Org/Model", model_id);
        }

        if talu::repo::repo_delete(&model_id) {
            if !args.model_uri_only {
                println!("Deleted {} from cache.", model_id);
            }
        } else {
            // Delete returned false - could be error or not found
            if let Some(err_msg) = last_error_message() {
                bail!("Error deleting from cache: {}", err_msg);
            }
            if !args.model_uri_only {
                println!("Model {} was not found in cache.", model_id);
            }
        }
    }
    Ok(())
}

pub(super) fn parse_scheme(scheme: &str) -> Option<talu::Scheme> {
    talu::Scheme::parse(scheme)
}

// Note: hf:// prefix is no longer supported. This function is kept for backward compatibility
// with any remaining code that might pass hf:// URIs, but will be a no-op for valid model IDs.
pub(super) fn strip_hf_prefix(value: &str) -> String {
    value.to_string()
}

pub(super) fn path_exists(path: &str) -> bool {
    std::path::Path::new(path).exists()
}

pub(super) fn is_model_id(value: &str) -> bool {
    talu::repo::is_model_id(value)
}

pub(super) fn resolve_model_path(path: &str) -> Result<String> {
    talu::repo::resolve_model_path(path).map_err(|e| anyhow!("{}", e))
}

/// Read a line from /dev/tty (bypasses piped stdin).
/// Returns None if no TTY is available.
fn tty_read_line(prompt: &str) -> Option<String> {
    let tty = std::fs::File::open("/dev/tty").ok()?;
    eprint!("{}", prompt);
    let mut reader = io::BufReader::new(tty);
    let mut answer = String::new();
    reader.read_line(&mut answer).ok()?;
    Some(answer)
}

/// Ask a yes/no question via TTY. Returns false if no TTY available.
fn tty_confirm(prompt: &str) -> bool {
    tty_read_line(prompt)
        .map(|a| matches!(a.trim(), "y" | "Y" | "yes" | "Yes" | "YES"))
        .unwrap_or(false)
}

/// Resolve a model argument for inference, prompting before download.
///
/// Resolution order:
/// 1. Local filesystem path (exists on disk) → use directly
/// 2. Talu local cache ($TALU_HOME/models/) → use cached path
/// 3. HuggingFace cache → use cached path
/// 4. Not a model ID (no org/name) → offer to search HuggingFace
/// 5. Model ID not found locally → ask user for download permission
pub(super) fn resolve_model_for_inference(model_arg: &str) -> Result<String> {
    // If it exists as a local path, use it directly
    if path_exists(model_arg) {
        return resolve_model_path(model_arg);
    }

    // Try offline resolution (checks Talu cache + HF cache, no network)
    if let Ok(resolved) = talu::repo::resolve_model_path_ex(model_arg, true) {
        return Ok(resolved);
    }

    // Not cached anywhere — if it doesn't look like org/model, offer search
    if !is_model_id(model_arg) {
        eprintln!("Model '{}' not found locally.", model_arg);
        if !tty_confirm("Search HuggingFace? [y/N] ") {
            bail!("Model not found: {}", model_arg);
        }
        let token = env::var("HF_TOKEN").ok();
        let results = repo_search(model_arg, 5, token.as_deref())?;
        if results.is_empty() {
            bail!("No models found matching '{}'.", model_arg);
        }
        eprintln!("Found {} models:", results.len());
        for (i, r) in results.iter().enumerate() {
            eprintln!("  [{}] {}", i + 1, r);
        }
        let choice = tty_read_line("Enter number to download (or Enter to cancel): ");
        let model_id = match choice {
            Some(ref s) => {
                let s = s.trim();
                if s.is_empty() {
                    bail!("Cancelled.");
                }
                let idx: usize = s.parse().map_err(|_| anyhow!("Invalid selection."))?;
                if idx < 1 || idx > results.len() {
                    bail!("Selection out of range.");
                }
                results[idx - 1].clone()
            }
            None => bail!("No TTY available for selection."),
        };
        eprintln!("Downloading '{}'...", model_id);
        repo_fetch_with_progress(&model_id, false, None)?;
        return resolve_model_path(&model_id);
    }

    eprintln!("Model '{}' not found locally.", model_arg);

    if !tty_confirm("Download from HuggingFace? [y/N] ") {
        bail!(
            "Model '{}' is not cached. Use 'talu get {}' to download it first.",
            model_arg,
            model_arg
        );
    }

    repo_fetch_with_progress(model_arg, false, None)?;
    resolve_model_path(model_arg)
}

pub(super) fn generation_config(model_dir: &str) -> Result<talu::GenerationConfigInfo> {
    talu::model::get_generation_config(model_dir).map_err(|e| anyhow!("{}", e))
}

/// Result type for cache path lookup - distinguishes "not cached" from errors.
enum CachePathResult {
    /// Model is cached at the given path.
    Cached(String),
    /// Model is not cached (no error).
    NotCached,
    /// Error occurred (e.g., permission denied, read-only filesystem).
    Error(String),
}

fn repo_get_cached_path_result(model_id: &str) -> CachePathResult {
    repo_get_cached_path_result_ex(model_id, true)
}

fn repo_get_cached_path_result_ex(model_id: &str, require_weights: bool) -> CachePathResult {
    match talu::repo::repo_get_cached_path_ex(model_id, require_weights) {
        Ok(path) => CachePathResult::Cached(path),
        Err(e) => {
            // Check if this is a "not cached" vs actual error
            let msg = e.to_string();
            if msg.contains("not cached") || msg.contains("Model not cached") {
                CachePathResult::NotCached
            } else {
                CachePathResult::Error(msg)
            }
        }
    }
}

pub(super) fn repo_get_cached_path(model_id: &str) -> Option<String> {
    match repo_get_cached_path_result(model_id) {
        CachePathResult::Cached(path) => Some(path),
        _ => None,
    }
}

pub(super) fn repo_fetch_with_progress(
    model_id: &str,
    force: bool,
    endpoint_url: Option<&str>,
) -> Result<String> {
    let token = env::var("HF_TOKEN").ok();

    // Create unified progress context
    let ctx = std::sync::Arc::new(std::sync::Mutex::new(UnifiedProgressCtx::new()));
    let ctx_clone = ctx.clone();

    let options = talu::DownloadOptions {
        token,
        force,
        endpoint_url: endpoint_url.map(|s| s.to_string()),
        ..Default::default()
    };

    let callback: talu::repo::ProgressCallback = Box::new(move |update| {
        if let Ok(mut guard) = ctx_clone.lock() {
            guard.on_download_update(&update);
        }
    });

    let result = talu::repo::repo_fetch(model_id, options, Some(callback));

    // Finalize progress bars
    if let Ok(mut guard) = ctx.lock() {
        guard.finish();
    }

    result.map_err(|e| anyhow!("{}", e))
}

pub(super) fn repo_fetch_no_progress(
    model_id: &str,
    force: bool,
    endpoint_url: Option<&str>,
) -> Result<String> {
    let token = env::var("HF_TOKEN").ok();
    let options = talu::DownloadOptions {
        token,
        force,
        endpoint_url: endpoint_url.map(|s| s.to_string()),
        ..Default::default()
    };
    talu::repo::repo_fetch(model_id, options, None).map_err(|e| anyhow!("{}", e))
}

pub(super) fn repo_fetch_with_sync_progress(
    model_id: &str,
    force: bool,
    endpoint_url: Option<&str>,
    file_bar: Option<&ProgressBar>,
    skip_weights: bool,
) -> Result<String> {
    let token = env::var("HF_TOKEN").ok();
    let options = talu::DownloadOptions {
        token,
        force,
        endpoint_url: endpoint_url.map(|s| s.to_string()),
        skip_weights,
        cancel_flag: None,
    };

    let callback = file_bar.map(|bar| {
        let file_bar = bar.clone();
        let mut current_file = String::new();
        Box::new(move |update: talu::DownloadProgress| {
            // line_id=0 is repo-level ("Downloading files") and carries filename in message.
            if update.line_id == 0 && !update.message.is_empty() {
                current_file = update.message.clone();
            }
            // line_id=1 is byte-level; add action carries filename in label.
            if update.line_id == 1
                && update.action == RepoProgressAction::Add
                && !update.label.is_empty()
            {
                current_file = update.label.clone();
            }
            if matches!(
                update.action,
                RepoProgressAction::Add | RepoProgressAction::Update
            ) && update.line_id == 1
            {
                let activity = format_sync_file_activity(
                    &update,
                    if current_file.is_empty() {
                        None
                    } else {
                        Some(current_file.as_str())
                    },
                );
                file_bar.set_message(activity);
            }
        }) as talu::repo::ProgressCallback
    });

    talu::repo::repo_fetch(model_id, options, callback).map_err(|e| anyhow!("{}", e))
}

pub(super) fn repo_list_models(
    require_weights: bool,
) -> Result<Vec<(String, String, talu::CacheOrigin)>> {
    let models = talu::repo::repo_list_models(require_weights).map_err(|e| anyhow!("{}", e))?;
    Ok(models
        .into_iter()
        .map(|m| (m.id, m.path, m.source))
        .collect())
}

pub(super) fn repo_list_files(model_path: &str, token: Option<&str>) -> Result<Vec<String>> {
    talu::repo::repo_list_files(model_path, token).map_err(|e| anyhow!("{}", e))
}

pub(super) fn repo_search(query: &str, limit: usize, token: Option<&str>) -> Result<Vec<String>> {
    talu::repo::repo_search(query, limit, token).map_err(|e| anyhow!("{}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parse_hf_model_size_prefers_lfs_size() {
        let json = r#"{
            "siblings": [
                {"rfilename":"README.md","size":1024},
                {"rfilename":"model-00001-of-00002.safetensors","lfs":{"size":3221225472}},
                {"rfilename":"model-00002-of-00002.safetensors","size":3221225472},
                {"rfilename":".gitattributes","size":123},
                {"rfilename":"subdir/file.txt","size":456}
            ]
        }"#;
        let size = parse_hf_model_size_bytes(json, false).expect("size parse");
        assert_eq!(size, 1024 + 3221225472 + 3221225472);
    }

    #[test]
    fn parse_hf_model_size_skip_weights_excludes_safetensors() {
        let json = r#"{
            "siblings": [
                {"rfilename":"README.md","size":1024},
                {"rfilename":"config.json","size":2048},
                {"rfilename":"tokenizer.json","size":4096},
                {"rfilename":"model-00001-of-00002.safetensors","lfs":{"size":3221225472}},
                {"rfilename":"model-00002-of-00002.safetensors","size":3221225472},
                {"rfilename":"model.safetensors.index.json","size":512}
            ]
        }"#;
        let size = parse_hf_model_size_bytes(json, true).expect("size parse");
        assert_eq!(size, 1024 + 2048 + 4096);
    }

    #[test]
    fn parse_hf_model_size_missing_siblings_errors() {
        let json = r#"{"id":"Qwen/Qwen3-0.6B"}"#;
        let err = parse_hf_model_size_bytes(json, false).expect_err("should fail");
        assert!(err.to_string().contains("siblings"));
    }

    #[test]
    fn parse_hf_model_size_all_missing_sizes_errors() {
        let json = r#"{
            "siblings": [
                {"rfilename":"config.json"},
                {"rfilename":"model.safetensors"},
                {"rfilename":"README.md"}
            ]
        }"#;
        let err = parse_hf_model_size_bytes(json, false).expect_err("should fail");
        assert!(err.to_string().contains("size metadata unavailable"));
    }

    #[test]
    fn parse_hf_tree_size_sums_top_level_files() {
        let json = r#"[
            {"path":"config.json","type":"file","size":1024},
            {"path":"model-00001-of-00002.safetensors","type":"file","lfs":{"size":3221225472}},
            {"path":"nested/file.txt","type":"file","size":1234},
            {"path":".gitattributes","type":"file","size":50},
            {"path":"subdir","type":"directory"}
        ]"#;
        let size = parse_hf_tree_size_bytes(json, false).expect("size parse");
        assert_eq!(size, 1024 + 3221225472);
    }

    #[test]
    fn parse_hf_tree_size_skip_weights_excludes_safetensors() {
        let json = r#"[
            {"path":"config.json","type":"file","size":1024},
            {"path":"tokenizer.json","type":"file","size":4096},
            {"path":"model-00001-of-00002.safetensors","type":"file","lfs":{"size":3221225472}},
            {"path":"model.safetensors.index.json","type":"file","size":512}
        ]"#;
        let size = parse_hf_tree_size_bytes(json, true).expect("size parse");
        assert_eq!(size, 1024 + 4096);
    }

    #[test]
    fn add_remove_pin_roundtrip() {
        let db_path = temp_meta_db_path("repo_pin_roundtrip");
        let store = PinStore::open(&db_path).expect("open pin store");

        cmd_add_pin(&store, "Qwen/Qwen3-0.6B").expect("add pin");
        cmd_add_pin(&store, "Qwen/Qwen3-0.6B").expect("add pin idempotent");
        let pinned = store.list_pinned_set().expect("list pinned set");
        assert!(pinned.contains("Qwen/Qwen3-0.6B"));

        cmd_remove_pin(&store, "Qwen/Qwen3-0.6B").expect("remove pin");
        cmd_remove_pin(&store, "Qwen/Qwen3-0.6B").expect("remove pin idempotent");
        let pinned_after = store.list_pinned_set().expect("list pinned set after");
        assert!(!pinned_after.contains("Qwen/Qwen3-0.6B"));
    }

    #[test]
    fn add_pin_rejects_invalid_model_uri() {
        let db_path = temp_meta_db_path("repo_pin_invalid_add");
        let store = PinStore::open(&db_path).expect("open pin store");
        assert!(cmd_add_pin(&store, "invalid-model").is_err());
    }

    #[test]
    fn remove_pin_rejects_invalid_model_uri() {
        let db_path = temp_meta_db_path("repo_pin_invalid_remove");
        let store = PinStore::open(&db_path).expect("open pin store");
        assert!(cmd_remove_pin(&store, "invalid-model").is_err());
    }

    fn temp_meta_db_path(prefix: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{}_{}", prefix, now));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir.join("meta.sqlite")
    }
}
