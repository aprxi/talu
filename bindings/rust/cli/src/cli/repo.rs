use std::env;

use anyhow::{anyhow, bail, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use super::GetArgs;
use super::RmArgs;
use talu::error::last_error_message;
use talu::RepoProgressAction;

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
    /// Sticky metadata used for optional history-style progress logging.
    line_meta: std::collections::HashMap<u8, ProgressLineMeta>,
}

struct ProgressLineMeta {
    label: String,
    total: u64,
    last_logged_current: u64,
}

impl UnifiedProgressCtx {
    pub(super) fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            bars: std::collections::HashMap::new(),
            line_meta: std::collections::HashMap::new(),
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
            false,
            true,
        );
    }

    /// Handle a progress update from convert.
    pub(super) fn on_convert_update(&mut self, update: &talu::ConvertProgress) {
        // Hide noisy high-frequency sub-lines from convert (block/finalize internals).
        // Keep only ordered milestone lines.
        if update.line_id >= 2 {
            return;
        }
        let action = match update.action {
            talu::ConvertProgressAction::Add => RepoProgressAction::Add,
            talu::ConvertProgressAction::Update => RepoProgressAction::Update,
            talu::ConvertProgressAction::Complete => RepoProgressAction::Complete,
        };
        // Only print ordered completion-style history lines for main packing line.
        let emit_history_line = update.line_id == 1;
        self.on_update(
            action,
            update.line_id,
            &update.label,
            &update.message,
            update.current,
            update.total,
            emit_history_line,
            false,
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
        emit_history_line: bool,
        animate: bool,
    ) {
        match action {
            RepoProgressAction::Add => {
                // Finish any existing bar with this ID
                if let Some(old_bar) = self.bars.remove(&line_id) {
                    old_bar.finish_and_clear();
                }
                self.line_meta.insert(
                    line_id,
                    ProgressLineMeta {
                        label: label.to_string(),
                        total,
                        last_logged_current: 0,
                    },
                );

                // Create new progress bar
                let bar = if total > 0 {
                    let bar = self.multi.add(ProgressBar::new(total));
                    if animate {
                        bar.set_style(
                            ProgressStyle::default_bar()
                                .template(
                                    "{spinner:.cyan} {prefix} [{bar:40.cyan/bright.black}] {pos}/{len} {msg}",
                                )
                                .unwrap()
                                .progress_chars("#-"),
                        );
                    } else {
                        bar.set_style(
                            ProgressStyle::default_bar()
                                .template("{prefix} [{bar:40.cyan/bright.black}] {pos}/{len} {msg}")
                                .unwrap()
                                .progress_chars("#-"),
                        );
                    }
                    bar.set_prefix(label.to_string());
                    bar.set_message(message.to_string());
                    if animate {
                        bar.enable_steady_tick(std::time::Duration::from_millis(100));
                    }
                    bar
                } else {
                    // Indeterminate (spinner)
                    let bar = self.multi.add(ProgressBar::new_spinner());
                    if animate {
                        bar.set_style(
                            ProgressStyle::default_spinner()
                                .template("{spinner:.cyan} {prefix} {msg}")
                                .unwrap(),
                        );
                    } else {
                        bar.set_style(
                            ProgressStyle::default_spinner()
                                .template("{prefix} {msg}")
                                .unwrap(),
                        );
                    }
                    bar.set_prefix(label.to_string());
                    bar.set_message(message.to_string());
                    if animate {
                        bar.enable_steady_tick(std::time::Duration::from_millis(100));
                        bar.tick(); // Force immediate first render
                    }
                    bar
                };

                self.bars.insert(line_id, bar);
                if emit_history_line {
                    let history_line =
                        if !message.is_empty() && message.starts_with("Calib tuning args:") {
                            message.to_string()
                        } else {
                            let label_text = if label.is_empty() { "Progress" } else { label };
                            if total > 0 {
                                format!("{label_text} 0/{total} {message}")
                            } else if message.is_empty() {
                                format!("{label_text} 0")
                            } else {
                                format!("{label_text} 0 {message}")
                            }
                        };
                    let _ = self.multi.println(history_line);
                }
            }
            RepoProgressAction::Update => {
                if let Some(bar) = self.bars.get(&line_id) {
                    // Update total if provided (for byte-level progress where total isn't known at start)
                    // Also switch from spinner style to bar style when total becomes known
                    if total > 0 && bar.length() != Some(total) {
                        bar.set_length(total);
                        // Switch to bar style for byte display
                        if animate {
                            bar.set_style(
                                ProgressStyle::default_bar()
                                    .template(
                                        "{spinner:.cyan} {prefix} [{bar:40.cyan/bright.black}] {bytes}/{total_bytes}",
                                    )
                                    .unwrap()
                                    .progress_chars("#-"),
                            );
                        } else {
                            bar.set_style(
                                ProgressStyle::default_bar()
                                    .template("{prefix} [{bar:40.cyan/bright.black}] {bytes}/{total_bytes}")
                                    .unwrap()
                                    .progress_chars("#-"),
                            );
                        }
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
                if emit_history_line && current > 0 {
                    let line = self
                        .line_meta
                        .entry(line_id)
                        .or_insert_with(|| ProgressLineMeta {
                            label: label.to_string(),
                            total,
                            last_logged_current: 0,
                        });
                    if !label.is_empty() {
                        line.label = label.to_string();
                    }
                    if total > 0 {
                        line.total = total;
                    }
                    if current != line.last_logged_current {
                        line.last_logged_current = current;
                        let label_text = if line.label.is_empty() {
                            "Progress"
                        } else {
                            &line.label
                        };
                        let history_line = if line.total > 0 {
                            format!("{label_text} {current}/{} {message}", line.total)
                        } else if message.is_empty() {
                            format!("{label_text} {current}")
                        } else {
                            format!("{label_text} {current} {message}")
                        };
                        let _ = self.multi.println(history_line);
                    }
                }
            }
            RepoProgressAction::Complete => {
                if let Some(bar) = self.bars.remove(&line_id) {
                    bar.finish_and_clear();
                }
                self.line_meta.remove(&line_id);
            }
        }
    }

    /// Finalize all progress bars and clear them before normal CLI output resumes.
    pub(super) fn finish(&mut self) {
        for (_, bar) in self.bars.drain() {
            bar.finish_and_clear();
        }
        self.line_meta.clear();
    }
}

pub(super) fn cmd_get(args: GetArgs) -> Result<()> {
    let target = args.target;

    let force = args.force;
    let endpoint_url = args.endpoint_url;

    if args.model_uri_only {
        let model_id = target.to_string();
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

pub(super) fn cmd_rm(args: RmArgs) -> Result<()> {
    for target in &args.targets {
        let model_id = target.to_string();
        if !is_model_id(&model_id) {
            bail!(
                "Error: Invalid model URI '{}'. Expected format: Org/Model",
                model_id
            );
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

pub(super) fn path_exists(path: &str) -> bool {
    std::path::Path::new(path).exists()
}

pub(super) fn is_model_id(value: &str) -> bool {
    talu::repo::is_model_id(value)
}

pub(super) fn resolve_model_path(path: &str) -> Result<String> {
    talu::repo::resolve_model_path(path).map_err(|e| anyhow!("{}", e))
}

/// Resolve a model argument for inference without interactive fallback.
///
/// Resolution order:
/// 1. Local filesystem path (exists on disk) → use directly
/// 2. Talu local cache ($TALU_HOME/models/) → use cached path
/// 3. HuggingFace cache → use cached path
/// 4. Missing model → fail with an explicit `talu get` instruction
pub(super) fn resolve_model_for_inference(model_arg: &str) -> Result<String> {
    // If it exists as a local path, use it directly
    if path_exists(model_arg) {
        return resolve_model_path(model_arg);
    }

    // Try offline resolution (checks Talu cache + HF cache, no network)
    if let Ok(resolved) = talu::repo::resolve_model_path_ex(model_arg, true) {
        return Ok(resolved);
    }

    if !is_model_id(model_arg) {
        bail!(
            "Model '{}' was not found locally. Use 'talu ls' to inspect cached models or 'talu get <Org/Model>' to download a model.",
            model_arg
        );
    }

    bail!(
        "Model '{}' is not cached. Use 'talu get {}' to download it first.",
        model_arg,
        model_arg
    );
}

/// Result type for cache path lookup - distinguishes "not cached" from errors.
enum CachePathResult {
    /// Model is cached at the given path.
    Cached(String),
    /// Model is not cached (no error).
    NotCached,
    /// Error occurred (e.g., permission denied, read-only filesystem).
    Error,
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
                CachePathResult::Error
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
