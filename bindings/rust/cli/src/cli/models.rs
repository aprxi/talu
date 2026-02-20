use std::collections::HashSet;
use std::env;
use std::io::{self, IsTerminal};

use anyhow::{anyhow, bail, Result};

use talu::{InferenceBackend, QuantMethod};

use crate::pin_store::PinStore;
use crate::provider::{get_provider, is_provider_prefix, provider_from_prefix};

use super::repo::{repo_list_files, repo_list_models, resolve_model_for_inference};
use super::util::{capitalize_first, format_date, format_size, truncate_str};
use super::{DescribeArgs, LsArgs};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalSourceFilter {
    All,
    ManagedOnly,
    HubOnly,
}

impl LocalSourceFilter {
    fn from_ls_args(args: &LsArgs) -> Self {
        if args.quantized_only {
            Self::ManagedOnly
        } else if args.hub_only {
            Self::HubOnly
        } else {
            Self::All
        }
    }

    fn allows_managed(self) -> bool {
        !matches!(self, Self::HubOnly)
    }

    fn allows_hub(self) -> bool {
        !matches!(self, Self::ManagedOnly)
    }
}

pub(super) fn cmd_ls(args: LsArgs) -> Result<()> {
    let source_filter = LocalSourceFilter::from_ls_args(&args);
    let endpoint_url_override = args.endpoint_url;

    // Resolve pinned model set when -P is active.
    let pinned_set = if args.pinned {
        let pin_bucket = if let Some(explicit) = args.bucket.clone() {
            explicit
        } else {
            crate::config::resolve_and_ensure_bucket(&args.profile)?
        };
        let pin_store = PinStore::open(&pin_bucket.join("meta.sqlite"))?;
        Some(pin_store.list_pinned()?.into_iter().collect::<HashSet<_>>())
    } else {
        None
    };

    // No target: list local cached models
    if args.target.is_none() {
        return cmd_ls_local_models(source_filter, pinned_set.as_ref());
    }

    let target = args.target.as_ref().unwrap();

    if args.pinned {
        bail!("'-P/--pinned' is only supported for local cache listings");
    }

    // Check if target is a provider prefix (e.g., "vllm::")
    if is_provider_prefix(target) {
        if source_filter != LocalSourceFilter::All {
            bail!("'-Q/--quantized-only' and '-H/--hub-only' are only supported for local cache listings");
        }
        return cmd_ls_provider(target, endpoint_url_override);
    }

    // Check if target contains "::" - it's a provider::model format
    // For ls, we treat this as listing the provider's models (ignore the model part)
    if let Some(pos) = target.find("::") {
        let provider_prefix = &target[..pos + 2]; // Include the "::"
        if is_provider_prefix(provider_prefix) {
            if source_filter != LocalSourceFilter::All {
                bail!("'-Q/--quantized-only' and '-H/--hub-only' are only supported for local cache listings");
            }
            println!(
                "Note: 'ls {}' lists all models on the provider.",
                provider_prefix
            );
            println!(
                "To generate with a specific model, use: talu -m {} \"prompt\"\n",
                target
            );
            return cmd_ls_provider(provider_prefix, endpoint_url_override);
        }
    }

    // Check if target contains "/" - it's a specific model, list its files
    if target.contains('/') {
        if source_filter != LocalSourceFilter::All {
            bail!("'-Q/--quantized-only' and '-H/--hub-only' are only supported for local cache listings");
        }
        let hf_token = env::var("HF_TOKEN").ok();
        let files = repo_list_files(target, hf_token.as_deref())?;

        println!("Files in {} ({} files):", target, files.len());
        for f in files {
            println!("  {}", f);
        }
        return Ok(());
    }

    // Otherwise, filter cached models by prefix (e.g., "Qwen")
    cmd_ls_prefix_filter(target, source_filter)
}

/// Abbreviate a path by replacing $HOME prefix with ~.
fn abbreviate_home(path: &str) -> String {
    if let Ok(home) = env::var("HOME") {
        if let Some(rest) = path.strip_prefix(&home) {
            return format!("~{rest}");
        }
    }
    path.to_string()
}

/// List cached models, grouped by source (managed first, then hub).
fn cmd_ls_local_models(
    source_filter: LocalSourceFilter,
    pinned_set: Option<&HashSet<String>>,
) -> Result<()> {
    let list = repo_list_models(false)?;

    let is_pinned = |id: &str| pinned_set.map_or(true, |s| s.contains(id));

    let local_models: Vec<_> = list
        .iter()
        .filter(|(id, _, s)| {
            source_filter.allows_managed()
                && *s == talu::CacheOrigin::Managed
                && is_pinned(id)
        })
        .map(|(id, path, _)| (id.clone(), path.clone()))
        .collect();
    let hub_models: Vec<_> = list
        .iter()
        .filter(|(id, _, s)| {
            source_filter.allows_hub() && *s == talu::CacheOrigin::Hub && is_pinned(id)
        })
        .map(|(id, path, _)| (id.clone(), path.clone()))
        .collect();

    // Resolve display paths for section headers
    let talu_dir = talu::repo::get_talu_home()
        .map(|h| format!("{h}/models"))
        .unwrap_or_else(|_| "~/.cache/talu/models".into());
    let hf_dir = talu::repo::get_hf_home()
        .map(|h| format!("{h}/hub"))
        .unwrap_or_else(|_| "~/.cache/huggingface/hub".into());

    let talu_display = abbreviate_home(&talu_dir);
    let hf_display = abbreviate_home(&hf_dir);

    let use_color = io::stdout().is_terminal();

    if source_filter.allows_hub() {
        // Hub models first (less important, scrolls off screen)
        if use_color {
            println!("\n\x1b[2m{hf_display}\x1b[0m");
        } else {
            println!("\n{hf_display}");
        }
        if hub_models.is_empty() {
            println!("  (empty)");
        } else {
            cmd_ls_display_models(&hub_models, use_color)?;
        }
    }

    if source_filter.allows_managed() {
        // Talu local models last (most important, always visible)
        if use_color {
            println!("\n\x1b[1m{talu_display}\x1b[0m \x1b[32m(managed)\x1b[0m");
        } else {
            println!("\n{talu_display} (managed)");
        }
        if local_models.is_empty() {
            println!("  (empty)");
        } else {
            cmd_ls_display_models(&local_models, use_color)?;
        }
    }

    let total = local_models.len() + hub_models.len();
    if total == 0 {
        if pinned_set.is_some() {
            println!("\nNo pinned models found in cache. Pin one with: talu pin Org/Model");
        } else {
            println!("\nNo models found. Download one with: talu get Org/Model");
        }
    }

    Ok(())
}

/// Display models in columnar format.
fn cmd_ls_display_models(list: &[(String, String)], use_color: bool) -> Result<()> {
    // Collect model info for columnar display
    struct Row {
        size_str: String,
        mtime: i64,
        date_str: String,
        type_str: String,
        quant_str: String,
        model_id: String,
    }

    let mut rows: Vec<Row> = Vec::new();
    let mut total_size = 0u64;

    for (model_id, cache_path) in list {
        let size = talu::repo::repo_size(model_id);
        let mtime = talu::repo::repo_mtime(model_id);
        let model_info = get_model_info_for_display(cache_path);

        total_size += size;
        rows.push(Row {
            size_str: format_size(size),
            mtime,
            date_str: format_date(mtime),
            type_str: truncate_str(&model_info.0, 10).to_string(),
            quant_str: model_info.1,
            model_id: model_id.clone(),
        });
    }

    // Oldest first, newest last; tie-break by model ID for deterministic output.
    rows.sort_by(|a, b| {
        a.mtime
            .cmp(&b.mtime)
            .then_with(|| a.model_id.cmp(&b.model_id))
    });

    // Compute column widths from data
    let w_size = rows
        .iter()
        .map(|r| r.size_str.len())
        .max()
        .unwrap_or(0)
        .max(4);
    let w_date = rows
        .iter()
        .map(|r| r.date_str.len())
        .max()
        .unwrap_or(0)
        .max(4);
    let w_type = rows
        .iter()
        .map(|r| r.type_str.len())
        .max()
        .unwrap_or(0)
        .max(4);
    let w_quant = rows
        .iter()
        .map(|r| r.quant_str.len())
        .max()
        .unwrap_or(0)
        .max(5);

    // Print header
    if use_color {
        println!(
            "\x1b[2m{:>w_size$}  {:>w_date$}  {:>w_type$}  {:>w_quant$}  MODEL\x1b[0m",
            "SIZE", "DATE", "TYPE", "QUANT",
        );
    } else {
        println!(
            "{:>w_size$}  {:>w_date$}  {:>w_type$}  {:>w_quant$}  MODEL",
            "SIZE", "DATE", "TYPE", "QUANT",
        );
    }

    // Print each model
    for row in &rows {
        if use_color {
            println!(
                "\x1b[2m{:>w_size$}  {:>w_date$}  {:>w_type$}  {:>w_quant$}\x1b[0m  {}",
                row.size_str, row.date_str, row.type_str, row.quant_str, row.model_id,
            );
        } else {
            println!(
                "{:>w_size$}  {:>w_date$}  {:>w_type$}  {:>w_quant$}  {}",
                row.size_str, row.date_str, row.type_str, row.quant_str, row.model_id,
            );
        }
    }

    // Print summary
    if use_color {
        println!(
            "\n\x1b[2mTotal: {} ({} models)\x1b[0m",
            format_size(total_size),
            rows.len()
        );
    } else {
        println!(
            "\nTotal: {} ({} models)",
            format_size(total_size),
            rows.len()
        );
    }
    Ok(())
}

/// List model IDs from a remote provider
pub(super) fn list_provider_models(
    _provider_name: &str,
    base_url: &str,
    api_key: Option<&str>,
) -> Result<Vec<String>> {
    let backend = InferenceBackend::new_openai_compatible(
        "unused", // Model ID not needed for listing
        base_url, api_key, 10_000, // 10 second timeout for listing
    )?;

    let models = backend.list_models()?;
    Ok(models.into_iter().map(|m| m.id).collect())
}

/// List models on a remote provider (e.g., "talu ls vllm::") via C API
fn cmd_ls_provider(target: &str, endpoint_url_override: Option<String>) -> Result<()> {
    let provider_name =
        provider_from_prefix(target).ok_or_else(|| anyhow!("Unknown provider: {}", target))?;

    let provider = get_provider(&provider_name)
        .ok_or_else(|| anyhow!("Unknown provider: {}", provider_name))?;

    // Determine endpoint URL
    let base_url = endpoint_url_override
        .or_else(|| env::var(format!("{}_ENDPOINT", provider_name.to_uppercase())).ok())
        .unwrap_or_else(|| provider.default_endpoint.to_string());

    // Get API key if needed
    let api_key = provider
        .api_key_env
        .as_ref()
        .and_then(|env_var| env::var(env_var).ok());

    let models = list_provider_models(&provider_name, &base_url, api_key.as_deref())?;

    if models.is_empty() {
        println!("No models found on {} ({})", provider_name, base_url);
        return Ok(());
    }

    println!("Models on {} ({}):", provider_name, base_url);
    for model in &models {
        println!("  {}", model);
    }
    println!("\n{} model(s) available", models.len());

    Ok(())
}

/// Filter cached models by prefix (e.g., "talu ls Qwen")
fn cmd_ls_prefix_filter(prefix: &str, source_filter: LocalSourceFilter) -> Result<()> {
    // Check local cache for matching models
    let list = repo_list_models(false)?;
    let local_matches: Vec<_> = list
        .into_iter()
        .filter(|(id, _, source)| {
            let source_matches = match source {
                talu::CacheOrigin::Managed => source_filter.allows_managed(),
                talu::CacheOrigin::Hub => source_filter.allows_hub(),
            };
            source_matches
                && (id.starts_with(prefix) || id.to_lowercase().starts_with(&prefix.to_lowercase()))
        })
        .map(|(id, path, _)| (id, path))
        .collect();

    if local_matches.is_empty() {
        if source_filter == LocalSourceFilter::ManagedOnly {
            println!("No managed cached models matching '{}'", prefix);
        } else if source_filter == LocalSourceFilter::HubOnly {
            println!("No HuggingFace cached models matching '{}'", prefix);
        } else {
            println!("No cached models matching '{}'", prefix);
        }

        // Check if it might be a provider name
        if let Some(provider) = get_provider(prefix) {
            println!(
                "\nDid you mean 'talu ls {}::' to list models on the {} server?",
                provider.name, provider.name
            );
            return Ok(());
        }

        println!("\nHints:");
        println!("  talu ls              List all cached models");
        println!("  talu ls vllm::       List models on vLLM server");
        return Ok(());
    }

    // Use same columnar format as cmd_ls_local_models
    let use_color = io::stdout().is_terminal();
    cmd_ls_display_models(&local_matches, use_color)
}

/// Get model type and quantization info for display
fn get_model_info_for_display(cache_path: &str) -> (String, String) {
    if cache_path.is_empty() {
        return ("-".to_string(), "-".to_string());
    }

    // Try to get model info via talu::model::describe
    let info = match talu::model::describe(cache_path) {
        Ok(info) => info,
        Err(_) => return ("-".to_string(), "-".to_string()),
    };

    let model_type = if info.model_type.is_empty() {
        "-".to_string()
    } else {
        // Normalize and shorten model type for display
        normalize_model_type(&info.model_type)
    };

    // Determine quantization scheme name
    let quant = format_quant_scheme(info.quant_method, info.quant_bits, info.quant_group_size);

    (model_type, quant)
}

/// Format quantization as a proper scheme name
pub(super) fn format_quant_scheme(method: QuantMethod, bits: i32, group_size: i32) -> String {
    match method {
        QuantMethod::None => "F16".to_string(),
        QuantMethod::Gaffine => {
            // GAF{bits}_{group_size} format
            format!("GAF{}_{}", bits, group_size)
        }
        QuantMethod::Mxfp4 => "MXFP4".to_string(),
        QuantMethod::Native => {
            // Native format (no longer used, fallback to GAF naming)
            format!("GAF{}_{}", bits, group_size)
        }
    }
}

/// Normalize model type for consistent display
fn normalize_model_type(model_type: &str) -> String {
    // Map common model types to shorter, capitalized display names
    let normalized = match model_type.to_lowercase().as_str() {
        "llama" => "Llama",
        "qwen2" | "qwen2_5" => "Qwen2",
        "qwen3" => "Qwen3",
        "qwen3_moe" => "Qwen3MoE",
        "gemma" => "Gemma",
        "gemma2" => "Gemma2",
        "gemma3" | "gemma3_text" => "Gemma3",
        "phi" | "phi3" | "phi4" => "Phi",
        "mistral" => "Mistral",
        "mistral3" => "Mistral3",
        "mixtral" => "Mixtral",
        "granite" => "Granite",
        "granitemoehybrid" => "GraniteMoE",
        "bert" => "BERT",
        "mpnet" => "MPNet",
        "gpt_oss" => "GPT-OSS",
        "smollm3" => "SmolLM3",
        "lfm2" => "LFM2",
        "deepseek_v3" => "DeepSeek",
        "kimi_vl" => "Kimi-VL",
        _ => return capitalize_first(model_type),
    };
    normalized.to_string()
}

pub(super) fn cmd_describe(args: DescribeArgs) -> Result<()> {
    let model_arg = args.model;
    let show_plan = args.plan;
    let json_output = args.json;

    // Resolve model path (handle HF model IDs)
    let model_path = resolve_model_for_inference(&model_arg)?;

    // Get model info
    let info = talu::model::describe(&model_path)?;

    if json_output {
        // Output as JSON
        let json = if show_plan {
            let plan = talu::model::execution_plan(&model_path)?;
            format!(
                r#"{{"model_type": "{}", "architecture": "{}", "num_layers": {}, "hidden_size": {}, "num_heads": {}, "num_kv_heads": {}, "head_dim": {}, "vocab_size": {}, "intermediate_size": {}, "max_seq_len": {}, "quant_bits": {}, "quant_group_size": {}, "num_experts": {}, "experts_per_token": {}, "tie_word_embeddings": {}, "use_gelu": {}, "is_supported": {}, "execution_plan": {{"matmul_kernel": "{}", "attention_type": "{}", "ffn_type": "{}", "uses_gqa": {}, "uses_moe": {}, "uses_quantization": {}}}}}"#,
                info.model_type,
                info.architecture,
                info.num_layers,
                info.hidden_size,
                info.num_heads,
                info.num_kv_heads,
                info.head_dim,
                info.vocab_size,
                info.intermediate_size,
                info.max_seq_len,
                info.quant_bits,
                info.quant_group_size,
                info.num_experts,
                info.experts_per_token,
                info.tie_word_embeddings,
                info.use_gelu,
                plan.is_supported,
                plan.matmul_kernel,
                plan.attention_type,
                plan.ffn_type,
                plan.uses_gqa,
                plan.uses_moe,
                plan.uses_quantization
            )
        } else {
            format!(
                r#"{{"model_type": "{}", "architecture": "{}", "num_layers": {}, "hidden_size": {}, "num_heads": {}, "num_kv_heads": {}, "head_dim": {}, "vocab_size": {}, "intermediate_size": {}, "max_seq_len": {}, "quant_bits": {}, "quant_group_size": {}, "num_experts": {}, "experts_per_token": {}, "tie_word_embeddings": {}, "use_gelu": {}}}"#,
                info.model_type,
                info.architecture,
                info.num_layers,
                info.hidden_size,
                info.num_heads,
                info.num_kv_heads,
                info.head_dim,
                info.vocab_size,
                info.intermediate_size,
                info.max_seq_len,
                info.quant_bits,
                info.quant_group_size,
                info.num_experts,
                info.experts_per_token,
                info.tie_word_embeddings,
                info.use_gelu
            )
        };
        println!("{}", json);
        return Ok(());
    }

    // Human-readable output
    let quant_str = format_quant_scheme(info.quant_method, info.quant_bits, info.quant_group_size);

    println!("MODEL INFO");
    println!(
        "  Type:           {}",
        if info.model_type.is_empty() {
            "-"
        } else {
            &info.model_type
        }
    );
    println!(
        "  Architecture:   {}",
        if info.architecture.is_empty() {
            "-"
        } else {
            &info.architecture
        }
    );
    println!("  Quantization:   {}", quant_str);
    println!();
    println!("DIMENSIONS");
    println!("  Layers:         {}", info.num_layers);
    println!("  Hidden Size:    {}", info.hidden_size);
    println!("  Attention Heads:{}", info.num_heads);
    println!("  KV Heads:       {}", info.num_kv_heads);
    println!("  Head Dim:       {}", info.head_dim);
    println!("  Vocab Size:     {}", info.vocab_size);
    println!("  Intermediate:   {}", info.intermediate_size);
    println!("  Max Seq Len:    {}", info.max_seq_len);

    if info.num_experts > 0 {
        println!();
        println!("MIXTURE OF EXPERTS");
        println!("  Num Experts:    {}", info.num_experts);
        println!("  Top-K:          {}", info.experts_per_token);
    }

    // Show execution plan if requested
    if show_plan {
        let plan = talu::model::execution_plan(&model_path)?;
        println!();

        // Show red warning if model is not supported
        if !plan.is_supported {
            eprintln!(
                "\x1b[31m⚠ WARNING: Model type '{}' is NOT SUPPORTED by talu\x1b[0m",
                info.model_type
            );
            eprintln!("\x1b[31m  The execution plan below shows predicted kernels, but this model cannot run.\x1b[0m");
            eprintln!();
        }

        plan.print_plan();
    }

    Ok(())
}
