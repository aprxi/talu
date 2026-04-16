use anyhow::{bail, Result};
use std::ffi::OsString;

use super::repo::{
    is_model_id, parse_scheme, repo_fetch_no_progress, repo_fetch_with_progress,
    repo_get_cached_path, UnifiedProgressCtx,
};
use super::{ConvertArgs, TokenizeArgs};

use talu::TokenizerHandle;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConvertProfile {
    Good,
    Custom,
}

impl ConvertProfile {
    fn as_talu(self) -> talu::ConvertProfile {
        match self {
            Self::Good => talu::ConvertProfile::Good,
            Self::Custom => talu::ConvertProfile::Custom,
        }
    }
}

#[derive(Default)]
struct CustomOverrides {
    iters: Option<u32>,
    samples: Option<u32>,
    seqlen: Option<u32>,
    batch_size: Option<u32>,
    nblocks: Option<u32>,
    seed: Option<u64>,
    preserve_blocks: Option<u32>,
    preserve_format: Option<Nvfp4PreserveFormatOverride>,
    lm_head_q: Option<bool>,
    small_model_preserve: Option<bool>,
    clip_mult: Option<f32>,
    scale_refine_mult: Option<f32>,
    replay: Option<Nvfp4ReplayOverride>,
    optimizer: Option<CalibrationOptimizerOverride>,
    clip_min: Option<f32>,
    clip_max: Option<f32>,
    shift_max: Option<f32>,
    adaptive_clip_floor: Option<f32>,
}

impl CustomOverrides {
    fn is_empty(&self) -> bool {
        self.iters.is_none()
            && self.samples.is_none()
            && self.seqlen.is_none()
            && self.batch_size.is_none()
            && self.nblocks.is_none()
            && self.seed.is_none()
            && self.preserve_blocks.is_none()
            && self.preserve_format.is_none()
            && self.lm_head_q.is_none()
            && self.small_model_preserve.is_none()
            && self.clip_mult.is_none()
            && self.scale_refine_mult.is_none()
            && self.replay.is_none()
            && self.optimizer.is_none()
            && self.clip_min.is_none()
            && self.clip_max.is_none()
            && self.shift_max.is_none()
            && self.adaptive_clip_floor.is_none()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Nvfp4ReplayOverride {
    Weighted,
    Xray,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Nvfp4PreserveFormatOverride {
    Bf16,
    Mxfp8,
}

impl Nvfp4PreserveFormatOverride {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "bf16" => Ok(Self::Bf16),
            "mxfp8" => Ok(Self::Mxfp8),
            other => bail!(
                "Invalid preserve_format value '{}'. Allowed: bf16|mxfp8",
                other
            ),
        }
    }

    fn as_env_value(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Mxfp8 => "mxfp8",
        }
    }
}

impl Nvfp4ReplayOverride {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "weighted" | "proxy" | "proxy_only" | "weight_only" => Ok(Self::Weighted),
            "xray" | "capture" | "capture_required" => Ok(Self::Xray),
            other => bail!(
                "Invalid replay value '{}'. Allowed: weighted|xray",
                other
            ),
        }
    }

    fn as_env_value(self) -> &'static str {
        match self {
            Self::Weighted => "weighted",
            Self::Xray => "xray",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CalibrationOptimizerOverride {
    Auto,
    Clip,
    Search,
    ClipSearch,
}

impl CalibrationOptimizerOverride {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "clip" => Ok(Self::Clip),
            "search" => Ok(Self::Search),
            "clip+search" | "clip_search" | "clip-search" => Ok(Self::ClipSearch),
            other => bail!(
                "Invalid optimizer value '{}'. Allowed: auto|clip|search|clip+search",
                other
            ),
        }
    }

    fn as_env_value(self) -> Option<&'static str> {
        match self {
            Self::Auto => None,
            Self::Clip => Some("clip"),
            Self::Search => Some("search"),
            Self::ClipSearch => Some("clip+search"),
        }
    }
}

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: CLI process sets this env var before conversion and restores it on Drop.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }

    fn unset(key: &'static str) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: CLI process removes this env var before conversion and restores it on Drop.
        unsafe { std::env::remove_var(key) };
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => {
                // SAFETY: Restoring previous process env value on scope exit.
                unsafe { std::env::set_var(self.key, value) };
            }
            None => {
                // SAFETY: Restoring previous "unset" state on scope exit.
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }
}

fn parse_u32_value(key: &str, value: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .map_err(|_| anyhow::anyhow!("Invalid value for {}: '{}'", key, value))
}

fn parse_u64_value(key: &str, value: &str) -> Result<u64> {
    value
        .parse::<u64>()
        .map_err(|_| anyhow::anyhow!("Invalid value for {}: '{}'", key, value))
}

fn parse_f32_value(key: &str, value: &str) -> Result<f32> {
    value
        .parse::<f32>()
        .map_err(|_| anyhow::anyhow!("Invalid value for {}: '{}'", key, value))
}

fn parse_bool_value(key: &str, value: &str) -> Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!(
            "Invalid value for {}: '{}' (expected true|false|1|0|on|off)",
            key,
            value
        ),
    }
}

fn is_help_token(value: &str) -> bool {
    matches!(value.trim().to_ascii_lowercase().as_str(), "help" | "list" | "?")
}

fn parse_custom_overrides(args: &[String]) -> Result<CustomOverrides> {
    let mut out = CustomOverrides::default();
    for arg in args {
        for spec in arg.split(',') {
            let item = spec.trim();
            if item.is_empty() {
                continue;
            }
            let (k, v) = item.split_once('=').ok_or_else(|| {
                anyhow::anyhow!("Invalid override '{}', expected key=value", item)
            })?;
            let key = k.trim().to_ascii_lowercase();
            match key.as_str() {
                "iters" | "iter" => out.iters = Some(parse_u32_value(&key, v.trim())?),
                "samples" | "sample" | "nsamples" => {
                    out.samples = Some(parse_u32_value(&key, v.trim())?)
                }
                "seqlen" | "seq_len" => out.seqlen = Some(parse_u32_value(&key, v.trim())?),
                "batch_size" | "batchsize" | "batch" => {
                    out.batch_size = Some(parse_u32_value(&key, v.trim())?)
                }
                "nblocks" | "nblock" | "blocks" => {
                    out.nblocks = Some(parse_u32_value(&key, v.trim())?)
                }
                "seed" => out.seed = Some(parse_u64_value(&key, v.trim())?),
                "preserve_blocks" | "mixed_preserve_blocks" | "nvfp4_preserve_blocks" => {
                    out.preserve_blocks = Some(parse_u32_value(&key, v.trim())?);
                }
                "preserve_format" | "mixed_preserve_format" | "nvfp4_preserve_format" => {
                    out.preserve_format = Some(Nvfp4PreserveFormatOverride::parse(v.trim())?);
                }
                "lm_head_q" | "lmhead_q" | "lm_head_quant" => {
                    out.lm_head_q = Some(parse_bool_value(&key, v.trim())?)
                }
                "small_model_preserve" | "smallpreserve" | "boundary_preserve" => {
                    out.small_model_preserve = Some(parse_bool_value(&key, v.trim())?)
                }
                "clip_mult" | "clip_multiplier" => {
                    let value = parse_f32_value(&key, v.trim())?;
                    if !(value.is_finite() && value > 0.0) {
                        bail!(
                            "Invalid value for {}: '{}' (expected finite > 0)",
                            key,
                            v.trim()
                        );
                    }
                    out.clip_mult = Some(value);
                }
                "scale_refine_mult" | "scale_mult" | "global_scale_mult" => {
                    let value = parse_f32_value(&key, v.trim())?;
                    if !(value.is_finite() && value > 0.0) {
                        bail!(
                            "Invalid value for {}: '{}' (expected finite > 0)",
                            key,
                            v.trim()
                        );
                    }
                    out.scale_refine_mult = Some(value);
                }
                "replay" | "replay_policy" => {
                    out.replay = Some(Nvfp4ReplayOverride::parse(v.trim())?)
                }
                "optimizer" | "optim" => {
                    out.optimizer = Some(CalibrationOptimizerOverride::parse(v.trim())?)
                }
                "clip_min" | "clipmin" => out.clip_min = Some(parse_f32_value(&key, v.trim())?),
                "clip_max" | "clipmax" => out.clip_max = Some(parse_f32_value(&key, v.trim())?),
                "shift_max" | "shiftmax" => {
                    out.shift_max = Some(parse_f32_value(&key, v.trim())?)
                }
                "adaptive_clip_floor" | "adaptive_clip_min" | "adaptivefloor" => {
                    out.adaptive_clip_floor = Some(parse_f32_value(&key, v.trim())?)
                }
                _ => bail!(
                    "Unknown override key '{}'. Allowed: iters,samples,seqlen,batch_size,nblocks,seed,preserve_blocks,preserve_format,lm_head_q,small_model_preserve,clip_mult,scale_refine_mult,replay,optimizer,clip_min,clip_max,shift_max,adaptive_clip_floor",
                    k.trim()
                ),
            }
        }
    }
    Ok(out)
}

fn unsupported_nvfp4_custom_overrides(overrides: &CustomOverrides) -> Vec<&'static str> {
    let mut unsupported = Vec::new();
    if overrides.iters.is_some() {
        unsupported.push("iters");
    }
    if overrides.samples.is_some() {
        unsupported.push("samples");
    }
    if overrides.seqlen.is_some() {
        unsupported.push("seqlen");
    }
    if overrides.batch_size.is_some() {
        unsupported.push("batch_size");
    }
    if overrides.nblocks.is_some() {
        unsupported.push("nblocks");
    }
    if overrides.optimizer.is_some() {
        unsupported.push("optimizer");
    }
    if overrides.clip_min.is_some() {
        unsupported.push("clip_min");
    }
    if overrides.clip_max.is_some() {
        unsupported.push("clip_max");
    }
    if overrides.shift_max.is_some() {
        unsupported.push("shift_max");
    }
    if overrides.adaptive_clip_floor.is_some() {
        unsupported.push("adaptive_clip_floor");
    }
    unsupported
}

pub(super) fn cmd_tokenize(args: TokenizeArgs) -> Result<()> {
    let model_path = args.model;
    if args.text.is_empty() {
        bail!("Usage: talu tokenize <model> <text>");
    }

    let text = args.text.join(" ");
    let tokenizer = TokenizerHandle::new(&model_path)?;
    let result = tokenizer.encode(&text)?;

    print!("Tokens ({}): [", result.tokens.len());
    for (idx, token) in result.tokens.iter().enumerate() {
        if idx > 0 {
            print!(", ");
        }
        print!("{}", token);
    }
    println!("]");

    Ok(())
}

pub(super) fn cmd_convert(args: ConvertArgs) -> Result<()> {
    let json_output = args.json;
    let model_uri_only = args.model_uri_only;
    let quiet = args.quiet;
    let scheme_lower = args.scheme.to_lowercase();
    if is_help_token(&scheme_lower) {
        print_available_schemes();
        return Ok(());
    }
    let scheme_enum = match parse_scheme(&args.scheme) {
        Some(s) => s,
        None => {
            if model_uri_only {
                bail!("Error: Unknown scheme '{}'.", args.scheme);
            }
            if json_output {
                println!(
                    r#"{{"success": false, "error": "Unknown scheme '{}'"}}"#,
                    args.scheme
                );
                return Ok(());
            }
            println!("Error: Unknown scheme '{}'.\n", args.scheme);
            print_available_schemes();
            std::process::exit(1);
        }
    };
    let mut raw_opts: Vec<String> = Vec::new();
    raw_opts.extend(args.opts.iter().cloned());
    // Legacy positional overrides (deprecated): keep working for now.
    raw_opts.extend(args.profile_overrides.iter().cloned());
    if raw_opts.len() == 1 && is_help_token(&raw_opts[0]) {
        print_profile_help_for_scheme(scheme_enum);
        return Ok(());
    }
    let custom_overrides = parse_custom_overrides(&raw_opts)?;
    let effective_profile = if custom_overrides.is_empty() {
        ConvertProfile::Good
    } else {
        ConvertProfile::Custom
    };

    let _optimizer_override_guard = if effective_profile == ConvertProfile::Custom {
        custom_overrides.optimizer.map(|override_value| {
            if let Some(env_value) = override_value.as_env_value() {
                ScopedEnvVar::set("TALU_CONVERT_CALIB_OPTIMIZER", env_value)
            } else {
                ScopedEnvVar::unset("TALU_CONVERT_CALIB_OPTIMIZER")
            }
        })
    } else {
        None
    };
    let mut _calibration_override_guards: Vec<ScopedEnvVar> = Vec::new();
    if effective_profile == ConvertProfile::Custom {
        if let Some(value) = custom_overrides.clip_min {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_CONVERT_CALIB_CLIP_MIN",
                &value.to_string(),
            ));
        }
        if let Some(value) = custom_overrides.clip_max {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_CONVERT_CALIB_CLIP_MAX",
                &value.to_string(),
            ));
        }
        if let Some(value) = custom_overrides.shift_max {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_CONVERT_CALIB_SHIFT_MAX",
                &value.to_string(),
            ));
        }
        if let Some(value) = custom_overrides.adaptive_clip_floor {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_CONVERT_CALIB_ADAPTIVE_CLIP_FLOOR",
                &value.to_string(),
            ));
        }
        if let Some(value) = custom_overrides.preserve_blocks {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_NVFP4_MIXED_PRESERVE_BLOCKS",
                &value.to_string(),
            ));
        }
        if let Some(value) = custom_overrides.preserve_format {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_NVFP4_PRESERVE_FORMAT",
                value.as_env_value(),
            ));
        }
        if let Some(value) = custom_overrides.lm_head_q {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_NVFP4_LM_HEAD_Q",
                if value { "1" } else { "0" },
            ));
        }
        if let Some(value) = custom_overrides.small_model_preserve {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_NVFP4_SMALL_MODEL_PRESERVE",
                if value { "1" } else { "0" },
            ));
        }
        if let Some(value) = custom_overrides.clip_mult {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_NVFP4_CUSTOM_CLIP_MULT",
                &value.to_string(),
            ));
        }
        if let Some(value) = custom_overrides.scale_refine_mult {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_NVFP4_CUSTOM_SCALE_REFINE_MULT",
                &value.to_string(),
            ));
        }
        if let Some(value) = custom_overrides.replay {
            _calibration_override_guards.push(ScopedEnvVar::set(
                "TALU_NVFP4_REPLAY_POLICY",
                value.as_env_value(),
            ));
        }
    }

    if model_uri_only && json_output {
        bail!("Error: --model-uri cannot be combined with --json.");
    }

    let model_arg = if let Some(model) = args.model {
        model
    } else {
        if model_uri_only {
            bail!("Error: --model-uri requires a model target (Org/Model or local path).");
        }
        if json_output {
            println!(r#"{{"success": false, "error": "No model specified"}}"#);
            return Ok(());
        }
        print_convert_usage();
        return Ok(());
    };

    // Resolve output directory: explicit --output, or $TALU_HOME/models
    let output_dir = args.output.unwrap_or_else(|| {
        talu::repo::get_talu_home()
            .map(|h| format!("{h}/models"))
            .unwrap_or_else(|_| "models".into())
    });

    if scheme_enum == talu::Scheme::Nvfp4 && effective_profile == ConvertProfile::Custom {
        let unsupported = unsupported_nvfp4_custom_overrides(&custom_overrides);
        if !unsupported.is_empty() {
            bail!(
                "NVFP4 custom supports only active overrides: preserve_blocks,preserve_format,lm_head_q,small_model_preserve,clip_mult,scale_refine_mult,replay (quality) and seed (reproducibility). Unsupported overrides: {}",
                unsupported.join(",")
            );
        }
    }

    if !quiet && !json_output && !model_uri_only {
        println!(
            "Converting model...\n  Source: {}\n  Scheme: {}\n  Output dir: {}",
            model_arg, args.scheme, output_dir
        );
        if effective_profile == ConvertProfile::Custom {
            println!("  Opts: {}", raw_opts.join(","));
        }
    }

    // Pre-fetch model with progress bar if it's a model ID (not a local path)
    // This reuses the same progress UI as `talu get`
    let model_path = if is_model_id(&model_arg) {
        // Check if already cached
        let cached = repo_get_cached_path(&model_arg);
        if cached.is_none() {
            if !quiet && !json_output && !model_uri_only {
                println!("Downloading model...");
            }
            if model_uri_only {
                repo_fetch_no_progress(&model_arg, false, None)?;
            } else {
                repo_fetch_with_progress(&model_arg, false, None)?;
            }
        }
        // Use the model ID - talu_convert will resolve to cached path
        model_arg.clone()
    } else {
        // Local path - use as-is
        model_arg.clone()
    };

    // Create unified progress context for conversion progress bar
    let ctx = if model_uri_only {
        None
    } else {
        Some(std::sync::Arc::new(std::sync::Mutex::new(
            UnifiedProgressCtx::new(),
        )))
    };
    let ctx_clone = ctx.clone();

    let options = talu::ConvertOptions {
        scheme: Some(scheme_enum),
        force: args.force,
        return_model_id: model_uri_only,
        calibration_profile: Some(effective_profile.as_talu()),
        calibration_seed: Some(custom_overrides.seed.unwrap_or(args.seed)),
        calibration_iters: if effective_profile == ConvertProfile::Custom {
            custom_overrides.iters
        } else {
            None
        },
        calibration_nsamples: if effective_profile == ConvertProfile::Custom {
            custom_overrides.samples
        } else {
            None
        },
        calibration_seqlen: if effective_profile == ConvertProfile::Custom {
            custom_overrides.seqlen
        } else {
            None
        },
        calibration_batch_size: if effective_profile == ConvertProfile::Custom {
            custom_overrides.batch_size
        } else {
            None
        },
        calibration_nblocks: if effective_profile == ConvertProfile::Custom {
            custom_overrides.nblocks
        } else {
            None
        },
    };

    let callback: Option<talu::convert::ConvertProgressCallback> = ctx_clone.map(|ctx| {
        Box::new(move |update| {
            if let Ok(mut guard) = ctx.lock() {
                guard.on_convert_update(&update);
            }
        }) as talu::convert::ConvertProgressCallback
    });

    let result = talu::convert::convert(&model_path, &output_dir, options, callback);

    // Finalize progress bars
    if let Some(ctx) = ctx {
        if let Ok(mut guard) = ctx.lock() {
            guard.finish();
        }
    }

    match result {
        Ok(convert_result) => {
            // Write talu_meta.json with source model info (best-effort).
            if is_model_id(&model_arg) {
                let meta_path =
                    std::path::Path::new(&convert_result.output_path).join("talu_meta.json");
                let meta = serde_json::json!({
                    "source_model_id": model_arg,
                    "scheme": args.scheme,
                    "profile": match effective_profile {
                        ConvertProfile::Good => "good",
                        ConvertProfile::Custom => "custom",
                    },
                    "seed": args.seed,
                    "opts": if effective_profile == ConvertProfile::Custom {
                        raw_opts
                    } else {
                        Vec::<String>::new()
                    },
                });
                let _ = std::fs::write(
                    &meta_path,
                    serde_json::to_string_pretty(&meta).unwrap_or_default(),
                );
            }

            // Output based on flags
            if json_output {
                println!(
                    r#"{{"success": true, "output_path": "{}"}}"#,
                    convert_result.output_path
                );
            } else if model_uri_only {
                println!("{}", convert_result.output_path);
            } else if quiet {
                println!("{}", convert_result.output_path);
            } else {
                println!("\nDone! Model saved to:\n  {}", convert_result.output_path);
            }
            Ok(())
        }
        Err(e) => {
            let msg = e.to_string();
            if json_output {
                // Escape the error message for JSON
                let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
                println!(r#"{{"success": false, "error": "{}"}}"#, escaped);
                return Ok(());
            }
            if !args.force && (msg.contains("InvalidConfig") || msg.contains("UnsupportedModel")) {
                bail!(
                    "Error: {}. Existing converted artifact may be stale; rerun with --force to rebuild it.",
                    msg
                );
            }
            bail!("Error: {}", msg);
        }
    }
}

fn print_convert_usage() {
    let usage = r#"Usage: talu convert <model> [options]

Convert a transformer model to quantized format.

Arguments:
  <model>           HuggingFace model ID (e.g., Qwen/Qwen3-0.6B) or local path

Options:
  --scheme NAME     Quantization scheme (default: tq4)
  --seed N          Deterministic calibration seed (default: 42)
  --output DIR      Output directory (default: $TALU_HOME/models)
  -f, --force       Overwrite existing output
  -q, --quiet       Output only the path (for scripting)
  --model-uri       Output only the URI (CI-friendly)
  --json            Output JSON: {"success": true, "output_path": "..."}
  --opts help       Show scheme-specific tunable options
  --opts KEY=VALUE  Override conversion knobs (activates custom behavior automatically). Repeat or pass CSV.
  OVERRIDE          (legacy positional, deprecated): same format as --opts

Talu Quantized (DEFAULT):
  tq4        4-bit quantized (DEFAULT). GROUP_SIZE env var overrides group size (default: 32).
  tq8        8-bit quantized. GROUP_SIZE env var overrides group size (default: 64).

Hardware Float / Native:
  fp8        FP8 E4M3, block_size=128x128
  mxfp8      MXFP8 E4M3 + E8M0 scales, block_size=32 (Blackwell tensor cores)
  nvfp4      NVFP4 surface (emits runtime-native 4-bit packed layout)

Output naming:
  Models are saved with hierarchical paths: {output_dir}/{org}/{model}-{SUFFIX}
  Examples:
    Qwen/Qwen3-0.6B + tq4     -> models/Qwen/Qwen3-0.6B-TQ4
    Qwen/Qwen3-0.6B + tq8     -> models/Qwen/Qwen3-0.6B-TQ8
    Qwen/Qwen3-0.6B + fp8     -> models/Qwen/Qwen3-0.6B-FP8
    Qwen/Qwen3-0.6B + mxfp8   -> models/Qwen/Qwen3-0.6B-MXFP8

Environment Variables:
  GROUP_SIZE=N      Group size override for tq4 (default: 32) / tq8 (default: 64). Accepts 32, 64, or 128.
  THREADS=N         Number of threads for quantization (default: CPU count)
  HF_TOKEN          API token for private models

Examples:
  talu convert Qwen/Qwen3-0.6B                       # tq4 (default)
  talu convert Qwen/Qwen3-0.6B --scheme tq8          # Near-lossless
  talu convert Qwen/Qwen3-0.6B --scheme tq4          # Highest accuracy
  talu convert ./models/Qwen--Qwen3-0.6B --output /tmp -f

  # CI/scripting: get just the path
  MODEL=$(talu convert Qwen/Qwen3-0.6B -q)

  # CI/scripting: get just the URI (model-uri mode)
  MODEL_URI=$(talu convert Qwen/Qwen3-0.6B --model-uri)

  # CI/scripting: get JSON output
  talu convert Qwen/Qwen3-0.6B --json
  # -> {"success": true, "output_path": "models/Qwen/Qwen3-0.6B-TQ4"}

  # Custom override example (auto-switches to custom behavior)
  talu convert Qwen/Qwen3.5-2B --scheme nvfp4 --opts replay=weighted,preserve_blocks=2,preserve_format=bf16,lm_head_q=1,small_model_preserve=0,clip_mult=1.00,scale_refine_mult=1.00

"#;
    print!("{}", usage);
}

fn print_available_schemes() {
    let schemes = r#"Available quantization schemes:

  Talu Quantized (DEFAULT):
    tq4        4-bit quantized (DEFAULT). GROUP_SIZE env var overrides group size (default: 32).
    tq8        8-bit quantized. GROUP_SIZE env var overrides group size (default: 64).

  Hardware Float / Native:
    fp8        FP8 E4M3, block_size=128x128
    mxfp8      MXFP8 E4M3 + E8M0 scales, block_size=32 (Blackwell tensor cores)
    nvfp4      NVFP4 surface (emits runtime-native 4-bit packed layout)

Usage: talu convert -m <model> --scheme <SCHEME>
Advanced: talu convert <model> --scheme mxfp8 --opts iters=...,samples=...
"#;
    print!("{}", schemes);
}

fn print_profile_help_for_scheme(scheme: talu::Scheme) {
    match scheme {
        talu::Scheme::Nvfp4 => {
            let text = r#"Options help for scheme: nvfp4

Custom parameters (active):
  preserve_blocks=INT>=0
      Keep an exact number of worst-scoring transformer blocks in dense precision.
      When omitted, converter auto-selects blocks from the default 10% budget.
  preserve_format=bf16|mxfp8
      Stored format for preserved dense blocks.
      bf16 is safest; mxfp8 reduces size for preserved blocks.
  lm_head_q=0|1
      Quantize lm_head when 1; keep dense when 0.
  small_model_preserve=0|1
      Preserve boundary layers for small models when 1.
  clip_mult=FLOAT>0
      Per-block clipping multiplier for block-scale derivation.
  scale_refine_mult=FLOAT>0
      Post multiplier on selected global scale.
  replay=weighted|xray
      weighted: weight-domain fast path.
      xray: activation-capture replay path (requires xray bridge).
  seed=INT
      Reproducibility only.

Example:
  talu convert <model> --scheme nvfp4 --opts replay=weighted,preserve_blocks=2,preserve_format=bf16,lm_head_q=1,small_model_preserve=0,clip_mult=1.00,scale_refine_mult=1.00
"#;
            print!("{}", text);
        }
        talu::Scheme::Mxfp8
        | talu::Scheme::Tq432
        | talu::Scheme::Tq464
        | talu::Scheme::Tq4128
        | talu::Scheme::Tq832
        | talu::Scheme::Tq864
        | talu::Scheme::Tq8128 => {
            let text = r#"Options help for scheme: grouped calibration path (tq*/mxfp8)

Custom parameters (active):
  iters=INT
      Optimization steps per tensor (more is slower).
  samples=INT
      Calibration sample count.
  seqlen=INT
      Calibration sequence length.
  batch_size=INT
      Calibration batch size.
  nblocks=INT
      Calibration block multiplier.
  optimizer=auto|clip|search|clip+search
      Select calibration optimizer mode.
  clip_min=FLOAT, clip_max=FLOAT
      Clip-search range.
  shift_max=FLOAT
      Max additive shift during search.
  adaptive_clip_floor=FLOAT
      Lower bound for adaptive clipping.
  seed=INT
      Reproducibility.

Example:
  talu convert <model> --scheme tq4 --opts optimizer=clip+search,iters=16,samples=128,seqlen=1024,batch_size=1,nblocks=1
"#;
            print!("{}", text);
        }
        _ => {
            let text = r#"Options help for this scheme

Custom parameters:
  This scheme does not expose scheme-specific profile tuning knobs today.
  seed=INT may still be accepted for reproducibility where calibration is used.
"#;
            print!("{}", text);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_custom_overrides, unsupported_nvfp4_custom_overrides};

    #[test]
    fn parse_custom_overrides_accepts_aliases_and_csv() {
        let args = vec![
            "iters=400,samples=256,seqlen=2048".to_string(),
            "batch=2,blocks=3,seed=123,preserve_blocks=2,preserve_format=mxfp8,lm_head_q=true,small_model_preserve=off,clip_mult=1.02,scale_refine_mult=0.99,replay=xray,optimizer=clip+search,clip_min=0.6,clip_max=1.8,shift_max=0.75,adaptive_clip_floor=0.5".to_string(),
        ];
        let parsed = parse_custom_overrides(&args).expect("parse");
        assert_eq!(parsed.iters, Some(400));
        assert_eq!(parsed.samples, Some(256));
        assert_eq!(parsed.seqlen, Some(2048));
        assert_eq!(parsed.batch_size, Some(2));
        assert_eq!(parsed.nblocks, Some(3));
        assert_eq!(parsed.seed, Some(123));
        assert_eq!(parsed.preserve_blocks, Some(2));
        assert_eq!(
            parsed.preserve_format,
            Some(super::Nvfp4PreserveFormatOverride::Mxfp8)
        );
        assert_eq!(parsed.lm_head_q, Some(true));
        assert_eq!(parsed.small_model_preserve, Some(false));
        assert!(parsed.clip_mult.is_some_and(|v| (v - 1.02).abs() < 1e-6));
        assert!(parsed.scale_refine_mult.is_some_and(|v| (v - 0.99).abs() < 1e-6));
        assert_eq!(parsed.replay, Some(super::Nvfp4ReplayOverride::Xray));
        assert_eq!(
            parsed.optimizer,
            Some(super::CalibrationOptimizerOverride::ClipSearch)
        );
        assert!(parsed.clip_min.is_some_and(|v| (v - 0.6).abs() < 1e-6));
        assert!(parsed.clip_max.is_some_and(|v| (v - 1.8).abs() < 1e-6));
        assert!(parsed.shift_max.is_some_and(|v| (v - 0.75).abs() < 1e-6));
        assert!(parsed
            .adaptive_clip_floor
            .is_some_and(|v| (v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn parse_custom_overrides_rejects_unknown_key() {
        let args = vec!["foo=1".to_string()];
        match parse_custom_overrides(&args) {
            Ok(_) => panic!("expected error"),
            Err(err) => assert!(err.to_string().contains("Unknown override key")),
        }
    }

    #[test]
    fn parse_custom_overrides_rejects_non_integer_values() {
        let args = vec!["iters=abc".to_string()];
        match parse_custom_overrides(&args) {
            Ok(_) => panic!("expected error"),
            Err(err) => assert!(err.to_string().contains("Invalid value for iters")),
        }
    }

    #[test]
    fn parse_custom_overrides_rejects_preserve_pct_key() {
        let args = vec!["preserve_pct=10".to_string()];
        match parse_custom_overrides(&args) {
            Ok(_) => panic!("expected error"),
            Err(err) => assert!(err.to_string().contains("Unknown override key")),
        }
    }

    #[test]
    fn unsupported_nvfp4_custom_overrides_reports_non_active_knobs() {
        let args = vec!["iters=64,optimizer=clip+search,preserve_blocks=3,seed=7".to_string()];
        let parsed = parse_custom_overrides(&args).expect("parse");
        let unsupported = unsupported_nvfp4_custom_overrides(&parsed);
        assert_eq!(unsupported, vec!["iters", "optimizer"]);
    }
}
