use anyhow::{bail, Result};

use super::repo::{
    is_model_id, parse_scheme, repo_fetch_no_progress, repo_fetch_with_progress,
    repo_get_cached_path, UnifiedProgressCtx,
};
use super::{ConvertArgs, TokenizeArgs};

use talu::TokenizerHandle;

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

    if model_uri_only && json_output {
        bail!("Error: --model-uri cannot be combined with --json.");
    }

    // Handle --scheme help/list to show available schemes (before model check)
    let scheme_lower = args.scheme.to_lowercase();
    if scheme_lower == "help" || scheme_lower == "list" || scheme_lower == "?" {
        print_available_schemes();
        return Ok(());
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

    if !quiet && !json_output && !model_uri_only {
        println!(
            "Converting model...\n  Source: {}\n  Scheme: {}\n  Output dir: {}",
            model_arg, args.scheme, output_dir
        );
    }

    // Pre-fetch model with progress bar if it's a model ID (not a local path)
    // This reuses the same progress UI as `talu get`
    let model_path = if is_model_id(&model_arg) {
        // Check if already cached (unless force is set)
        let cached = repo_get_cached_path(&model_arg);
        if cached.is_none() || args.force {
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
  --scheme NAME     Quantization scheme (default: gaf4_64)
  --output DIR      Output directory (default: $TALU_HOME/models)
  -f, --force       Overwrite existing output
  -q, --quiet       Output only the path (for scripting)
  --model-uri       Output only the URI (CI-friendly)
  --json            Output JSON: {"success": true, "output_path": "..."}

Grouped Affine (DEFAULT):
  gaf4_32    4-bit, group_size=32 (highest accuracy)
  gaf4_64    4-bit, group_size=64 (DEFAULT, balanced)
  gaf4_128   4-bit, group_size=128 (smallest)
  gaf8_32    8-bit, group_size=32
  gaf8_64    8-bit, group_size=64
  gaf8_128   8-bit, group_size=128

Output naming:
  Models are saved with hierarchical paths: {output_dir}/{org}/{model}-{SUFFIX}
  Examples:
    Qwen/Qwen3-0.6B + gaf4_64 -> models/Qwen/Qwen3-0.6B-GAF4
    Qwen/Qwen3-0.6B + gaf8_64 -> models/Qwen/Qwen3-0.6B-GAF8

Environment Variables:
  THREADS=N         Number of threads for quantization (default: CPU count)
  HF_TOKEN          API token for private models

Examples:
  talu convert Qwen/Qwen3-0.6B                       # gaf4_64 (default)
  talu convert Qwen/Qwen3-0.6B --scheme gaf8_64     # Near-lossless
  talu convert Qwen/Qwen3-0.6B --scheme gaf4_32    # Highest accuracy
  talu convert ./models/Qwen--Qwen3-0.6B --output /tmp -f

  # CI/scripting: get just the path
  MODEL=$(talu convert Qwen/Qwen3-0.6B -q)

  # CI/scripting: get just the URI (model-uri mode)
  MODEL_URI=$(talu convert Qwen/Qwen3-0.6B --model-uri)

  # CI/scripting: get JSON output
  talu convert Qwen/Qwen3-0.6B --json
  # -> {"success": true, "output_path": "models/Qwen/Qwen3-0.6B-GAF4"}

"#;
    print!("{}", usage);
}

fn print_available_schemes() {
    let schemes = r#"Available quantization schemes:

  Grouped Affine (DEFAULT):
    gaf4_32    4-bit, group_size=32 (highest accuracy)
    gaf4_64    4-bit, group_size=64 (DEFAULT, balanced)
    gaf4_128   4-bit, group_size=128 (smallest)
    gaf8_32    8-bit, group_size=32
    gaf8_64    8-bit, group_size=64
    gaf8_128   8-bit, group_size=128

Usage: talu convert -m <model> --scheme <SCHEME>
"#;
    print!("{}", schemes);
}
