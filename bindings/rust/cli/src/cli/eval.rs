use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::ffi::c_void;
use std::fs;
use std::time::Instant;

use super::{EvalArgs, EvalCommands, EvalPplArgs};
use talu::error::last_error_message;
use talu::{InferenceBackend, TokenizerHandle};

unsafe extern "C" {
    fn talu_scheduler_score_tokens_joint(
        reference_backend: *mut c_void,
        model_backend: *mut c_void,
        context_tokens: *const u32,
        context_len: usize,
        target_tokens: *const u32,
        target_len: usize,
        max_context: usize,
        out_reference_nll: *mut c_void,
        out_model_nll: *mut c_void,
        out_kld: *mut c_void,
        out_scored_tokens: *mut c_void,
    ) -> i32;
}

#[derive(Debug, Clone, Serialize)]
struct ModelEvalMetrics {
    model: String,
    scored_tokens: usize,
    ppl: f64,
    elapsed_s: f64,
}

#[derive(Debug, Clone, Serialize)]
struct EvalPplOutput {
    dataset: String,
    tokenizer_model: String,
    max_tokens: usize,
    context: usize,
    model: ModelEvalMetrics,
    reference: Option<ModelEvalMetrics>,
    ppl_ratio: Option<f64>,
    ppl_delta_pct: Option<f64>,
    kld_total: Option<f64>,
    kld_per_token: Option<f64>,
    kld_scored_tokens: Option<usize>,
    kld_elapsed_s: Option<f64>,
}

pub(super) fn cmd_eval(args: EvalArgs) -> Result<()> {
    match args.command {
        EvalCommands::Ppl(ppl_args) => cmd_eval_ppl(ppl_args),
    }
}

fn select_eval_tokens(tokens: &[u32], max_tokens: usize) -> &[u32] {
    if max_tokens == 0 {
        return tokens;
    }
    // Need one extra token because scoring uses tokens[0] as context seed and
    // scores tokens[1..].
    let keep = (max_tokens + 1).min(tokens.len());
    &tokens[..keep]
}

fn evaluate_model_ppl(
    model: &str,
    context: &[u32],
    targets: &[u32],
    max_context: usize,
) -> Result<ModelEvalMetrics> {
    let backend = InferenceBackend::new(model)
        .with_context(|| format!("failed to initialize backend for model '{}'", model))?;

    let mut nll_sum: f64 = 0.0;
    let mut scored_tokens: usize = 0;
    let started = Instant::now();

    // SAFETY: pointers are valid for provided lengths and out-params are initialized.
    let rc = unsafe {
        talu_sys::talu_scheduler_score_tokens_nll(
            backend.as_ptr(),
            if context.is_empty() {
                std::ptr::null()
            } else {
                context.as_ptr()
            },
            context.len(),
            if targets.is_empty() {
                std::ptr::null()
            } else {
                targets.as_ptr()
            },
            targets.len(),
            max_context,
            &mut nll_sum as *mut f64 as *mut std::ffi::c_void,
            &mut scored_tokens as *mut usize as *mut std::ffi::c_void,
        )
    };

    if rc != 0 {
        let detail = last_error_message().unwrap_or_else(|| "scoring failed".to_string());
        bail!(
            "failed to score model '{}': {} (code={})",
            model,
            detail,
            rc
        );
    }

    if scored_tokens == 0 {
        bail!("scoring produced zero tokens for model '{}'", model);
    }

    let avg_nll = nll_sum / scored_tokens as f64;
    let ppl = if avg_nll.is_finite() {
        avg_nll.exp()
    } else {
        f64::INFINITY
    };

    Ok(ModelEvalMetrics {
        model: model.to_string(),
        scored_tokens,
        ppl,
        elapsed_s: started.elapsed().as_secs_f64(),
    })
}

fn evaluate_joint_metrics(
    reference_model: &str,
    model: &str,
    context: &[u32],
    targets: &[u32],
    max_context: usize,
) -> Result<(ModelEvalMetrics, ModelEvalMetrics, f64, usize, f64)> {
    let reference_backend = InferenceBackend::new(reference_model).with_context(|| {
        format!(
            "failed to initialize reference backend for model '{}'",
            reference_model
        )
    })?;
    let model_backend = InferenceBackend::new(model)
        .with_context(|| format!("failed to initialize backend for model '{}'", model))?;

    let mut reference_nll_sum: f64 = 0.0;
    let mut model_nll_sum: f64 = 0.0;
    let mut kld_sum: f64 = 0.0;
    let mut scored_tokens: usize = 0;
    let started = Instant::now();

    // SAFETY: backend handles are valid, token pointers are valid for lengths,
    // and out-params are valid writable pointers.
    let rc = unsafe {
        talu_scheduler_score_tokens_joint(
            reference_backend.as_ptr(),
            model_backend.as_ptr(),
            if context.is_empty() {
                std::ptr::null()
            } else {
                context.as_ptr()
            },
            context.len(),
            if targets.is_empty() {
                std::ptr::null()
            } else {
                targets.as_ptr()
            },
            targets.len(),
            max_context,
            &mut reference_nll_sum as *mut f64 as *mut c_void,
            &mut model_nll_sum as *mut f64 as *mut c_void,
            &mut kld_sum as *mut f64 as *mut c_void,
            &mut scored_tokens as *mut usize as *mut c_void,
        )
    };

    if rc != 0 {
        let detail = last_error_message().unwrap_or_else(|| "joint scoring failed".to_string());
        bail!(
            "failed to score joint metrics for model '{}' vs reference '{}': {} (code={})",
            model,
            reference_model,
            detail,
            rc
        );
    }
    if scored_tokens == 0 {
        bail!(
            "joint scoring produced zero tokens for model '{}' vs reference '{}'",
            model,
            reference_model
        );
    }

    let elapsed_s = started.elapsed().as_secs_f64();
    let reference_avg_nll = reference_nll_sum / scored_tokens as f64;
    let model_avg_nll = model_nll_sum / scored_tokens as f64;

    let reference_metrics = ModelEvalMetrics {
        model: reference_model.to_string(),
        scored_tokens,
        ppl: if reference_avg_nll.is_finite() {
            reference_avg_nll.exp()
        } else {
            f64::INFINITY
        },
        elapsed_s,
    };
    let model_metrics = ModelEvalMetrics {
        model: model.to_string(),
        scored_tokens,
        ppl: if model_avg_nll.is_finite() {
            model_avg_nll.exp()
        } else {
            f64::INFINITY
        },
        elapsed_s,
    };

    Ok((
        model_metrics,
        reference_metrics,
        kld_sum,
        scored_tokens,
        elapsed_s,
    ))
}

fn cmd_eval_ppl(args: EvalPplArgs) -> Result<()> {
    let dataset_text = fs::read_to_string(&args.dataset)
        .with_context(|| format!("failed to read dataset file {}", args.dataset.display()))?;
    if dataset_text.trim().is_empty() {
        bail!("dataset file is empty: {}", args.dataset.display());
    }

    // Use reference tokenizer when provided, otherwise the evaluated model tokenizer.
    let tokenizer_model = args.reference.as_deref().unwrap_or(args.model.as_str());
    let tokenizer = TokenizerHandle::new(tokenizer_model)
        .with_context(|| format!("failed to load tokenizer for '{}'", tokenizer_model))?;
    let encoded = tokenizer
        .encode(&dataset_text)
        .with_context(|| format!("failed to tokenize dataset with '{}'", tokenizer_model))?;

    let eval_tokens = select_eval_tokens(&encoded.tokens, args.max_tokens);
    if eval_tokens.len() < 2 {
        bail!(
            "dataset produced {} tokens; need at least 2 tokens to score next-token likelihood",
            eval_tokens.len()
        );
    }

    let context = &eval_tokens[..1];
    let targets = &eval_tokens[1..];

    let (
        model_metrics,
        reference_metrics,
        ppl_ratio,
        ppl_delta_pct,
        kld_total,
        kld_per_token,
        kld_scored_tokens,
        kld_elapsed_s,
    ) = if let Some(reference_model) = args.reference.as_deref() {
        let (model_metrics, reference_metrics, kld_sum, kld_tokens, kld_elapsed) =
            evaluate_joint_metrics(reference_model, &args.model, context, targets, args.context)?;
        let ratio = model_metrics.ppl / reference_metrics.ppl;
        (
            model_metrics,
            Some(reference_metrics),
            Some(ratio),
            Some((ratio - 1.0) * 100.0),
            Some(kld_sum),
            Some(kld_sum / kld_tokens as f64),
            Some(kld_tokens),
            Some(kld_elapsed),
        )
    } else {
        let model_metrics = evaluate_model_ppl(&args.model, context, targets, args.context)?;
        (model_metrics, None, None, None, None, None, None, None)
    };

    let output = EvalPplOutput {
        dataset: args.dataset.display().to_string(),
        tokenizer_model: tokenizer_model.to_string(),
        max_tokens: args.max_tokens,
        context: args.context,
        model: model_metrics,
        reference: reference_metrics,
        ppl_ratio,
        ppl_delta_pct,
        kld_total,
        kld_per_token,
        kld_scored_tokens,
        kld_elapsed_s,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&output)?);
        return Ok(());
    }

    println!("Eval PPL");
    println!("  Dataset: {}", output.dataset);
    println!("  Tokenizer: {}", output.tokenizer_model);
    println!("  Context: {}", output.context);
    println!(
        "  Max tokens: {}",
        if output.max_tokens == 0 {
            "all".to_string()
        } else {
            output.max_tokens.to_string()
        }
    );

    println!("\nModel");
    println!("  URI: {}", output.model.model);
    println!("  Scored tokens: {}", output.model.scored_tokens);
    println!("  PPL: {:.6}", output.model.ppl);
    println!("  Elapsed: {:.2}s", output.model.elapsed_s);

    if let Some(reference) = &output.reference {
        println!("\nReference");
        println!("  URI: {}", reference.model);
        println!("  Scored tokens: {}", reference.scored_tokens);
        println!("  PPL: {:.6}", reference.ppl);
        println!("  Elapsed: {:.2}s", reference.elapsed_s);

        if let Some(ratio) = output.ppl_ratio {
            println!("\nDelta (model vs reference)");
            println!("  PPL ratio: {:.6} (lower than 1.0 is better)", ratio);
        }
        if let Some(delta_pct) = output.ppl_delta_pct {
            println!("  PPL delta: {:+.2}% (negative is better)", delta_pct);
        }
        if let Some(kld) = output.kld_per_token {
            println!("  KLD/token: {:.8} (lower is better)", kld);
        }
        if let Some(kld_total) = output.kld_total {
            println!("  KLD total: {:.8}", kld_total);
        }
        if let Some(kld_tokens) = output.kld_scored_tokens {
            println!("  KLD scored tokens: {}", kld_tokens);
        }
        if let Some(kld_elapsed) = output.kld_elapsed_s {
            println!("  KLD elapsed: {:.2}s", kld_elapsed);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::select_eval_tokens;

    #[test]
    fn select_eval_tokens_keeps_all_when_unbounded() {
        let tokens = [1u32, 2, 3, 4];
        assert_eq!(select_eval_tokens(&tokens, 0), &tokens);
    }

    #[test]
    fn select_eval_tokens_reserves_seed_token() {
        let tokens = [1u32, 2, 3, 4, 5];
        assert_eq!(select_eval_tokens(&tokens, 3), &[1, 2, 3, 4]);
    }
}
