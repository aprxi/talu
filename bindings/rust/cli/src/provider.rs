use anyhow::{anyhow, Result};

use talu::InferenceBackend;

/// Validate a model target for local-only inference.
///
/// talu CLI is local-inference-only. Namespaced backends (`foo::model`) are rejected.
pub fn ensure_local_model_target(model: &str) -> Result<()> {
    if model.contains("::") {
        return Err(anyhow!(
            "Unsupported backend namespace in '{}'. talu is local-inference-only; pass a local path or HuggingFace model ID.",
            model
        ));
    }
    Ok(())
}

pub fn create_backend_for_model(model: &str) -> Result<InferenceBackend> {
    create_backend_for_model_with_progress(model, None)
}

pub fn create_backend_for_model_with_progress(
    model: &str,
    callback: Option<talu::LoadProgressCallback>,
) -> Result<InferenceBackend> {
    ensure_local_model_target(model)?;
    Ok(InferenceBackend::new_with_progress(model, callback)?)
}
