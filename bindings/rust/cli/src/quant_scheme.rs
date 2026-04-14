use std::path::Path;

use talu::QuantMethod;

/// Format quantization method for display.
pub(crate) fn format_quant_scheme(method: QuantMethod, bits: i32, group_size: i32) -> String {
    match method {
        QuantMethod::None => "F16".to_string(),
        QuantMethod::Gaffine => format!("TQ{}_{}", bits, group_size),
        QuantMethod::Mxfp4 => "MXFP4".to_string(),
        QuantMethod::Native => format!("TQ{}_{}", bits, group_size),
        QuantMethod::Fp8 => "FP8".to_string(),
        QuantMethod::Mxfp8 => "MXFP8".to_string(),
    }
}

/// Format quantization method for display, with config-based overrides.
///
/// This keeps runtime quantization internals unchanged while preserving
/// user-facing scheme names from conversion metadata (e.g. NVFP4).
pub(crate) fn format_quant_scheme_for_path(
    model_path: &str,
    method: QuantMethod,
    bits: i32,
    group_size: i32,
) -> String {
    detect_scheme_override(model_path)
        .unwrap_or_else(|| format_quant_scheme(method, bits, group_size))
}

fn detect_scheme_override(model_path: &str) -> Option<String> {
    let config_path = Path::new(model_path).join("config.json");
    let config_raw = std::fs::read_to_string(config_path).ok()?;
    let config_json: serde_json::Value = serde_json::from_str(&config_raw).ok()?;
    let quant_config = config_json.get("quantization_config")?;
    let quant_method = quant_config
        .get("quant_method")?
        .as_str()?
        .trim()
        .to_ascii_lowercase();

    if quant_method == "modelopt" {
        let quant_algo = quant_config
            .get("quant_algo")
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_ascii_lowercase());
        if matches!(quant_algo.as_deref(), Some("nvfp4")) {
            return Some("NVFP4".to_string());
        }
    }

    match quant_method.as_str() {
        "nvfp4" => Some("NVFP4".to_string()),
        "mxfp8" => Some("MXFP8".to_string()),
        "mxfp4" => Some("MXFP4".to_string()),
        "fp8" => Some("FP8".to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_quant_scheme_uses_nvfp4_override_from_config() {
        let temp = tempfile::tempdir().expect("tempdir");
        let config_path = temp.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"quantization_config":{"quant_method":"nvfp4"}}"#,
        )
        .expect("write config");

        let scheme = format_quant_scheme_for_path(
            temp.path().to_str().expect("utf8 path"),
            QuantMethod::Gaffine,
            4,
            32,
        );
        assert_eq!(scheme, "NVFP4");
    }

    #[test]
    fn format_quant_scheme_uses_modelopt_nvfp4_override_from_config() {
        let temp = tempfile::tempdir().expect("tempdir");
        let config_path = temp.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}}"#,
        )
        .expect("write config");

        let scheme = format_quant_scheme_for_path(
            temp.path().to_str().expect("utf8 path"),
            QuantMethod::Gaffine,
            4,
            64,
        );
        assert_eq!(scheme, "NVFP4");
    }

    #[test]
    fn format_quant_scheme_falls_back_when_no_override() {
        let temp = tempfile::tempdir().expect("tempdir");
        let scheme = format_quant_scheme_for_path(
            temp.path().to_str().expect("utf8 path"),
            QuantMethod::Gaffine,
            4,
            32,
        );
        assert_eq!(scheme, "TQ4_32");
    }
}
