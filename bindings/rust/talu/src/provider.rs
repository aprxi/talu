//! Safe wrappers for talu provider registry.

use crate::error::error_from_last_or;
use crate::Result;
use std::ffi::{CStr, CString};

/// Information about a model provider.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Provider name (e.g., "openai", "anthropic", "vllm").
    pub name: String,
    /// Default API endpoint URL.
    pub default_endpoint: String,
    /// Environment variable name for API key.
    pub api_key_env: String,
}

impl ProviderInfo {
    fn from_c(info: &talu_sys::CProviderInfo) -> Self {
        Self {
            name: if info.name.is_null() {
                String::new()
            } else {
                // SAFETY: name is a valid C string from the C API.
                unsafe { CStr::from_ptr(info.name) }
                    .to_string_lossy()
                    .into_owned()
            },
            default_endpoint: if info.default_endpoint.is_null() {
                String::new()
            } else {
                // SAFETY: default_endpoint is a valid C string from the C API.
                unsafe { CStr::from_ptr(info.default_endpoint) }
                    .to_string_lossy()
                    .into_owned()
            },
            api_key_env: if info.api_key_env.is_null() {
                String::new()
            } else {
                // SAFETY: api_key_env is a valid C string from the C API.
                unsafe { CStr::from_ptr(info.api_key_env) }
                    .to_string_lossy()
                    .into_owned()
            },
        }
    }
}

/// Result of parsing a model target string.
#[derive(Debug, Clone)]
pub struct ParsedModelTarget {
    /// The provider info if a provider prefix was found.
    pub provider: ProviderInfo,
    /// The model ID portion (after the provider prefix).
    pub model_id: String,
}

/// Returns the number of registered providers.
pub fn provider_count() -> usize {
    // SAFETY: talu_provider_count is a simple getter with no preconditions.
    unsafe { talu_sys::talu_provider_count() }
}

/// Gets provider info by index.
pub fn provider_get(index: usize) -> Result<ProviderInfo> {
    let mut info = talu_sys::CProviderInfo::default();
    // SAFETY: info is a valid mutable pointer to a CProviderInfo struct.
    let rc = unsafe { talu_sys::talu_provider_get(index, &mut info) };
    if rc != 0 {
        return Err(error_from_last_or("Failed to get provider"));
    }
    Ok(ProviderInfo::from_c(&info))
}

/// Gets provider info by name.
pub fn provider_get_by_name(name: &str) -> Result<ProviderInfo> {
    let c_name = CString::new(name)?;
    let mut info = talu_sys::CProviderInfo::default();
    // SAFETY: c_name is a valid null-terminated string, info is a valid mutable pointer.
    let rc = unsafe { talu_sys::talu_provider_get_by_name(c_name.as_ptr(), &mut info) };
    if rc != 0 {
        return Err(error_from_last_or("Provider not found"));
    }
    Ok(ProviderInfo::from_c(&info))
}

/// Checks if a model ID string has a provider prefix (e.g., "openai/gpt-4").
pub fn has_provider_prefix(model_id: &str) -> bool {
    let Ok(c_str) = CString::new(model_id) else {
        return false;
    };
    // SAFETY: c_str is a valid null-terminated string.
    unsafe { talu_sys::talu_provider_has_prefix(c_str.as_ptr()) == 1 }
}

/// Parses a model target string to extract provider and model ID.
///
/// For example, "openai/gpt-4" returns provider info for "openai" and model_id "gpt-4".
pub fn parse_model_target(model: &str) -> Option<ParsedModelTarget> {
    let c_model = CString::new(model).ok()?;
    let mut info = talu_sys::CProviderInfo::default();
    let mut start: usize = 0;
    let mut len: usize = 0;

    // SAFETY: All pointers are valid. c_model is null-terminated.
    let code = unsafe {
        talu_sys::talu_provider_parse(
            c_model.as_ptr(),
            &mut info,
            &mut start as *mut _ as *mut std::ffi::c_void,
            &mut len as *mut _ as *mut std::ffi::c_void,
        )
    };

    if code != 0 {
        return None;
    }

    let model_id = if start < model.len() && len > 0 {
        model[start..start + len].to_string()
    } else {
        model.to_string()
    };

    Some(ParsedModelTarget {
        provider: ProviderInfo::from_c(&info),
        model_id,
    })
}

/// Returns a list of all registered providers.
pub fn list_providers() -> Vec<ProviderInfo> {
    let count = provider_count();
    (0..count).filter_map(|i| provider_get(i).ok()).collect()
}
