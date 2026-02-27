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

// =============================================================================
// Provider Config (stateless, path-based runtime configuration)
// =============================================================================

// Re-declare FFI functions with correct pointer/return-type signatures where
// the auto-generated bindings differ from the actual C ABI.
unsafe extern "C" {
    #[link_name = "talu_provider_config_list_free"]
    fn talu_provider_config_list_free_raw(list: *mut talu_sys::CProviderConfigList);
    #[link_name = "talu_provider_config_list_remote_models"]
    fn talu_provider_config_list_remote_models_raw(
        db_root: *const std::os::raw::c_char,
    ) -> talu_sys::CRemoteModelListResult;
    #[link_name = "talu_provider_config_resolve_credentials_free"]
    fn talu_provider_config_resolve_credentials_free_raw(
        creds: *mut talu_sys::CProviderCredentials,
    );
    // The binding generator maps Zig i8 to *mut c_void; redeclare with correct type.
    #[link_name = "talu_provider_config_set"]
    fn talu_provider_config_set_raw(
        db_root: *const std::os::raw::c_char,
        name: *const std::os::raw::c_char,
        enabled: i8,
        api_key: *const std::os::raw::c_char,
        base_url: *const std::os::raw::c_char,
    ) -> std::os::raw::c_int;

    // Health check — returns by value, declare manually.
    #[link_name = "talu_provider_config_health"]
    fn talu_provider_config_health_raw(
        db_root: *const std::os::raw::c_char,
        name: *const std::os::raw::c_char,
    ) -> CProviderHealthResult;
    #[link_name = "talu_provider_config_health_free"]
    fn talu_provider_config_health_free_raw(result: *mut CProviderHealthResult);
}

/// C-ABI health check result (matches Zig CProviderHealthResult).
#[repr(C)]
struct CProviderHealthResult {
    ok: u8,
    model_count: usize,
    error_message: *const std::os::raw::c_char,
}

/// Provider with merged runtime configuration.
#[derive(Debug, Clone)]
pub struct ProviderWithConfig {
    pub name: String,
    pub default_endpoint: String,
    pub api_key_env: Option<String>,
    pub enabled: bool,
    pub has_api_key: bool,
    pub base_url_override: Option<String>,
    pub effective_endpoint: String,
}

impl ProviderWithConfig {
    fn from_c(c: &talu_sys::CProviderWithConfig) -> Self {
        Self {
            name: c_ptr_to_string(c.name),
            default_endpoint: c_ptr_to_string(c.default_endpoint),
            api_key_env: c_ptr_to_option_string(c.api_key_env),
            enabled: c.enabled != 0,
            has_api_key: c.has_api_key != 0,
            base_url_override: c_ptr_to_option_string(c.base_url_override),
            effective_endpoint: c_ptr_to_string(c.effective_endpoint),
        }
    }
}

/// List all providers with their merged runtime configuration.
///
/// `db_root` is the KV store root path (e.g., "~/.talu/db/default/kv").
/// Stateless: no init/deinit required.
pub fn provider_config_list(db_root: &str) -> Result<Vec<ProviderWithConfig>> {
    let c_root = CString::new(db_root)?;
    // SAFETY: c_root is a valid null-terminated string.
    let mut list = unsafe { talu_sys::talu_provider_config_list(c_root.as_ptr()) };
    if list.error_code != 0 {
        // SAFETY: list was returned by talu_provider_config_list.
        unsafe { talu_provider_config_list_free_raw(&mut list) };
        return Err(error_from_last_or("Failed to list provider configs"));
    }

    let mut result = Vec::with_capacity(list.count);
    if !list.items.is_null() && list.count > 0 {
        // SAFETY: Non-null items ptr with valid count from C API.
        let slice = unsafe { std::slice::from_raw_parts(list.items, list.count) };
        for item in slice {
            result.push(ProviderWithConfig::from_c(item));
        }
    }

    // SAFETY: list was returned by talu_provider_config_list.
    unsafe { talu_provider_config_list_free_raw(&mut list) };
    Ok(result)
}

/// Update configuration for a named provider using merge semantics.
///
/// `db_root` is the KV store root path.
/// `enabled`: `None` = keep existing, `Some(true/false)` = set.
/// `api_key`: `None` = keep existing, `Some("")` = clear, `Some(val)` = set.
/// `base_url`: `None` = keep existing, `Some("")` = clear, `Some(val)` = set.
pub fn provider_config_set(
    db_root: &str,
    name: &str,
    enabled: Option<bool>,
    api_key: Option<&str>,
    base_url: Option<&str>,
) -> Result<()> {
    let c_root = CString::new(db_root)?;
    let c_name = CString::new(name)?;
    let c_api_key = api_key.map(CString::new).transpose()?;
    let c_base_url = base_url.map(CString::new).transpose()?;

    let api_key_ptr = c_api_key.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let base_url_ptr = c_base_url.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

    // Map Option<bool> to i8: None=-1 (keep), Some(false)=0, Some(true)=1.
    let enabled_i8: i8 = match enabled {
        None => -1,
        Some(false) => 0,
        Some(true) => 1,
    };

    // SAFETY: All pointers are valid null-terminated strings or null.
    // Uses the corrected FFI declaration (i8 instead of *mut c_void).
    let rc = unsafe {
        talu_provider_config_set_raw(
            c_root.as_ptr(),
            c_name.as_ptr(),
            enabled_i8,
            api_key_ptr,
            base_url_ptr,
        )
    };
    if rc != 0 {
        return Err(error_from_last_or("Failed to set provider config"));
    }
    Ok(())
}

/// List models from all enabled remote providers.
///
/// `db_root` is the KV store root path.
/// Each model ID is prefixed with `{provider}::` (e.g., `openai::gpt-4o`).
pub fn provider_config_list_remote_models(db_root: &str) -> Result<Vec<crate::RemoteModelInfo>> {
    let c_root = CString::new(db_root)?;
    // SAFETY: c_root is a valid null-terminated string. We use the corrected signature.
    let mut result = unsafe { talu_provider_config_list_remote_models_raw(c_root.as_ptr()) };
    if result.error_code != 0 {
        // SAFETY: result was returned by talu_provider_config_list_remote_models.
        unsafe { talu_sys::talu_provider_config_list_remote_models_free(&mut result) };
        return Err(error_from_last_or("Failed to list remote models"));
    }

    let mut models = Vec::with_capacity(result.count);
    if !result.models.is_null() && result.count > 0 {
        // SAFETY: Non-null models ptr with valid count from C API.
        let slice = unsafe { std::slice::from_raw_parts(result.models, result.count) };
        for info in slice {
            models.push(crate::RemoteModelInfo {
                id: c_ptr_to_string(info.id),
                object: c_ptr_to_string(info.object),
                created: info.created,
                owned_by: c_ptr_to_string(info.owned_by),
            });
        }
    }

    // SAFETY: result was returned by talu_provider_config_list_remote_models.
    unsafe { talu_sys::talu_provider_config_list_remote_models_free(&mut result) };
    Ok(models)
}

/// Resolved credentials for a provider (endpoint + API key).
#[derive(Debug, Clone)]
pub struct ProviderCredentials {
    pub effective_endpoint: String,
    pub api_key: Option<String>,
}

/// Resolve the effective endpoint and API key for a named provider.
///
/// `db_root` is the KV store root path.
/// Returns the resolved endpoint (config > env > default) and API key (config > env > None).
pub fn provider_config_resolve_credentials(
    db_root: &str,
    name: &str,
) -> Result<ProviderCredentials> {
    let c_root = CString::new(db_root)?;
    let c_name = CString::new(name)?;
    // SAFETY: Both pointers are valid null-terminated strings.
    let mut creds = unsafe {
        talu_sys::talu_provider_config_resolve_credentials(c_root.as_ptr(), c_name.as_ptr())
    };
    if creds.error_code != 0 {
        // SAFETY: creds was returned by talu_provider_config_resolve_credentials.
        unsafe { talu_provider_config_resolve_credentials_free_raw(&mut creds) };
        return Err(error_from_last_or("Failed to resolve provider credentials"));
    }

    let result = ProviderCredentials {
        effective_endpoint: c_ptr_to_string(creds.effective_endpoint),
        api_key: c_ptr_to_option_string(creds.api_key),
    };

    // SAFETY: creds was returned by talu_provider_config_resolve_credentials.
    unsafe { talu_provider_config_resolve_credentials_free_raw(&mut creds) };
    Ok(result)
}

/// Result of a provider health check.
#[derive(Debug, Clone)]
pub struct ProviderHealthResult {
    pub ok: bool,
    pub model_count: usize,
    pub error_message: Option<String>,
}

/// Check connectivity to a provider by hitting its /models endpoint.
///
/// `db_root` is the KV store root path.
/// Returns health result (ok + model_count or error_message).
pub fn provider_config_health(db_root: &str, name: &str) -> ProviderHealthResult {
    let Ok(c_root) = CString::new(db_root) else {
        return ProviderHealthResult {
            ok: false,
            model_count: 0,
            error_message: Some("Invalid db_root".to_string()),
        };
    };
    let Ok(c_name) = CString::new(name) else {
        return ProviderHealthResult {
            ok: false,
            model_count: 0,
            error_message: Some("Invalid provider name".to_string()),
        };
    };

    // SAFETY: Both pointers are valid null-terminated strings.
    let mut result =
        unsafe { talu_provider_config_health_raw(c_root.as_ptr(), c_name.as_ptr()) };

    let health = ProviderHealthResult {
        ok: result.ok != 0,
        model_count: result.model_count,
        error_message: if result.error_message.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(result.error_message) }
                    .to_string_lossy()
                    .into_owned(),
            )
        },
    };

    // SAFETY: result was returned by talu_provider_config_health.
    unsafe { talu_provider_config_health_free_raw(&mut result) };
    health
}

fn c_ptr_to_string(ptr: *const std::os::raw::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        // SAFETY: ptr is a valid C string from the C API.
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

fn c_ptr_to_option_string(ptr: *const std::os::raw::c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        // SAFETY: ptr is a valid C string from the C API.
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}
