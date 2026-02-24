//! Safe wrappers for talu UI plugin discovery.

use crate::error::error_from_last_or;
use crate::Result;
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

/// Discovered plugin information.
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Plugin identifier (derived from dirname or filename without extension).
    pub plugin_id: String,
    /// Absolute path to the plugin directory.
    pub plugin_dir: String,
    /// Relative entry path within plugin_dir (e.g., "index.ts").
    pub entry_path: String,
    /// Raw JSON manifest string (talu.json content or inferred defaults).
    pub manifest_json: String,
    /// True if directory plugin, false if single-file.
    pub is_directory: bool,
}

impl PluginInfo {
    fn from_c(info: &talu_sys::CPluginInfo) -> Self {
        Self {
            plugin_id: c_str_to_string(info.plugin_id),
            plugin_dir: c_str_to_string(info.plugin_dir),
            entry_path: c_str_to_string(info.entry_path),
            manifest_json: c_str_to_string(info.manifest_json),
            is_directory: info.is_directory,
        }
    }
}

/// Return the default plugins directory (~/.talu/plugins/).
pub fn default_plugins_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("TALU_PLUGINS_DIR") {
        return PathBuf::from(dir);
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".talu")
        .join("plugins")
}

/// Scan a plugins directory and return discovered plugins.
///
/// If `plugins_dir` is `None`, uses the default (~/.talu/plugins/).
pub fn scan_plugins(plugins_dir: Option<&Path>) -> Result<Vec<PluginInfo>> {
    let c_dir = match plugins_dir {
        Some(dir) => {
            let s = dir.to_string_lossy();
            Some(CString::new(s.as_ref())?)
        }
        None => None,
    };

    let dir_ptr = c_dir.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

    let mut list: *mut talu_sys::CPluginList = std::ptr::null_mut();
    // SAFETY: dir_ptr is either null (use default) or a valid C string.
    // list is a valid out pointer.
    let rc = unsafe {
        talu_sys::talu_plugins_scan(dir_ptr, &mut list as *mut _)
    };
    if rc != 0 {
        return Err(error_from_last_or("Failed to scan plugins"));
    }

    let result = if list.is_null() {
        Vec::new()
    } else {
        // SAFETY: list is a valid pointer returned by talu_plugins_scan.
        let count = unsafe { talu_sys::talu_plugins_list_count(list) } as usize;
        let mut plugins = Vec::with_capacity(count);
        for i in 0..count {
            // SAFETY: index is within bounds.
            if let Some(info) = unsafe { talu_sys::talu_plugins_list_get(list, i as u32).as_ref() }
            {
                plugins.push(PluginInfo::from_c(info));
            }
        }
        plugins
    };

    // SAFETY: list was returned by talu_plugins_scan and must be freed.
    if !list.is_null() {
        unsafe { talu_sys::talu_plugins_list_free(list) };
    }

    Ok(result)
}

fn c_str_to_string(ptr: *const std::ffi::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: ptr is a valid null-terminated C string from the C API.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
