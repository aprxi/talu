//! Safe wrappers for talu repository/cache operations.

use crate::error::error_from_last_or;
use crate::Result;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;

/// Where a cached model originated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOrigin {
    /// HuggingFace Hub cache (~/.cache/huggingface/hub/).
    Hub,
    /// Talu managed cache (~/.cache/talu/models/).
    Managed,
}

/// A cached model entry.
#[derive(Debug, Clone)]
pub struct CachedModel {
    /// The model ID (e.g., "meta-llama/Llama-3.2-1B").
    pub id: String,
    /// The local filesystem path to the cached model.
    pub path: String,
    /// Where this model originated from.
    pub source: CacheOrigin,
}

/// Options for downloading a model.
#[derive(Debug, Clone, Default)]
pub struct DownloadOptions {
    /// HuggingFace token for private repos.
    pub token: Option<String>,
    /// Force re-download even if cached.
    pub force: bool,
    /// Custom endpoint URL.
    pub endpoint_url: Option<String>,
    /// Skip downloading weight files (.safetensors).
    pub skip_weights: bool,
}

/// Progress action type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressAction {
    /// Adding a new progress line.
    Add,
    /// Updating an existing progress line.
    Update,
    /// Completing/removing a progress line.
    Complete,
}

impl From<talu_sys::ProgressAction> for ProgressAction {
    fn from(action: talu_sys::ProgressAction) -> Self {
        match action {
            talu_sys::ProgressAction::Add => ProgressAction::Add,
            talu_sys::ProgressAction::Update => ProgressAction::Update,
            talu_sys::ProgressAction::Complete => ProgressAction::Complete,
        }
    }
}

/// Progress update during a download.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// The action for this update.
    pub action: ProgressAction,
    /// Line ID for multi-line progress.
    pub line_id: u8,
    /// Name/label of the current file being downloaded.
    pub label: String,
    /// Progress message.
    pub message: String,
    /// Bytes downloaded so far.
    pub current: u64,
    /// Total bytes to download.
    pub total: u64,
}

/// Callback type for download progress updates.
pub type ProgressCallback = Box<dyn FnMut(DownloadProgress) + Send>;

/// Returns the HuggingFace home directory path (e.g., ~/.cache/huggingface).
/// Respects HF_HOME environment variable.
pub fn get_hf_home() -> Result<String> {
    let mut out: *mut c_char = std::ptr::null_mut();

    // SAFETY: out is a valid mutable pointer.
    let rc = unsafe { talu_sys::talu_repo_get_hf_home(&mut out as *mut _ as *mut c_void) };

    if rc != 0 || out.is_null() {
        return Err(error_from_last_or("Failed to get HF home"));
    }

    // SAFETY: out is a valid C string returned by the C API.
    let path = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();

    // SAFETY: out was allocated by talu and must be freed.
    unsafe { talu_sys::talu_text_free(out) };

    Ok(path)
}

/// Returns the Talu home directory path (e.g., ~/.cache/talu).
/// Respects TALU_HOME environment variable.
pub fn get_talu_home() -> Result<String> {
    let mut out: *mut c_char = std::ptr::null_mut();

    // SAFETY: out is a valid mutable pointer.
    let rc = unsafe { talu_sys::talu_repo_get_talu_home(&mut out as *mut _ as *mut c_void) };

    if rc != 0 || out.is_null() {
        return Err(error_from_last_or("Failed to get Talu home"));
    }

    // SAFETY: out is a valid C string returned by the C API.
    let path = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();

    // SAFETY: out was allocated by talu and must be freed.
    unsafe { talu_sys::talu_text_free(out) };

    Ok(path)
}

/// Returns the size in bytes of a cached model.
pub fn repo_size(model_id: &str) -> u64 {
    let Ok(c_str) = CString::new(model_id) else {
        return 0;
    };
    // SAFETY: c_str is a valid null-terminated string.
    unsafe { talu_sys::talu_repo_size(c_str.as_ptr()) }
}

/// Returns the modification time (Unix timestamp) of a cached model.
pub fn repo_mtime(model_id: &str) -> i64 {
    let Ok(c_str) = CString::new(model_id) else {
        return 0;
    };
    // SAFETY: c_str is a valid null-terminated string.
    unsafe { talu_sys::talu_repo_mtime(c_str.as_ptr()) }
}

/// Checks if a model is in the local cache (HuggingFace Hub or Talu cache).
pub fn repo_is_cached(model_id: &str) -> bool {
    let Ok(c_str) = CString::new(model_id) else {
        return false;
    };
    // SAFETY: c_str is a valid null-terminated string.
    unsafe { talu_sys::talu_repo_is_cached(c_str.as_ptr()) != 0 }
}

/// Checks if the cache directory exists for a model.
pub fn repo_cache_dir_exists(model_id: &str) -> bool {
    let Ok(c_str) = CString::new(model_id) else {
        return false;
    };
    // SAFETY: c_str is a valid null-terminated string.
    unsafe { talu_sys::talu_repo_cache_dir_exists(c_str.as_ptr()) != 0 }
}

/// Deletes a cached model. Returns true if deletion was successful.
pub fn repo_delete(model_id: &str) -> bool {
    let Ok(c_str) = CString::new(model_id) else {
        return false;
    };
    // SAFETY: c_str is a valid null-terminated string.
    unsafe { talu_sys::talu_repo_delete(c_str.as_ptr()) != 0 }
}

/// Checks if a string looks like a model ID (e.g., "owner/repo").
pub fn is_model_id(value: &str) -> bool {
    let Ok(c_str) = CString::new(value) else {
        return false;
    };
    // SAFETY: c_str is a valid null-terminated string.
    unsafe { talu_sys::talu_repo_is_model_id(c_str.as_ptr()) != 0 }
}

/// Gets the cached local path for a model, if it exists.
pub fn repo_get_cached_path(model_id: &str) -> Result<String> {
    repo_get_cached_path_ex(model_id, true)
}

/// Returns the cached path for a model, optionally requiring weights.
pub fn repo_get_cached_path_ex(model_id: &str, require_weights: bool) -> Result<String> {
    let c_str = CString::new(model_id)?;
    let mut out: *mut c_char = std::ptr::null_mut();

    // SAFETY: c_str is valid, out is a valid mutable pointer.
    let rc = unsafe {
        talu_sys::talu_repo_get_cached_path(
            c_str.as_ptr(),
            require_weights,
            &mut out as *mut _ as *mut c_void,
        )
    };

    if rc != 0 || out.is_null() {
        return Err(error_from_last_or("Model not cached"));
    }

    // SAFETY: out is a valid C string returned by the C API.
    let path = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();

    // SAFETY: out was allocated by talu and must be freed.
    unsafe { talu_sys::talu_text_free(out) };

    Ok(path)
}

/// Resolves a model path (handles model IDs, local paths, URLs).
/// May download from HuggingFace if not cached.
pub fn resolve_model_path(path: &str) -> Result<String> {
    resolve_model_path_ex(path, false)
}

/// Resolves a model path, optionally in offline mode (cache-only, no downloads).
pub fn resolve_model_path_ex(path: &str, offline: bool) -> Result<String> {
    resolve_model_path_full(path, offline, true)
}

/// Resolves a model path with full control over offline mode and weight requirements.
pub fn resolve_model_path_full(
    path: &str,
    offline: bool,
    require_weights: bool,
) -> Result<String> {
    let c_str = CString::new(path)?;
    let mut out: *mut c_char = std::ptr::null_mut();

    // SAFETY: c_str is valid, out is a valid mutable pointer.
    let rc = unsafe {
        talu_sys::talu_repo_resolve_path(
            c_str.as_ptr(),
            offline,
            std::ptr::null(),
            std::ptr::null(),
            require_weights,
            &mut out as *mut _ as *mut c_void,
        )
    };

    if rc != 0 || out.is_null() {
        return Err(error_from_last_or("Failed to resolve model path"));
    }

    // SAFETY: out is a valid C string returned by the C API.
    let resolved = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();

    // SAFETY: out was allocated by talu and must be freed.
    unsafe { talu_sys::talu_text_free(out) };

    Ok(resolved)
}

/// Lists all cached models.
pub fn repo_list_models(require_weights: bool) -> Result<Vec<CachedModel>> {
    let mut list: *mut talu_sys::CachedModelList = std::ptr::null_mut();

    // SAFETY: list is a valid mutable pointer.
    let rc = unsafe {
        talu_sys::talu_repo_list_models(require_weights, &mut list as *mut _ as *mut *mut c_void)
    };

    if rc != 0 || list.is_null() {
        return Err(error_from_last_or("Could not list cached models"));
    }

    // SAFETY: list is valid after successful call.
    let count = unsafe { talu_sys::talu_repo_list_count(list) };
    let mut out = Vec::with_capacity(count);

    for idx in 0..count {
        let mut id_ptr: *const c_char = std::ptr::null();
        let mut path_ptr: *const c_char = std::ptr::null();

        // SAFETY: list is valid, idx is in bounds, pointers are valid.
        unsafe {
            talu_sys::talu_repo_list_get_id(list, idx, &mut id_ptr as *mut _ as *mut c_void);
            talu_sys::talu_repo_list_get_path(list, idx, &mut path_ptr as *mut _ as *mut c_void);
        }

        if !id_ptr.is_null() {
            // SAFETY: id_ptr is a valid C string from the list.
            let id = unsafe { CStr::from_ptr(id_ptr) }
                .to_string_lossy()
                .into_owned();

            let path = if !path_ptr.is_null() {
                // SAFETY: path_ptr is a valid C string from the list.
                unsafe { CStr::from_ptr(path_ptr) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                String::new()
            };

            // SAFETY: list is valid, idx is in bounds.
            let source_val = unsafe { talu_sys::talu_repo_list_get_source(list, idx) };
            let source = match source_val {
                1 => CacheOrigin::Managed,
                _ => CacheOrigin::Hub,
            };

            out.push(CachedModel { id, path, source });
        }
    }

    // SAFETY: list was allocated by talu and must be freed.
    unsafe { talu_sys::talu_repo_list_free(list) };

    Ok(out)
}

/// Lists files in a remote repository.
pub fn repo_list_files(model_path: &str, token: Option<&str>) -> Result<Vec<String>> {
    let c_path = CString::new(model_path)?;
    let c_token = token.map(|t| CString::new(t)).transpose()?;
    let token_ptr = c_token
        .as_ref()
        .map(|t| t.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut list: *mut talu_sys::StringList = std::ptr::null_mut();

    // SAFETY: All pointers are valid or null.
    let rc = unsafe {
        talu_sys::talu_repo_list(
            c_path.as_ptr(),
            token_ptr,
            &mut list as *mut _ as *mut *mut c_void,
        )
    };

    if rc != 0 || list.is_null() {
        return Err(error_from_last_or(&format!(
            "Could not list files for {}",
            model_path
        )));
    }

    let result = string_list_to_vec(list);

    // SAFETY: list was allocated by talu and must be freed.
    unsafe { talu_sys::talu_repo_string_list_free(list) };

    Ok(result)
}

/// Searches for models on HuggingFace Hub.
pub fn repo_search(query: &str, limit: usize, token: Option<&str>) -> Result<Vec<String>> {
    let c_query = CString::new(query)?;
    let c_token = token.map(|t| CString::new(t)).transpose()?;
    let token_ptr = c_token
        .as_ref()
        .map(|t| t.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut list: *mut talu_sys::StringList = std::ptr::null_mut();

    // SAFETY: All pointers are valid or null.
    let rc = unsafe {
        talu_sys::talu_repo_search(
            c_query.as_ptr(),
            limit,
            token_ptr,
            std::ptr::null(),
            &mut list as *mut _ as *mut *mut c_void,
        )
    };

    if rc != 0 || list.is_null() {
        return Err(error_from_last_or("Search failed"));
    }

    let result = string_list_to_vec(list);

    // SAFETY: list was allocated by talu and must be freed.
    unsafe { talu_sys::talu_repo_string_list_free(list) };

    Ok(result)
}

/// Helper to convert a StringList to Vec<String>.
fn string_list_to_vec(list: *mut talu_sys::StringList) -> Vec<String> {
    // SAFETY: list is assumed to be valid (caller's responsibility).
    let count = unsafe { talu_sys::talu_repo_string_list_count(list) };
    let mut out = Vec::with_capacity(count);

    for idx in 0..count {
        let mut ptr: *const c_char = std::ptr::null();
        // SAFETY: list is valid, idx is in bounds, ptr is a valid mutable pointer.
        unsafe {
            talu_sys::talu_repo_string_list_get(list, idx, &mut ptr as *mut _ as *mut c_void);
        }
        if !ptr.is_null() {
            // SAFETY: ptr is a valid C string from the list.
            out.push(
                unsafe { CStr::from_ptr(ptr) }
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }

    out
}

/// Callback wrapper for C API progress callback.
struct ProgressContext {
    callback: ProgressCallback,
}

extern "C" fn progress_callback_wrapper(
    update: *const talu_sys::ProgressUpdate,
    user_data: *mut c_void,
) {
    if update.is_null() || user_data.is_null() {
        return;
    }

    // SAFETY: user_data is a valid pointer to ProgressContext created by repo_fetch.
    let ctx = unsafe { &mut *(user_data as *mut ProgressContext) };

    // SAFETY: update is a valid pointer passed from C.
    let update_ref = unsafe { &*update };

    let label = if update_ref.label.is_null() {
        String::new()
    } else {
        // SAFETY: label is a valid C string from the C API.
        unsafe { CStr::from_ptr(update_ref.label) }
            .to_string_lossy()
            .into_owned()
    };

    let message = if update_ref.message.is_null() {
        String::new()
    } else {
        // SAFETY: message is a valid C string from the C API.
        unsafe { CStr::from_ptr(update_ref.message) }
            .to_string_lossy()
            .into_owned()
    };

    let progress = DownloadProgress {
        action: ProgressAction::from(update_ref.action),
        line_id: update_ref.line_id,
        label,
        message,
        current: update_ref.current,
        total: update_ref.total,
    };

    (ctx.callback)(progress);
}

/// Fetches/downloads a model from HuggingFace Hub.
pub fn repo_fetch(
    model_id: &str,
    options: DownloadOptions,
    callback: Option<ProgressCallback>,
) -> Result<String> {
    let c_model = CString::new(model_id)?;
    let c_token = options
        .token
        .as_ref()
        .map(|t| CString::new(t.as_str()))
        .transpose()?;
    let c_endpoint = options
        .endpoint_url
        .as_ref()
        .map(|e| CString::new(e.as_str()))
        .transpose()?;

    let mut ctx = callback.map(|cb| Box::new(ProgressContext { callback: cb }));
    let ctx_ptr = ctx
        .as_mut()
        .map(|c| c.as_mut() as *mut ProgressContext as *mut c_void)
        .unwrap_or(std::ptr::null_mut());

    let progress_cb = if ctx.is_some() {
        progress_callback_wrapper as *mut c_void
    } else {
        std::ptr::null_mut()
    };

    let mut c_options = talu_sys::DownloadOptions {
        token: c_token
            .as_ref()
            .map(|t| t.as_ptr())
            .unwrap_or(std::ptr::null()),
        progress_callback: progress_cb,
        user_data: ctx_ptr,
        force: options.force,
        endpoint_url: c_endpoint
            .as_ref()
            .map(|e| e.as_ptr())
            .unwrap_or(std::ptr::null()),
        skip_weights: options.skip_weights,
    };

    let mut out: *mut c_char = std::ptr::null_mut();

    // SAFETY: All pointers are valid or null, c_model is null-terminated.
    let rc = unsafe {
        talu_sys::talu_repo_fetch(
            c_model.as_ptr(),
            &mut c_options,
            &mut out as *mut _ as *mut c_void,
        )
    };

    if rc != 0 || out.is_null() {
        return Err(error_from_last_or(&format!("Failed to fetch {}", model_id)));
    }

    // SAFETY: out is a valid C string returned by the C API.
    let path = unsafe { CStr::from_ptr(out) }
        .to_string_lossy()
        .into_owned();

    // SAFETY: out was allocated by talu and must be freed.
    unsafe { talu_sys::talu_text_free(out) };

    Ok(path)
}

/// Returns the total size in bytes of all cached models.
pub fn repo_total_size() -> u64 {
    // SAFETY: talu_repo_total_size is a simple getter with no preconditions.
    unsafe { talu_sys::talu_repo_total_size() }
}

/// A rich search result from HuggingFace Hub.
#[derive(Debug, Clone)]
pub struct HfSearchResult {
    pub model_id: String,
    pub downloads: i64,
    pub likes: i64,
    pub last_modified: String,
    pub pipeline_tag: String,
    /// Total parameter count (from safetensors metadata). 0 if unknown.
    pub params_total: i64,
}

/// Sort mode for HuggingFace search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSort {
    Trending,
    Downloads,
    Likes,
    LastModified,
}

impl SearchSort {
    fn to_u8(self) -> u8 {
        match self {
            SearchSort::Trending => 0,
            SearchSort::Downloads => 1,
            SearchSort::Likes => 2,
            SearchSort::LastModified => 3,
        }
    }
}

/// Sort direction for HuggingFace search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchDirection {
    Descending,
    Ascending,
}

impl SearchDirection {
    fn to_u8(self) -> u8 {
        match self {
            SearchDirection::Descending => 0,
            SearchDirection::Ascending => 1,
        }
    }
}

/// Searches HuggingFace Hub with rich metadata (downloads, likes, dates).
pub fn repo_search_rich(
    query: &str,
    limit: usize,
    token: Option<&str>,
    filter: Option<&str>,
    sort: SearchSort,
    direction: SearchDirection,
    library: Option<&str>,
) -> Result<Vec<HfSearchResult>> {
    let c_query = CString::new(query)?;
    let c_token = token.map(|t| CString::new(t)).transpose()?;
    let c_filter = filter.map(|f| CString::new(f)).transpose()?;
    let c_library = library.map(|l| CString::new(l)).transpose()?;

    let token_ptr = c_token
        .as_ref()
        .map(|t| t.as_ptr())
        .unwrap_or(std::ptr::null());
    let filter_ptr = c_filter
        .as_ref()
        .map(|f| f.as_ptr())
        .unwrap_or(std::ptr::null());
    let library_ptr = c_library
        .as_ref()
        .map(|l| l.as_ptr())
        .unwrap_or(std::ptr::null());

    let mut list: *mut c_void = std::ptr::null_mut();

    // SAFETY: All pointers are valid or null.
    let rc = unsafe {
        talu_sys::talu_repo_search_rich(
            c_query.as_ptr(),
            limit,
            token_ptr,
            std::ptr::null(), // endpoint_url
            filter_ptr,
            sort.to_u8(),
            direction.to_u8(),
            library_ptr,
            &mut list,
        )
    };

    if rc != 0 || list.is_null() {
        return Err(error_from_last_or("Rich search failed"));
    }

    // SAFETY: list is valid after successful call.
    let count = unsafe { talu_sys::talu_repo_search_result_count(list) };
    let mut out = Vec::with_capacity(count);

    for idx in 0..count {
        let mut id_ptr: *const c_char = std::ptr::null();
        let mut modified_ptr: *const c_char = std::ptr::null();
        let mut tag_ptr: *const c_char = std::ptr::null();

        // SAFETY: list is valid, idx is in bounds, pointers are valid.
        unsafe {
            talu_sys::talu_repo_search_result_get_id(
                list,
                idx,
                &mut id_ptr as *mut _ as *mut c_void,
            );
            talu_sys::talu_repo_search_result_get_last_modified(
                list,
                idx,
                &mut modified_ptr as *mut _ as *mut c_void,
            );
            talu_sys::talu_repo_search_result_get_pipeline_tag(
                list,
                idx,
                &mut tag_ptr as *mut _ as *mut c_void,
            );
        }

        let model_id = if !id_ptr.is_null() {
            // SAFETY: id_ptr is a valid C string from the list.
            unsafe { CStr::from_ptr(id_ptr) }
                .to_string_lossy()
                .into_owned()
        } else {
            continue;
        };

        let last_modified = if !modified_ptr.is_null() {
            // SAFETY: modified_ptr is a valid C string from the list.
            unsafe { CStr::from_ptr(modified_ptr) }
                .to_string_lossy()
                .into_owned()
        } else {
            String::new()
        };

        let pipeline_tag = if !tag_ptr.is_null() {
            // SAFETY: tag_ptr is a valid C string from the list.
            unsafe { CStr::from_ptr(tag_ptr) }
                .to_string_lossy()
                .into_owned()
        } else {
            String::new()
        };

        // SAFETY: list is valid, idx is in bounds.
        let downloads = unsafe { talu_sys::talu_repo_search_result_get_downloads(list, idx) };
        let likes = unsafe { talu_sys::talu_repo_search_result_get_likes(list, idx) };
        let params_total = unsafe { talu_sys::talu_repo_search_result_get_params(list, idx) };

        out.push(HfSearchResult {
            model_id,
            downloads,
            likes,
            last_modified,
            pipeline_tag,
            params_total,
        });
    }

    // SAFETY: list was allocated by talu and must be freed.
    unsafe { talu_sys::talu_repo_search_result_free(list) };

    Ok(out)
}
