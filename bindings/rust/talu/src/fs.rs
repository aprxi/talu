//! Safe wrappers for workspace-scoped filesystem C APIs (`talu_fs_*`).

use std::ffi::{c_void, CString};
use std::os::raw::{c_char, c_int};

const ERROR_CODE_OK: i32 = 0;
const ERROR_CODE_IO_FILE_NOT_FOUND: i32 = 500;
const ERROR_CODE_IO_PERMISSION_DENIED: i32 = 501;
const ERROR_CODE_IO_PATH_INVALID: i32 = 505;
const ERROR_CODE_IO_PATH_OUTSIDE_WORKSPACE: i32 = 506;
const ERROR_CODE_IO_PARENT_NOT_FOUND: i32 = 507;
const ERROR_CODE_IO_IS_DIRECTORY: i32 = 508;
const ERROR_CODE_IO_NOT_DIRECTORY: i32 = 509;
const ERROR_CODE_IO_NOT_EMPTY: i32 = 510;
const ERROR_CODE_IO_FILE_TOO_BIG: i32 = 511;
const ERROR_CODE_INVALID_ARGUMENT: i32 = 901;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CTaluFsStat {
    exists: bool,
    is_file: bool,
    is_dir: bool,
    is_symlink: bool,
    size: u64,
    mode: u32,
    modified_at: i64,
    created_at: i64,
    _reserved: [u8; 32],
}

impl Default for CTaluFsStat {
    fn default() -> Self {
        Self {
            exists: false,
            is_file: false,
            is_dir: false,
            is_symlink: false,
            size: 0,
            mode: 0,
            modified_at: 0,
            created_at: 0,
            _reserved: [0; 32],
        }
    }
}

unsafe extern "C" {
    #[link_name = "talu_fs_create"]
    fn talu_fs_create_raw(workspace_dir: *const c_char, out_handle: *mut *mut c_void) -> c_int;
    #[link_name = "talu_fs_free"]
    fn talu_fs_free_raw(handle: *mut c_void);
    #[link_name = "talu_fs_read"]
    fn talu_fs_read_raw(
        handle: *mut c_void,
        path: *const c_char,
        max_bytes: usize,
        out_content: *mut *const u8,
        out_content_len: *mut usize,
        out_size: *mut u64,
        out_truncated: *mut bool,
    ) -> c_int;
    #[link_name = "talu_fs_write"]
    fn talu_fs_write_raw(
        handle: *mut c_void,
        path: *const c_char,
        content: *const u8,
        content_len: usize,
        mkdir: bool,
        out_bytes_written: *mut usize,
    ) -> c_int;
    #[link_name = "talu_fs_edit"]
    fn talu_fs_edit_raw(
        handle: *mut c_void,
        path: *const c_char,
        old_text: *const u8,
        old_len: usize,
        new_text: *const u8,
        new_len: usize,
        replace_all: bool,
        out_replacements: *mut usize,
    ) -> c_int;
    #[link_name = "talu_fs_stat"]
    fn talu_fs_stat_raw(
        handle: *mut c_void,
        path: *const c_char,
        out_stat: *mut CTaluFsStat,
    ) -> c_int;
    #[link_name = "talu_fs_list"]
    fn talu_fs_list_raw(
        handle: *mut c_void,
        path: *const c_char,
        glob: *const c_char,
        recursive: bool,
        limit: usize,
        out_json: *mut *const u8,
        out_json_len: *mut usize,
    ) -> c_int;
    #[link_name = "talu_fs_remove"]
    fn talu_fs_remove_raw(handle: *mut c_void, path: *const c_char, recursive: bool) -> c_int;
    #[link_name = "talu_fs_mkdir"]
    fn talu_fs_mkdir_raw(handle: *mut c_void, path: *const c_char, recursive: bool) -> c_int;
    #[link_name = "talu_fs_rename"]
    fn talu_fs_rename_raw(handle: *mut c_void, from: *const c_char, to: *const c_char) -> c_int;
    #[link_name = "talu_fs_free_string"]
    fn talu_fs_free_string_raw(ptr: *const u8, len: usize);
}

#[derive(Debug, Clone)]
pub enum FsError {
    InvalidArgument(String),
    InvalidPath(String),
    PermissionDenied(String),
    NotFound(String),
    ParentNotFound(String),
    IsDirectory(String),
    NotDirectory(String),
    NotEmpty(String),
    TooLarge(String),
    Io(String),
}

impl FsError {
    fn from_code(code: i32, fallback: &str) -> Self {
        let detail = crate::error::last_error_message().unwrap_or_else(|| fallback.to_string());
        match code {
            ERROR_CODE_INVALID_ARGUMENT => Self::InvalidArgument(detail),
            ERROR_CODE_IO_PATH_INVALID => Self::InvalidPath(detail),
            ERROR_CODE_IO_PATH_OUTSIDE_WORKSPACE | ERROR_CODE_IO_PERMISSION_DENIED => {
                Self::PermissionDenied(detail)
            }
            ERROR_CODE_IO_FILE_NOT_FOUND => Self::NotFound(detail),
            ERROR_CODE_IO_PARENT_NOT_FOUND => Self::ParentNotFound(detail),
            ERROR_CODE_IO_IS_DIRECTORY => Self::IsDirectory(detail),
            ERROR_CODE_IO_NOT_DIRECTORY => Self::NotDirectory(detail),
            ERROR_CODE_IO_NOT_EMPTY => Self::NotEmpty(detail),
            ERROR_CODE_IO_FILE_TOO_BIG => Self::TooLarge(detail),
            _ => Self::Io(detail),
        }
    }
}

impl std::fmt::Display for FsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            Self::InvalidPath(msg) => write!(f, "Invalid path: {msg}"),
            Self::PermissionDenied(msg) => write!(f, "Permission denied: {msg}"),
            Self::NotFound(msg) => write!(f, "Not found: {msg}"),
            Self::ParentNotFound(msg) => write!(f, "Parent not found: {msg}"),
            Self::IsDirectory(msg) => write!(f, "Path is a directory: {msg}"),
            Self::NotDirectory(msg) => write!(f, "Path is not a directory: {msg}"),
            Self::NotEmpty(msg) => write!(f, "Directory is not empty: {msg}"),
            Self::TooLarge(msg) => write!(f, "File too large: {msg}"),
            Self::Io(msg) => write!(f, "Filesystem error: {msg}"),
        }
    }
}

impl std::error::Error for FsError {}

#[derive(Debug, Clone)]
pub struct FsReadResult {
    pub content: Vec<u8>,
    pub size: u64,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsWriteResult {
    pub bytes_written: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsEditResult {
    pub replacements: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsStat {
    pub exists: bool,
    pub is_file: bool,
    pub is_dir: bool,
    pub is_symlink: bool,
    pub size: u64,
    pub mode: u32,
    pub modified_at: i64,
    pub created_at: i64,
}

/// RAII handle for workspace-scoped filesystem operations.
#[derive(Debug)]
pub struct FsHandle {
    handle: *mut c_void,
}

impl FsHandle {
    /// Open a filesystem handle rooted at `workspace_dir`.
    pub fn open(workspace_dir: &str) -> Result<Self, FsError> {
        let c_workspace = CString::new(workspace_dir).map_err(|_| {
            FsError::InvalidArgument("workspace_dir contains null byte".to_string())
        })?;

        let mut handle: *mut c_void = std::ptr::null_mut();
        // SAFETY: c_workspace and out pointer are valid for the duration of this call.
        let rc = unsafe { talu_fs_create_raw(c_workspace.as_ptr(), &mut handle) };
        if rc != ERROR_CODE_OK {
            return Err(FsError::from_code(rc, "failed to create filesystem handle"));
        }
        if handle.is_null() {
            return Err(FsError::Io(
                "filesystem handle create succeeded but returned null".to_string(),
            ));
        }

        Ok(Self { handle })
    }

    /// Read file bytes from `path`.
    pub fn read(&self, path: &str, max_bytes: usize) -> Result<FsReadResult, FsError> {
        let c_path = CString::new(path)
            .map_err(|_| FsError::InvalidArgument("path contains null byte".to_string()))?;

        let mut out_ptr: *const u8 = std::ptr::null();
        let mut out_len: usize = 0;
        let mut out_size: u64 = 0;
        let mut out_truncated = false;

        // SAFETY: handle and c_path are valid; output pointers are writable.
        let rc = unsafe {
            talu_fs_read_raw(
                self.handle,
                c_path.as_ptr(),
                max_bytes,
                &mut out_ptr,
                &mut out_len,
                &mut out_size,
                &mut out_truncated,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(FsError::from_code(rc, "failed to read file"));
        }

        let content = if out_ptr.is_null() || out_len == 0 {
            Vec::new()
        } else {
            // SAFETY: C API returns a valid buffer of out_len bytes on success.
            let bytes = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
            bytes.to_vec()
        };
        // SAFETY: pointer+len originate from talu_fs_read and are freed exactly once here.
        unsafe { talu_fs_free_string_raw(out_ptr, out_len) };

        Ok(FsReadResult {
            content,
            size: out_size,
            truncated: out_truncated,
        })
    }

    /// Write file bytes to `path`.
    pub fn write(&self, path: &str, content: &[u8], mkdir: bool) -> Result<FsWriteResult, FsError> {
        let c_path = CString::new(path)
            .map_err(|_| FsError::InvalidArgument("path contains null byte".to_string()))?;

        let content_ptr = if content.is_empty() {
            std::ptr::null()
        } else {
            content.as_ptr()
        };
        let mut written = 0usize;

        // SAFETY: pointers and lengths are valid for the call.
        let rc = unsafe {
            talu_fs_write_raw(
                self.handle,
                c_path.as_ptr(),
                content_ptr,
                content.len(),
                mkdir,
                &mut written,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(FsError::from_code(rc, "failed to write file"));
        }

        Ok(FsWriteResult {
            bytes_written: written,
        })
    }

    /// Edit file by replacing `old_text` with `new_text`.
    pub fn edit(
        &self,
        path: &str,
        old_text: &[u8],
        new_text: &[u8],
        replace_all: bool,
    ) -> Result<FsEditResult, FsError> {
        let c_path = CString::new(path)
            .map_err(|_| FsError::InvalidArgument("path contains null byte".to_string()))?;

        let old_ptr = if old_text.is_empty() {
            std::ptr::null()
        } else {
            old_text.as_ptr()
        };
        let new_ptr = if new_text.is_empty() {
            std::ptr::null()
        } else {
            new_text.as_ptr()
        };
        let mut replacements = 0usize;

        // SAFETY: pointers and lengths are valid for this call.
        let rc = unsafe {
            talu_fs_edit_raw(
                self.handle,
                c_path.as_ptr(),
                old_ptr,
                old_text.len(),
                new_ptr,
                new_text.len(),
                replace_all,
                &mut replacements,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(FsError::from_code(rc, "failed to edit file"));
        }

        Ok(FsEditResult { replacements })
    }

    /// Stat filesystem metadata for `path`.
    pub fn stat(&self, path: &str) -> Result<FsStat, FsError> {
        let c_path = CString::new(path)
            .map_err(|_| FsError::InvalidArgument("path contains null byte".to_string()))?;
        let mut out = CTaluFsStat::default();

        // SAFETY: handle/path are valid and out is writable.
        let rc = unsafe { talu_fs_stat_raw(self.handle, c_path.as_ptr(), &mut out) };
        if rc != ERROR_CODE_OK {
            return Err(FsError::from_code(rc, "failed to stat path"));
        }

        Ok(FsStat {
            exists: out.exists,
            is_file: out.is_file,
            is_dir: out.is_dir,
            is_symlink: out.is_symlink,
            size: out.size,
            mode: out.mode,
            modified_at: out.modified_at,
            created_at: out.created_at,
        })
    }

    /// List directory as JSON payload produced by core.
    pub fn list_json(
        &self,
        path: &str,
        glob: Option<&str>,
        recursive: bool,
        limit: usize,
    ) -> Result<String, FsError> {
        let c_path = CString::new(path)
            .map_err(|_| FsError::InvalidArgument("path contains null byte".to_string()))?;
        let c_glob = if let Some(g) = glob {
            Some(
                CString::new(g)
                    .map_err(|_| FsError::InvalidArgument("glob contains null byte".to_string()))?,
            )
        } else {
            None
        };

        let mut out_ptr: *const u8 = std::ptr::null();
        let mut out_len: usize = 0;
        // SAFETY: handle/path are valid; optional glob pointer stays alive for the call.
        let rc = unsafe {
            talu_fs_list_raw(
                self.handle,
                c_path.as_ptr(),
                c_glob.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
                recursive,
                limit,
                &mut out_ptr,
                &mut out_len,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(FsError::from_code(rc, "failed to list directory"));
        }

        let json = if out_ptr.is_null() || out_len == 0 {
            String::new()
        } else {
            // SAFETY: C API returns a valid UTF-8/bytes buffer of out_len bytes.
            let bytes = unsafe { std::slice::from_raw_parts(out_ptr, out_len) };
            String::from_utf8_lossy(bytes).into_owned()
        };
        // SAFETY: pointer+len originate from talu_fs_list and are freed exactly once here.
        unsafe { talu_fs_free_string_raw(out_ptr, out_len) };

        Ok(json)
    }

    /// Remove path (file or directory).
    pub fn remove(&self, path: &str, recursive: bool) -> Result<(), FsError> {
        let c_path = CString::new(path)
            .map_err(|_| FsError::InvalidArgument("path contains null byte".to_string()))?;

        // SAFETY: handle/path are valid for this call.
        let rc = unsafe { talu_fs_remove_raw(self.handle, c_path.as_ptr(), recursive) };
        if rc != ERROR_CODE_OK {
            return Err(FsError::from_code(rc, "failed to remove path"));
        }
        Ok(())
    }

    /// Create a directory at `path`.
    pub fn mkdir(&self, path: &str, recursive: bool) -> Result<(), FsError> {
        let c_path = CString::new(path)
            .map_err(|_| FsError::InvalidArgument("path contains null byte".to_string()))?;

        // SAFETY: handle/path are valid for this call.
        let rc = unsafe { talu_fs_mkdir_raw(self.handle, c_path.as_ptr(), recursive) };
        if rc != ERROR_CODE_OK {
            return Err(FsError::from_code(rc, "failed to mkdir"));
        }
        Ok(())
    }

    /// Rename `from` to `to`.
    pub fn rename(&self, from: &str, to: &str) -> Result<(), FsError> {
        let c_from = CString::new(from)
            .map_err(|_| FsError::InvalidArgument("from contains null byte".to_string()))?;
        let c_to = CString::new(to)
            .map_err(|_| FsError::InvalidArgument("to contains null byte".to_string()))?;

        // SAFETY: handle/paths are valid for this call.
        let rc = unsafe { talu_fs_rename_raw(self.handle, c_from.as_ptr(), c_to.as_ptr()) };
        if rc != ERROR_CODE_OK {
            return Err(FsError::from_code(rc, "failed to rename path"));
        }
        Ok(())
    }
}

impl Drop for FsHandle {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // SAFETY: handle is owned by this wrapper and freed exactly once on drop.
        unsafe { talu_fs_free_raw(self.handle) };
        self.handle = std::ptr::null_mut();
    }
}

#[cfg(test)]
mod tests {
    use super::FsHandle;

    #[test]
    fn fs_handle_roundtrip_read_write() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let workspace = tmp.path().to_string_lossy().to_string();
        let fs = FsHandle::open(&workspace).expect("open fs handle");

        fs.write("hello.txt", b"hello world", false).expect("write");
        let read = fs.read("hello.txt", 1024).expect("read");
        assert_eq!(read.content, b"hello world");
        assert_eq!(read.size, 11);
        assert!(!read.truncated);
    }
}
