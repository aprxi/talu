//! Safe wrappers for core collab resource APIs.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::{c_char, c_int};

const ERROR_CODE_OK: i32 = 0;
const ERROR_CODE_RESOURCE_BUSY: i32 = 701;
const ERROR_CODE_RESOURCE_EXHAUSTED: i32 = 905;
const ERROR_CODE_INVALID_ARGUMENT: i32 = 901;

#[repr(C)]
#[derive(Default)]
struct CCollabSession {
    namespace: *const c_char,
    participant_id: *const c_char,
    participant_kind: u8,
    _flags_reserved: [u8; 7],
    status: *const c_char,
    _reserved: [u8; 8],
    _arena: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct CCollabSummary {
    namespace: *const c_char,
    meta_json: *const c_char,
    total_live_entries: usize,
    batched_pending: usize,
    ephemeral_live_entries: usize,
    watch_published: u64,
    _reserved: [u8; 8],
    _arena: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct CCollabValue {
    data: *mut u8,
    len: usize,
    updated_at_ms: i64,
    found: bool,
    _reserved: [u8; 7],
    _arena: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct CCollabOpResult {
    op_key: *const c_char,
    accepted: bool,
    _flags_reserved: [u8; 7],
    _arena: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct CCollabHistoryEntry {
    actor_id: *const c_char,
    actor_seq: u64,
    op_id: *const c_char,
    payload: *const u8,
    payload_len: usize,
    updated_at_ms: i64,
    key: *const c_char,
    _reserved: [u8; 8],
}

#[repr(C)]
#[derive(Default)]
struct CCollabHistoryList {
    items: *mut CCollabHistoryEntry,
    count: usize,
    _arena: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct CCollabWatchEvent {
    seq: u64,
    event_type: u8,
    durability_class: u8,
    has_durability: bool,
    ttl_ms: u64,
    has_ttl: bool,
    _flags_reserved: [u8; 6],
    key: *const c_char,
    value_len: usize,
    updated_at_ms: i64,
    _reserved: [u8; 8],
}

#[repr(C)]
#[derive(Default)]
struct CCollabWatchBatch {
    items: *mut CCollabWatchEvent,
    count: usize,
    lost: bool,
    _flags_reserved: [u8; 7],
    _arena: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct CCollabWatchWaitResult {
    published_seq: u64,
    timed_out: bool,
    _flags_reserved: [u8; 7],
}

unsafe extern "C" {
    #[link_name = "talu_collab_init"]
    fn talu_collab_init_raw(
        db_path: *const c_char,
        resource_kind: *const c_char,
        resource_id: *const c_char,
        out_handle: *mut *mut c_void,
    ) -> c_int;
    #[link_name = "talu_collab_free"]
    fn talu_collab_free_raw(handle: *mut c_void);
    #[link_name = "talu_collab_open_session"]
    fn talu_collab_open_session_raw(
        handle: *mut c_void,
        participant_id: *const c_char,
        participant_kind: u8,
        role: *const c_char,
        out_session: *mut CCollabSession,
    ) -> c_int;
    #[link_name = "talu_collab_get_summary"]
    fn talu_collab_get_summary_raw(handle: *mut c_void, out_summary: *mut CCollabSummary) -> c_int;
    #[link_name = "talu_collab_get_snapshot"]
    fn talu_collab_get_snapshot_raw(handle: *mut c_void, out_value: *mut CCollabValue) -> c_int;
    #[link_name = "talu_collab_submit_op"]
    fn talu_collab_submit_op_raw(
        handle: *mut c_void,
        actor_id: *const c_char,
        actor_seq: u64,
        op_id: *const c_char,
        payload: *const u8,
        payload_len: usize,
        issued_at_ms: i64,
        has_issued_at_ms: bool,
        snapshot: *const u8,
        snapshot_len: usize,
        has_snapshot: bool,
        out_result: *mut CCollabOpResult,
    ) -> c_int;
    #[link_name = "talu_collab_get_history"]
    fn talu_collab_get_history_raw(
        handle: *mut c_void,
        after_key: *const c_char,
        limit: usize,
        out_history: *mut CCollabHistoryList,
    ) -> c_int;
    #[link_name = "talu_collab_clear_snapshot"]
    fn talu_collab_clear_snapshot_raw(
        handle: *mut c_void,
        actor_id: *const c_char,
        actor_kind: u8,
        role: *const c_char,
        op_kind: *const c_char,
        out_result: *mut CCollabOpResult,
    ) -> c_int;
    #[link_name = "talu_collab_put_presence"]
    fn talu_collab_put_presence_raw(
        handle: *mut c_void,
        participant_id: *const c_char,
        payload: *const u8,
        payload_len: usize,
        ttl_ms: u64,
        out_ttl_ms: *mut u64,
    ) -> c_int;
    #[link_name = "talu_collab_get_presence"]
    fn talu_collab_get_presence_raw(
        handle: *mut c_void,
        participant_id: *const c_char,
        out_value: *mut CCollabValue,
    ) -> c_int;
    #[link_name = "talu_collab_watch_drain"]
    fn talu_collab_watch_drain_raw(
        handle: *mut c_void,
        after_seq: u64,
        max_events: usize,
        out_batch: *mut CCollabWatchBatch,
    ) -> c_int;
    #[link_name = "talu_collab_watch_wait"]
    fn talu_collab_watch_wait_raw(
        handle: *mut c_void,
        after_seq: u64,
        timeout_ms: u64,
        out_result: *mut CCollabWatchWaitResult,
    ) -> c_int;
    #[link_name = "talu_collab_free_session"]
    fn talu_collab_free_session_raw(session: *mut CCollabSession);
    #[link_name = "talu_collab_free_summary"]
    fn talu_collab_free_summary_raw(summary: *mut CCollabSummary);
    #[link_name = "talu_collab_free_value"]
    fn talu_collab_free_value_raw(value: *mut CCollabValue);
    #[link_name = "talu_collab_free_op_result"]
    fn talu_collab_free_op_result_raw(result: *mut CCollabOpResult);
    #[link_name = "talu_collab_free_history"]
    fn talu_collab_free_history_raw(history: *mut CCollabHistoryList);
    #[link_name = "talu_collab_free_watch_batch"]
    fn talu_collab_free_watch_batch_raw(batch: *mut CCollabWatchBatch);
}

#[derive(Debug, Clone)]
pub enum CollabError {
    InvalidArgument(String),
    Busy(String),
    ResourceExhausted(String),
    Storage(String),
}

impl CollabError {
    fn from_code(code: i32, fallback: &str) -> Self {
        let detail = crate::error::last_error_message().unwrap_or_else(|| fallback.to_string());
        match code {
            ERROR_CODE_INVALID_ARGUMENT => Self::InvalidArgument(detail),
            ERROR_CODE_RESOURCE_BUSY => Self::Busy(detail),
            ERROR_CODE_RESOURCE_EXHAUSTED => Self::ResourceExhausted(detail),
            _ => Self::Storage(detail),
        }
    }
}

impl std::fmt::Display for CollabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            Self::Busy(msg) => write!(f, "Resource busy: {msg}"),
            Self::ResourceExhausted(msg) => write!(f, "Resource exhausted: {msg}"),
            Self::Storage(msg) => write!(f, "Storage error: {msg}"),
        }
    }
}

impl std::error::Error for CollabError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ParticipantKind {
    Human = 0,
    Agent = 1,
    External = 2,
    System = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchEventType {
    Put,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchDurability {
    Strong,
    Batched,
    Ephemeral,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionInfo {
    pub namespace: String,
    pub participant_id: String,
    pub participant_kind: ParticipantKind,
    pub status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceSummary {
    pub namespace: String,
    pub meta_json: Option<String>,
    pub total_live_entries: usize,
    pub batched_pending: usize,
    pub ephemeral_live_entries: usize,
    pub watch_published: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryValue {
    pub data: Vec<u8>,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpSubmitResult {
    pub op_key: String,
    pub accepted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryEntry {
    pub actor_id: String,
    pub actor_seq: u64,
    pub op_id: String,
    pub payload: Vec<u8>,
    pub updated_at_ms: i64,
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchEvent {
    pub seq: u64,
    pub event_type: WatchEventType,
    pub key: String,
    pub value_len: usize,
    pub durability: Option<WatchDurability>,
    pub ttl_ms: Option<u64>,
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchBatch {
    pub events: Vec<WatchEvent>,
    pub lost: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchWaitResult {
    pub published_seq: u64,
    pub timed_out: bool,
}

#[derive(Debug)]
pub struct CollabHandle {
    handle: *mut c_void,
}

unsafe impl Send for CollabHandle {}
unsafe impl Sync for CollabHandle {}

impl CollabHandle {
    pub fn open(
        db_root: &str,
        resource_kind: &str,
        resource_id: &str,
    ) -> Result<Self, CollabError> {
        let db_root_c = to_cstring(db_root, "db_root")?;
        let kind_c = to_cstring(resource_kind, "resource_kind")?;
        let id_c = to_cstring(resource_id, "resource_id")?;

        let mut handle = std::ptr::null_mut();
        let rc = unsafe {
            talu_collab_init_raw(
                db_root_c.as_ptr(),
                kind_c.as_ptr(),
                id_c.as_ptr(),
                &mut handle,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(
                rc,
                "failed to initialize collab handle",
            ));
        }
        if handle.is_null() {
            return Err(CollabError::Storage(
                "collab init succeeded but returned null handle".to_string(),
            ));
        }
        Ok(Self { handle })
    }

    pub fn open_session(
        &self,
        participant_id: &str,
        participant_kind: ParticipantKind,
        role: Option<&str>,
    ) -> Result<SessionInfo, CollabError> {
        let participant_c = to_cstring(participant_id, "participant_id")?;
        let role_c = to_optional_cstring(role, "role")?;
        let mut out = CCollabSession::default();
        let rc = unsafe {
            talu_collab_open_session_raw(
                self.handle,
                participant_c.as_ptr(),
                participant_kind as u8,
                opt_ptr(&role_c),
                &mut out,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(rc, "failed to open collab session"));
        }
        let info = SessionInfo {
            namespace: cstr_to_string(out.namespace, "namespace")?,
            participant_id: cstr_to_string(out.participant_id, "participant_id")?,
            participant_kind: participant_kind_from_raw(out.participant_kind)?,
            status: cstr_to_string(out.status, "status")?,
        };
        unsafe { talu_collab_free_session_raw(&mut out) };
        Ok(info)
    }

    pub fn summary(&self) -> Result<ResourceSummary, CollabError> {
        let mut out = CCollabSummary::default();
        let rc = unsafe { talu_collab_get_summary_raw(self.handle, &mut out) };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(rc, "failed to read collab summary"));
        }
        let summary = ResourceSummary {
            namespace: cstr_to_string(out.namespace, "namespace")?,
            meta_json: optional_cstr_to_string(out.meta_json),
            total_live_entries: out.total_live_entries,
            batched_pending: out.batched_pending,
            ephemeral_live_entries: out.ephemeral_live_entries,
            watch_published: out.watch_published,
        };
        unsafe { talu_collab_free_summary_raw(&mut out) };
        Ok(summary)
    }

    pub fn snapshot(&self) -> Result<Option<BinaryValue>, CollabError> {
        let mut out = CCollabValue::default();
        let rc = unsafe { talu_collab_get_snapshot_raw(self.handle, &mut out) };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(rc, "failed to read collab snapshot"));
        }
        let value = collab_value_to_option(&mut out);
        unsafe { talu_collab_free_value_raw(&mut out) };
        value
    }

    pub fn submit_op(
        &self,
        actor_id: &str,
        actor_seq: u64,
        op_id: &str,
        payload: &[u8],
        issued_at_ms: Option<i64>,
        snapshot: Option<&[u8]>,
    ) -> Result<OpSubmitResult, CollabError> {
        let actor_c = to_cstring(actor_id, "actor_id")?;
        let op_c = to_cstring(op_id, "op_id")?;
        let payload_ptr = if payload.is_empty() {
            std::ptr::null()
        } else {
            payload.as_ptr()
        };
        let snapshot_ptr = snapshot
            .and_then(|bytes| {
                if bytes.is_empty() {
                    None
                } else {
                    Some(bytes.as_ptr())
                }
            })
            .unwrap_or(std::ptr::null());
        let snapshot_len = snapshot.map_or(0, |bytes| bytes.len());

        let mut out = CCollabOpResult::default();
        let rc = unsafe {
            talu_collab_submit_op_raw(
                self.handle,
                actor_c.as_ptr(),
                actor_seq,
                op_c.as_ptr(),
                payload_ptr,
                payload.len(),
                issued_at_ms.unwrap_or_default(),
                issued_at_ms.is_some(),
                snapshot_ptr,
                snapshot_len,
                snapshot.is_some(),
                &mut out,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(rc, "failed to submit collab op"));
        }
        let result = OpSubmitResult {
            op_key: cstr_to_string(out.op_key, "op_key")?,
            accepted: out.accepted,
        };
        unsafe { talu_collab_free_op_result_raw(&mut out) };
        Ok(result)
    }


    pub fn history(
        &self,
        after_key: Option<&str>,
        limit: usize,
    ) -> Result<Vec<HistoryEntry>, CollabError> {
        let after_key_c = to_optional_cstring(after_key, "after_key")?;
        let mut out = CCollabHistoryList::default();
        let rc = unsafe {
            talu_collab_get_history_raw(self.handle, opt_ptr(&after_key_c), limit, &mut out)
        };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(rc, "failed to read collab history"));
        }

        let mut entries = Vec::with_capacity(out.count);
        if !out.items.is_null() && out.count > 0 {
            let items = unsafe { std::slice::from_raw_parts(out.items, out.count) };
            for item in items {
                let payload = if item.payload.is_null() || item.payload_len == 0 {
                    Vec::new()
                } else {
                    unsafe { std::slice::from_raw_parts(item.payload, item.payload_len) }.to_vec()
                };
                entries.push(HistoryEntry {
                    actor_id: cstr_to_string(item.actor_id, "actor_id")?,
                    actor_seq: item.actor_seq,
                    op_id: cstr_to_string(item.op_id, "op_id")?,
                    payload,
                    updated_at_ms: item.updated_at_ms,
                    key: cstr_to_string(item.key, "key")?,
                });
            }
        }
        unsafe { talu_collab_free_history_raw(&mut out) };
        Ok(entries)
    }

    pub fn clear_snapshot(
        &self,
        actor_id: &str,
        actor_kind: ParticipantKind,
        role: Option<&str>,
        op_kind: &str,
    ) -> Result<OpSubmitResult, CollabError> {
        let actor_c = to_cstring(actor_id, "actor_id")?;
        let role_c = to_optional_cstring(role, "role")?;
        let op_kind_c = to_cstring(op_kind, "op_kind")?;
        let mut out = CCollabOpResult::default();
        let rc = unsafe {
            talu_collab_clear_snapshot_raw(
                self.handle,
                actor_c.as_ptr(),
                actor_kind as u8,
                opt_ptr(&role_c),
                op_kind_c.as_ptr(),
                &mut out,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(rc, "failed to clear collab snapshot"));
        }
        let result = OpSubmitResult {
            op_key: cstr_to_string(out.op_key, "op_key")?,
            accepted: out.accepted,
        };
        unsafe { talu_collab_free_op_result_raw(&mut out) };
        Ok(result)
    }

    pub fn put_presence(
        &self,
        participant_id: &str,
        payload: &[u8],
        ttl_ms: u64,
    ) -> Result<u64, CollabError> {
        let participant_c = to_cstring(participant_id, "participant_id")?;
        let payload_ptr = if payload.is_empty() {
            std::ptr::null()
        } else {
            payload.as_ptr()
        };
        let mut out_ttl = 0_u64;
        let rc = unsafe {
            talu_collab_put_presence_raw(
                self.handle,
                participant_c.as_ptr(),
                payload_ptr,
                payload.len(),
                ttl_ms,
                &mut out_ttl,
            )
        };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(rc, "failed to put collab presence"));
        }
        Ok(out_ttl)
    }

    pub fn presence(&self, participant_id: &str) -> Result<Option<BinaryValue>, CollabError> {
        let participant_c = to_cstring(participant_id, "participant_id")?;
        let mut out = CCollabValue::default();
        let rc =
            unsafe { talu_collab_get_presence_raw(self.handle, participant_c.as_ptr(), &mut out) };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(rc, "failed to read collab presence"));
        }
        let value = collab_value_to_option(&mut out);
        unsafe { talu_collab_free_value_raw(&mut out) };
        value
    }

    pub fn watch_drain(
        &self,
        after_seq: u64,
        max_events: usize,
    ) -> Result<WatchBatch, CollabError> {
        let mut out = CCollabWatchBatch::default();
        let rc =
            unsafe { talu_collab_watch_drain_raw(self.handle, after_seq, max_events, &mut out) };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(
                rc,
                "failed to drain collab watch events",
            ));
        }

        let mut events = Vec::with_capacity(out.count);
        if !out.items.is_null() && out.count > 0 {
            let items = unsafe { std::slice::from_raw_parts(out.items, out.count) };
            for item in items {
                events.push(WatchEvent {
                    seq: item.seq,
                    event_type: match item.event_type {
                        0 => WatchEventType::Put,
                        1 => WatchEventType::Delete,
                        _ => {
                            unsafe { talu_collab_free_watch_batch_raw(&mut out) };
                            return Err(CollabError::Storage(
                                "invalid collab watch event type".to_string(),
                            ));
                        }
                    },
                    key: cstr_to_string(item.key, "key")?,
                    value_len: item.value_len,
                    durability: if item.has_durability {
                        Some(match item.durability_class {
                            0 => WatchDurability::Strong,
                            1 => WatchDurability::Batched,
                            2 => WatchDurability::Ephemeral,
                            _ => {
                                unsafe { talu_collab_free_watch_batch_raw(&mut out) };
                                return Err(CollabError::Storage(
                                    "invalid collab watch durability".to_string(),
                                ));
                            }
                        })
                    } else {
                        None
                    },
                    ttl_ms: if item.has_ttl {
                        Some(item.ttl_ms)
                    } else {
                        None
                    },
                    updated_at_ms: item.updated_at_ms,
                });
            }
        }
        let batch = WatchBatch {
            events,
            lost: out.lost,
        };
        unsafe { talu_collab_free_watch_batch_raw(&mut out) };
        Ok(batch)
    }

    pub fn watch_wait(
        &self,
        after_seq: u64,
        timeout_ms: u64,
    ) -> Result<WatchWaitResult, CollabError> {
        let mut out = CCollabWatchWaitResult::default();
        let rc =
            unsafe { talu_collab_watch_wait_raw(self.handle, after_seq, timeout_ms, &mut out) };
        if rc != ERROR_CODE_OK {
            return Err(CollabError::from_code(
                rc,
                "failed to wait for collab watch event",
            ));
        }
        Ok(WatchWaitResult {
            published_seq: out.published_seq,
            timed_out: out.timed_out,
        })
    }
}

impl Drop for CollabHandle {
    fn drop(&mut self) {
        unsafe { talu_collab_free_raw(self.handle) };
    }
}

fn collab_value_to_option(out: &mut CCollabValue) -> Result<Option<BinaryValue>, CollabError> {
    if !out.found {
        return Ok(None);
    }
    if out.data.is_null() && out.len > 0 {
        return Err(CollabError::Storage(
            "collab value returned invalid value buffer".to_string(),
        ));
    }
    let data = if out.len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(out.data, out.len) }.to_vec()
    };
    Ok(Some(BinaryValue {
        data,
        updated_at_ms: out.updated_at_ms,
    }))
}

fn participant_kind_from_raw(raw: u8) -> Result<ParticipantKind, CollabError> {
    match raw {
        0 => Ok(ParticipantKind::Human),
        1 => Ok(ParticipantKind::Agent),
        2 => Ok(ParticipantKind::External),
        3 => Ok(ParticipantKind::System),
        _ => Err(CollabError::Storage(
            "collab returned invalid participant kind".to_string(),
        )),
    }
}

fn cstr_to_string(ptr: *const c_char, field: &str) -> Result<String, CollabError> {
    if ptr.is_null() {
        return Err(CollabError::Storage(format!(
            "collab output missing required field {field}"
        )));
    }
    Ok(unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned())
}

fn optional_cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

fn to_cstring(value: &str, field: &str) -> Result<CString, CollabError> {
    CString::new(value)
        .map_err(|_| CollabError::InvalidArgument(format!("{field} contains null byte")))
}

fn to_optional_cstring(value: Option<&str>, field: &str) -> Result<Option<CString>, CollabError> {
    value.map(|raw| to_cstring(raw, field)).transpose()
}

fn opt_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(std::ptr::null(), |v| v.as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collab_presence_roundtrip_same_handle() {
        let temp = tempfile::TempDir::new().expect("temp dir");
        let db_root = temp.path().join("kv");
        std::fs::create_dir_all(&db_root).expect("create db root");

        let handle = CollabHandle::open(
            db_root.to_str().expect("utf8 path"),
            "file_buffer",
            "main.zig",
        )
        .expect("open collab");

        let ttl_ms = handle
            .put_presence("human:1", br#"{"cursor":42}"#, 0)
            .expect("put presence");
        assert!(ttl_ms > 0);

        let value = handle
            .presence("human:1")
            .expect("get presence")
            .expect("presence exists");
        assert_eq!(value.data, br#"{"cursor":42}"#);
    }

    #[test]
    fn collab_clear_snapshot_removes_snapshot() {
        let temp = tempfile::TempDir::new().expect("temp dir");
        let db_root = temp.path().join("kv");
        std::fs::create_dir_all(&db_root).expect("create db root");

        let handle = CollabHandle::open(
            db_root.to_str().expect("utf8 path"),
            "workdir_file",
            "notes.txt",
        )
        .expect("open collab");

        handle
            .submit_op(
                "system:agent_fs",
                1,
                "op-1",
                br#"{"type":"fs_write"}"#,
                Some(1),
                Some(b"hello"),
            )
            .expect("submit initial snapshot");
        handle
            .clear_snapshot("system:agent_fs", ParticipantKind::System, Some("sync"), "fs_delete")
            .expect("clear snapshot");

        let snapshot = handle.snapshot().expect("read snapshot");
        assert!(snapshot.is_none());
    }

}
