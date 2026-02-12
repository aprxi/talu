#[cfg(test)]
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::path::Path;
use std::ptr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};

#[repr(C)]
struct sqlite3 {
    _private: [u8; 0],
}

#[repr(C)]
struct sqlite3_stmt {
    _private: [u8; 0],
}

#[link(name = "sqlite3")]
unsafe extern "C" {
    fn sqlite3_open_v2(
        filename: *const c_char,
        pp_db: *mut *mut sqlite3,
        flags: c_int,
        z_vfs: *const c_char,
    ) -> c_int;
    fn sqlite3_close(db: *mut sqlite3) -> c_int;
    fn sqlite3_exec(
        db: *mut sqlite3,
        sql: *const c_char,
        callback: Option<
            extern "C" fn(*mut c_void, c_int, *mut *mut c_char, *mut *mut c_char) -> c_int,
        >,
        data: *mut c_void,
        errmsg: *mut *mut c_char,
    ) -> c_int;
    fn sqlite3_prepare_v2(
        db: *mut sqlite3,
        sql: *const c_char,
        n_byte: c_int,
        pp_stmt: *mut *mut sqlite3_stmt,
        pz_tail: *mut *const c_char,
    ) -> c_int;
    fn sqlite3_finalize(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_step(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_bind_text(
        stmt: *mut sqlite3_stmt,
        idx: c_int,
        value: *const c_char,
        n: c_int,
        destructor: Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> c_int;
    fn sqlite3_bind_int64(stmt: *mut sqlite3_stmt, idx: c_int, value: i64) -> c_int;
    fn sqlite3_column_int64(stmt: *mut sqlite3_stmt, col: c_int) -> i64;
    fn sqlite3_column_type(stmt: *mut sqlite3_stmt, col: c_int) -> c_int;
    fn sqlite3_column_text(stmt: *mut sqlite3_stmt, col: c_int) -> *const c_char;
    fn sqlite3_changes(db: *mut sqlite3) -> c_int;
    fn sqlite3_errmsg(db: *mut sqlite3) -> *const c_char;
}

const SQLITE_OK: c_int = 0;
const SQLITE_ROW: c_int = 100;
const SQLITE_DONE: c_int = 101;
const SQLITE_NULL: c_int = 5;

const SQLITE_OPEN_READWRITE: c_int = 0x0000_0002;
const SQLITE_OPEN_CREATE: c_int = 0x0000_0004;
const SQLITE_OPEN_FULLMUTEX: c_int = 0x0001_0000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PinnedModelEntry {
    pub model_uri: String,
    pub pinned_at_ms: i64,
    pub size_bytes: Option<u64>,
    pub size_updated_at_ms: Option<i64>,
}

/// Profile-local pin storage backed by SQLite (`<bucket>/meta.sqlite`).
pub(crate) struct PinStore {
    db: *mut sqlite3,
}

impl Drop for PinStore {
    fn drop(&mut self) {
        if self.db.is_null() {
            return;
        }
        // SAFETY: `self.db` is an owned sqlite handle created by `sqlite3_open_v2`.
        unsafe {
            let _ = sqlite3_close(self.db);
        }
        self.db = ptr::null_mut();
    }
}

impl PinStore {
    pub(crate) fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }

        let c_path = CString::new(path.to_string_lossy().as_ref())?;
        let mut db: *mut sqlite3 = ptr::null_mut();

        // SAFETY: pointers are valid and out-param `db` is writable.
        let rc = unsafe {
            sqlite3_open_v2(
                c_path.as_ptr(),
                &mut db,
                SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                ptr::null(),
            )
        };

        if rc != SQLITE_OK || db.is_null() {
            let msg = sqlite_error_message(db);
            if !db.is_null() {
                // SAFETY: `db` is a live sqlite handle on open failure.
                unsafe {
                    let _ = sqlite3_close(db);
                }
            }
            return Err(anyhow!("Failed to open SQLite metadata DB: {}", msg));
        }

        let store = Self { db };
        store.exec_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = FULL;
            PRAGMA busy_timeout = 5000;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS pinned_models (
                model_uri TEXT PRIMARY KEY,
                pinned_at_ms INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pinned_models_pinned_at
            ON pinned_models(pinned_at_ms DESC);

            CREATE TABLE IF NOT EXISTS pinned_model_sizes (
                model_uri TEXT PRIMARY KEY REFERENCES pinned_models(model_uri) ON DELETE CASCADE,
                size_bytes INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            ",
        )?;

        Ok(store)
    }

    pub(crate) fn pin(&self, model_uri: &str) -> Result<bool> {
        let mut stmt = self.prepare(
            "INSERT OR IGNORE INTO pinned_models(model_uri, pinned_at_ms) VALUES (?1, ?2)",
        )?;
        stmt.bind_text(1, model_uri)?;
        stmt.bind_i64(2, now_unix_ms())?;
        stmt.step_done()?;
        Ok(self.changes() > 0)
    }

    pub(crate) fn unpin(&self, model_uri: &str) -> Result<bool> {
        let mut stmt = self.prepare("DELETE FROM pinned_models WHERE model_uri = ?1")?;
        stmt.bind_text(1, model_uri)?;
        stmt.step_done()?;
        Ok(self.changes() > 0)
    }

    pub(crate) fn upsert_size_bytes(&self, model_uri: &str, size_bytes: u64) -> Result<()> {
        let size_i64 = i64::try_from(size_bytes).map_err(|_| anyhow!("size exceeds i64 range"))?;
        let mut stmt = self.prepare(
            "
            INSERT INTO pinned_model_sizes(model_uri, size_bytes, updated_at_ms)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(model_uri) DO UPDATE SET
                size_bytes = excluded.size_bytes,
                updated_at_ms = excluded.updated_at_ms
            ",
        )?;
        stmt.bind_text(1, model_uri)?;
        stmt.bind_i64(2, size_i64)?;
        stmt.bind_i64(3, now_unix_ms())?;
        stmt.step_done()?;
        Ok(())
    }

    pub(crate) fn clear_size_bytes(&self, model_uri: &str) -> Result<()> {
        let mut stmt = self.prepare("DELETE FROM pinned_model_sizes WHERE model_uri = ?1")?;
        stmt.bind_text(1, model_uri)?;
        stmt.step_done()?;
        Ok(())
    }

    pub(crate) fn list_pinned_entries(&self) -> Result<Vec<PinnedModelEntry>> {
        let mut stmt = self.prepare(
            "
            SELECT p.model_uri, p.pinned_at_ms, s.size_bytes, s.updated_at_ms
            FROM pinned_models AS p
            LEFT JOIN pinned_model_sizes AS s ON s.model_uri = p.model_uri
            ORDER BY p.pinned_at_ms DESC, p.model_uri ASC
            ",
        )?;

        let mut out = Vec::new();
        loop {
            match stmt.step()? {
                SQLITE_ROW => {
                    let size_bytes = stmt.column_i64_opt(2)?.and_then(|v| u64::try_from(v).ok());
                    out.push(PinnedModelEntry {
                        model_uri: stmt.column_text(0)?,
                        pinned_at_ms: stmt.column_i64(1)?,
                        size_bytes,
                        size_updated_at_ms: stmt.column_i64_opt(3)?,
                    });
                }
                SQLITE_DONE => break,
                code => return Err(anyhow!("sqlite3_step failed with code {}", code)),
            }
        }
        Ok(out)
    }

    pub(crate) fn list_pinned(&self) -> Result<Vec<String>> {
        Ok(self
            .list_pinned_entries()?
            .into_iter()
            .map(|entry| entry.model_uri)
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn list_pinned_set(&self) -> Result<HashSet<String>> {
        Ok(self.list_pinned()?.into_iter().collect())
    }

    fn exec_batch(&self, sql: &str) -> Result<()> {
        let c_sql = CString::new(sql)?;
        // SAFETY: `self.db` is valid and `c_sql` is a null-terminated SQL string.
        let rc = unsafe {
            sqlite3_exec(
                self.db,
                c_sql.as_ptr(),
                None,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        if rc != SQLITE_OK {
            return Err(anyhow!("SQLite exec failed: {}", self.error_message()));
        }
        Ok(())
    }

    fn prepare(&self, sql: &str) -> Result<Statement<'_>> {
        let c_sql = CString::new(sql)?;
        let mut stmt: *mut sqlite3_stmt = ptr::null_mut();
        // SAFETY: `self.db` is valid; out-param `stmt` is writable.
        let rc =
            unsafe { sqlite3_prepare_v2(self.db, c_sql.as_ptr(), -1, &mut stmt, ptr::null_mut()) };
        if rc != SQLITE_OK || stmt.is_null() {
            return Err(anyhow!("Failed to prepare SQL: {}", self.error_message()));
        }
        Ok(Statement {
            store: self,
            stmt,
            bound_text: Vec::new(),
        })
    }

    fn changes(&self) -> i32 {
        // SAFETY: `self.db` is a valid sqlite handle.
        unsafe { sqlite3_changes(self.db) }
    }

    fn error_message(&self) -> String {
        sqlite_error_message(self.db)
    }
}

struct Statement<'a> {
    store: &'a PinStore,
    stmt: *mut sqlite3_stmt,
    // Keep bound strings alive until statement is stepped/finalized.
    bound_text: Vec<CString>,
}

impl Drop for Statement<'_> {
    fn drop(&mut self) {
        if self.stmt.is_null() {
            return;
        }
        // SAFETY: `self.stmt` is an owned prepared statement.
        unsafe {
            let _ = sqlite3_finalize(self.stmt);
        }
        self.stmt = ptr::null_mut();
    }
}

impl Statement<'_> {
    fn bind_text(&mut self, idx: i32, value: &str) -> Result<()> {
        let c_value = CString::new(value)?;
        // SAFETY: statement is valid and pointer stays alive via `bound_text`.
        let rc = unsafe { sqlite3_bind_text(self.stmt, idx, c_value.as_ptr(), -1, None) };
        if rc != SQLITE_OK {
            return Err(anyhow!(
                "Failed to bind text parameter: {}",
                self.store.error_message()
            ));
        }
        self.bound_text.push(c_value);
        Ok(())
    }

    fn bind_i64(&mut self, idx: i32, value: i64) -> Result<()> {
        // SAFETY: statement is valid and accepts integer binding.
        let rc = unsafe { sqlite3_bind_int64(self.stmt, idx, value) };
        if rc != SQLITE_OK {
            return Err(anyhow!(
                "Failed to bind integer parameter: {}",
                self.store.error_message()
            ));
        }
        Ok(())
    }

    fn step(&mut self) -> Result<i32> {
        // SAFETY: statement is valid and can be stepped.
        let rc = unsafe { sqlite3_step(self.stmt) };
        if rc == SQLITE_ROW || rc == SQLITE_DONE {
            Ok(rc)
        } else {
            Err(anyhow!(
                "sqlite3_step failed: {}",
                self.store.error_message()
            ))
        }
    }

    fn step_done(&mut self) -> Result<()> {
        match self.step()? {
            SQLITE_DONE => Ok(()),
            SQLITE_ROW => Err(anyhow!("Unexpected row returned for write statement")),
            _ => Err(anyhow!("Unexpected sqlite step result")),
        }
    }

    fn column_text(&self, idx: i32) -> Result<String> {
        // SAFETY: statement is positioned on a row; SQLite returns pointer valid until next step/finalize.
        let ptr = unsafe { sqlite3_column_text(self.stmt, idx) };
        if ptr.is_null() {
            return Ok(String::new());
        }
        // SAFETY: `ptr` points to a null-terminated UTF-8-ish string owned by SQLite.
        let text = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();
        Ok(text)
    }

    fn column_i64(&self, idx: i32) -> Result<i64> {
        // SAFETY: statement is positioned on a row and column is valid.
        Ok(unsafe { sqlite3_column_int64(self.stmt, idx) })
    }

    fn column_i64_opt(&self, idx: i32) -> Result<Option<i64>> {
        // SAFETY: statement is positioned on a row and column is valid.
        let col_type = unsafe { sqlite3_column_type(self.stmt, idx) };
        if col_type == SQLITE_NULL {
            return Ok(None);
        }
        Ok(Some(self.column_i64(idx)?))
    }
}

fn sqlite_error_message(db: *mut sqlite3) -> String {
    if db.is_null() {
        return "unknown sqlite error".to_string();
    }
    // SAFETY: `db` is a valid sqlite handle and returns a static error string pointer.
    let msg = unsafe { sqlite3_errmsg(db) };
    if msg.is_null() {
        "unknown sqlite error".to_string()
    } else {
        // SAFETY: `msg` is a null-terminated C string managed by SQLite.
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    }
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_unpin_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("meta.sqlite");
        let store = PinStore::open(&db).expect("open");

        assert!(store.pin("Qwen/Qwen3-0.6B").expect("pin"));
        assert!(!store.pin("Qwen/Qwen3-0.6B").expect("idempotent pin"));

        let set = store.list_pinned_set().expect("list set");
        assert!(set.contains("Qwen/Qwen3-0.6B"));

        assert!(store.unpin("Qwen/Qwen3-0.6B").expect("unpin"));
        assert!(!store.unpin("Qwen/Qwen3-0.6B").expect("idempotent unpin"));

        let set = store.list_pinned_set().expect("list set after");
        assert!(!set.contains("Qwen/Qwen3-0.6B"));
    }

    #[test]
    fn pin_size_roundtrip_and_cascade() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("meta.sqlite");
        let store = PinStore::open(&db).expect("open");

        assert!(store.pin("Qwen/Qwen3-0.6B").expect("pin"));
        store
            .upsert_size_bytes("Qwen/Qwen3-0.6B", 1_234_567_890)
            .expect("upsert size");

        let entries = store.list_pinned_entries().expect("list entries");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].model_uri, "Qwen/Qwen3-0.6B");
        assert_eq!(entries[0].size_bytes, Some(1_234_567_890));
        assert!(entries[0].size_updated_at_ms.is_some());

        assert!(store.unpin("Qwen/Qwen3-0.6B").expect("unpin"));

        assert!(store.pin("Qwen/Qwen3-0.6B").expect("re-pin"));
        let entries = store.list_pinned_entries().expect("list entries");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].size_bytes, None);
        assert_eq!(entries[0].size_updated_at_ms, None);
    }
}
