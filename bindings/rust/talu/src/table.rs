//! Table engine — safe Rust bindings over `talu_db_table_*` C API.
//!
//! Provides a domain-agnostic table with scan, get, append, and delete
//! operations. Payloads are opaque bytes; the caller encodes/decodes them.

use std::os::raw::c_void;
use std::path::Path;
use std::ptr;

const ERROR_CODE_OK: i32 = 0;

/// Error type for table operations.
#[derive(Debug)]
pub enum TableError {
    InvalidArgument(String),
    StorageError(String),
    ReadOnly,
}

impl std::fmt::Display for TableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableError::InvalidArgument(s) => write!(f, "Invalid argument: {s}"),
            TableError::StorageError(s) => write!(f, "Storage error: {s}"),
            TableError::ReadOnly => write!(f, "Table is read-only"),
        }
    }
}

impl std::error::Error for TableError {}

impl TableError {
    fn from_code(code: i32) -> Self {
        let detail = crate::error::last_error_message().unwrap_or_default();
        match code {
            901 => TableError::InvalidArgument(detail),
            _ => TableError::StorageError(detail),
        }
    }
}

/// Column shape constants matching `types.ColumnShape`.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum ColumnShape {
    Scalar = 1,
    Vector = 2,
    VarBytes = 3,
}

/// Physical type constants matching `types.PhysicalType`.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum PhysicalType {
    U64 = 3,
    I64 = 7,
    F64 = 11,
    Binary = 20,
}

/// Filter operation for scan queries.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum FilterOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

/// A column value for write operations.
pub struct ColumnValue {
    pub column_id: u32,
    pub shape: ColumnShape,
    pub phys_type: PhysicalType,
    pub dims: u16,
    pub data: Vec<u8>,
}

/// A scalar column filter for scan operations.
pub struct ColumnFilter {
    pub column_id: u32,
    pub op: FilterOp,
    pub value: u64,
}

/// Compaction policy for opening a table.
pub struct CompactionPolicy {
    pub active_schema_ids: Vec<u16>,
    pub tombstone_schema_id: Option<u16>,
    pub dedup_column_id: u32,
    pub ts_column_id: u32,
    pub ttl_column_id: Option<u32>,
}

/// Parameters for a scan operation.
pub struct ScanParams {
    pub schema_id: u16,
    pub additional_schema_ids: Vec<u16>,
    pub filters: Vec<ColumnFilter>,
    pub extra_columns: Vec<u32>,
    pub dedup_column_id: u32,
    pub delete_schema_id: Option<u16>,
    pub ts_column_id: u32,
    pub ttl_column_id: Option<u32>,
    pub limit: u32,
    pub cursor_ts: Option<i64>,
    pub cursor_hash: Option<u64>,
    pub payload_column_id: u32,
    pub reverse: bool,
}

impl Default for ScanParams {
    fn default() -> Self {
        Self {
            schema_id: 0,
            additional_schema_ids: Vec::new(),
            filters: Vec::new(),
            extra_columns: Vec::new(),
            dedup_column_id: 1,
            delete_schema_id: None,
            ts_column_id: 2,
            ttl_column_id: None,
            limit: 0,
            cursor_ts: None,
            cursor_hash: None,
            payload_column_id: 20,
            reverse: true,
        }
    }
}

/// A scalar column value from a scanned row.
#[derive(Debug, Clone)]
pub struct ScalarColumn {
    pub column_id: u32,
    pub value: u64,
}

/// A row returned by scan or get.
#[derive(Debug, Clone)]
pub struct Row {
    pub scalars: Vec<ScalarColumn>,
    pub payload: Vec<u8>,
}

/// Result of a scan operation.
#[derive(Debug)]
pub struct ScanResult {
    pub rows: Vec<Row>,
    pub has_more: bool,
}

/// Handle to a table (wraps opaque C pointer).
pub struct TableHandle {
    raw: *mut c_void,
}

// Safety: The underlying C handle is thread-safe (uses granular locking).
unsafe impl Send for TableHandle {}

impl TableHandle {
    /// Open a read-write table.
    pub fn open(
        db_root: &Path,
        namespace: &str,
        policy: &CompactionPolicy,
    ) -> Result<Self, TableError> {
        let root_str = db_root.to_string_lossy();
        let root_bytes = root_str.as_bytes();
        let ns_bytes = namespace.as_bytes();

        let c_policy = talu_sys::CCompactionPolicy {
            active_schema_ids: if policy.active_schema_ids.is_empty() {
                ptr::null()
            } else {
                policy.active_schema_ids.as_ptr()
            },
            active_schema_count: policy.active_schema_ids.len() as u32,
            tombstone_schema_id: policy.tombstone_schema_id.unwrap_or(0),
            dedup_column_id: policy.dedup_column_id,
            ts_column_id: policy.ts_column_id,
            ttl_column_id: policy.ttl_column_id.unwrap_or(0),
            _pad: [0; 2],
        };

        let mut handle: *mut c_void = ptr::null_mut();
        let code = unsafe {
            talu_sys::talu_db_table_open(
                root_bytes.as_ptr(),
                root_bytes.len(),
                ns_bytes.as_ptr(),
                ns_bytes.len(),
                &c_policy,
                &mut handle,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(TableError::from_code(code));
        }
        Ok(Self { raw: handle })
    }

    /// Open a read-only table.
    pub fn open_readonly(db_root: &Path, namespace: &str) -> Result<Self, TableError> {
        let root_str = db_root.to_string_lossy();
        let root_bytes = root_str.as_bytes();
        let ns_bytes = namespace.as_bytes();

        let mut handle: *mut c_void = ptr::null_mut();
        let code = unsafe {
            talu_sys::talu_db_table_open_readonly(
                root_bytes.as_ptr(),
                root_bytes.len(),
                ns_bytes.as_ptr(),
                ns_bytes.len(),
                &mut handle,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(TableError::from_code(code));
        }
        Ok(Self { raw: handle })
    }

    /// Append a row with the given schema ID and column values.
    pub fn append_row(&self, schema_id: u16, columns: &[ColumnValue]) -> Result<(), TableError> {
        let c_cols: Vec<talu_sys::CColumnValue> = columns
            .iter()
            .map(|c| talu_sys::CColumnValue {
                column_id: c.column_id,
                shape: c.shape as u8,
                phys_type: c.phys_type as u8,
                dims: c.dims,
                data: c.data.as_ptr(),
                data_len: c.data.len(),
            })
            .collect();

        let code = unsafe {
            talu_sys::talu_db_table_append_row(
                self.raw,
                schema_id,
                c_cols.as_ptr(),
                c_cols.len() as u32,
            )
        };

        if code != ERROR_CODE_OK {
            return Err(TableError::from_code(code));
        }
        Ok(())
    }

    /// Write a tombstone for the given primary key hash.
    pub fn delete_row(&self, pk_hash: u64, ts: i64) -> Result<(), TableError> {
        let code = unsafe { talu_sys::talu_db_table_delete_row(self.raw, pk_hash, ts) };
        if code != ERROR_CODE_OK {
            return Err(TableError::from_code(code));
        }
        Ok(())
    }

    /// Flush buffered rows to disk.
    pub fn flush(&self) -> Result<(), TableError> {
        let code = unsafe { talu_sys::talu_db_table_flush(self.raw) };
        if code != ERROR_CODE_OK {
            return Err(TableError::from_code(code));
        }
        Ok(())
    }

    /// Scan the table with the given parameters.
    pub fn scan(&self, params: &ScanParams) -> Result<ScanResult, TableError> {
        let c_filters: Vec<talu_sys::CColumnFilter> = params
            .filters
            .iter()
            .map(|f| talu_sys::CColumnFilter {
                column_id: f.column_id,
                op: f.op as u8,
                _pad: [0; 3],
                value: f.value,
            })
            .collect();

        let c_params = talu_sys::CScanParams {
            schema_id: params.schema_id,
            additional_schema_ids: if params.additional_schema_ids.is_empty() {
                ptr::null()
            } else {
                params.additional_schema_ids.as_ptr()
            },
            additional_schema_count: params.additional_schema_ids.len() as u32,
            filters: if c_filters.is_empty() {
                ptr::null()
            } else {
                c_filters.as_ptr()
            },
            filter_count: c_filters.len() as u32,
            dedup_column_id: params.dedup_column_id,
            delete_schema_id: params.delete_schema_id.unwrap_or(0),
            ts_column_id: params.ts_column_id,
            ttl_column_id: params.ttl_column_id.unwrap_or(0),
            limit: params.limit,
            cursor_ts: params.cursor_ts.unwrap_or(0),
            cursor_hash: params.cursor_hash.unwrap_or(0),
            payload_column_id: params.payload_column_id,
            reverse: params.reverse,
            _pad: [0; 3],
            extra_columns: if params.extra_columns.is_empty() {
                ptr::null()
            } else {
                params.extra_columns.as_ptr()
            },
            extra_column_count: params.extra_columns.len() as u32,
        };

        let mut iter = std::mem::MaybeUninit::<talu_sys::CRowIterator>::uninit();
        let code = unsafe { talu_sys::talu_db_table_scan(self.raw, &c_params, iter.as_mut_ptr()) };

        if code != ERROR_CODE_OK {
            return Err(TableError::from_code(code));
        }

        let iter = unsafe { iter.assume_init() };
        let result = extract_scan_result(&iter);

        // Free the arena-allocated C memory.
        let mut iter_for_free = iter;
        unsafe { talu_sys::talu_db_table_free_rows(&mut iter_for_free) };

        Ok(result)
    }

    /// Point lookup by primary hash, with optional legacy hash fallback.
    pub fn get(
        &self,
        schema_id: u16,
        pk_hash: u64,
        legacy_hash: Option<u64>,
    ) -> Result<Option<Row>, TableError> {
        let mut iter = std::mem::MaybeUninit::<talu_sys::CRowIterator>::uninit();
        let code = unsafe {
            talu_sys::talu_db_table_get(
                self.raw,
                schema_id,
                pk_hash,
                legacy_hash.unwrap_or(0),
                iter.as_mut_ptr(),
            )
        };

        if code != ERROR_CODE_OK {
            return Err(TableError::from_code(code));
        }

        let iter = unsafe { iter.assume_init() };
        let result = extract_scan_result(&iter);

        let mut iter_for_free = iter;
        unsafe { talu_sys::talu_db_table_free_rows(&mut iter_for_free) };

        Ok(result.rows.into_iter().next())
    }
}

impl Drop for TableHandle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { talu_sys::talu_db_table_close(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

/// Convert a C row iterator into Rust-owned scan result.
fn extract_scan_result(iter: &talu_sys::CRowIterator) -> ScanResult {
    let mut rows = Vec::with_capacity(iter.count as usize);

    if !iter.rows.is_null() && iter.count > 0 {
        for i in 0..iter.count as usize {
            let c_row = unsafe { &*iter.rows.add(i) };
            let mut scalars = Vec::with_capacity(c_row.scalar_count as usize);
            if !c_row.scalars.is_null() {
                for j in 0..c_row.scalar_count as usize {
                    let s = unsafe { &*c_row.scalars.add(j) };
                    scalars.push(ScalarColumn {
                        column_id: s.column_id,
                        value: s.value,
                    });
                }
            }

            let payload = if c_row.payload.is_null() || c_row.payload_len == 0 {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(c_row.payload, c_row.payload_len) }.to_vec()
            };

            rows.push(Row { scalars, payload });
        }
    }

    ScanResult {
        rows,
        has_more: iter.has_more,
    }
}
