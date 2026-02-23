//! Table-plane aliases over document storage.
//!
//! This module aligns SDK naming with the unified DB API. It currently maps
//! table operations to the document-backed table implementation.

pub use crate::documents::{
    ChangeAction, ChangeRecord, CompactionStats, DocumentError as TableError,
    DocumentRecord as TableRecord, DocumentSummary as TableSummary, DocumentsHandle as TableHandle,
    SearchResult as TableSearchResult,
};
