//! Raw SQLite C bindings for the DB SQL virtual table/query layer.
//!
//! This module intentionally exposes the raw `sqlite3` API surface via cImport.

pub const sqlite3 = @cImport({
    @cInclude("sqlite3.h");
});
