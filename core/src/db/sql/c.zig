//! Raw SQLite C bindings for the DB SQL virtual table/query layer.
//!
//! This module intentionally exposes the raw `sqlite3` API surface via cImport.

pub const sqlite3 = @cImport({
    @cInclude("sqlite3.h");
});

/// SQLITE_TRANSIENT sentinel: tells SQLite to copy the data immediately.
/// The C macro ((void(*)(void*))-1) fails Zig 0.15+ comptime alignment checks.
/// We use a mutable var to force runtime evaluation and bypass the check.
var sqlite_transient_addr: usize = ~@as(usize, 0);

pub inline fn SQLITE_TRANSIENT() sqlite3.sqlite3_destructor_type {
    return @ptrFromInt(sqlite_transient_addr);
}
