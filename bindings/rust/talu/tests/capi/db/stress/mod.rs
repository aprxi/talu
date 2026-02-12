//! Stress & boundary condition tests.
//!
//! Exercises the storage engine at resource limits:
//! - Auto-flush threshold (64KB buffer → block materialization)
//! - Huge payloads (single messages exceeding typical buffer sizes)

mod flush;
mod payloads;
mod schema;
