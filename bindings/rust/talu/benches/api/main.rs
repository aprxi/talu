mod chat;
mod metadata;
mod sessions;

use criterion::{criterion_group, criterion_main};

// @bench: append_message_latency          — per-message FFI append (lock + WAL + fsync)
// @bench: metadata_update_latency         — notify_session_update cost
// @bench: sessions/list_no_query          — list 200 sessions, no search (baseline)
// @bench: sessions/search_title_hit       — text search matching title metadata
// @bench: sessions/search_content_partial — text search matching 1/5 topics (40 sessions)
// @bench: sessions/search_content_all     — text search matching all sessions (limit-clamped)
// @bench: sessions/search_no_match        — text search with zero results (worst-case scan)

criterion_group!(
    api_benches,
    chat::bench_append_message,
    metadata::bench_metadata_update,
    sessions::bench_list_sessions,
);
criterion_main!(api_benches);
