mod chat_turn;
mod context;
mod session_scan;

use criterion::{criterion_group, criterion_main};

// @bench: rag_context_retrieval_20_of_50      — load conversation + read 20 messages by index
// @bench: rag_session_scan/list_200_sessions  — list all sessions (limit=200)
// @bench: rag_session_scan/list_and_filter_200 — list + client-side title filter
// @bench: rag_chat_turn_latency               — append → embed → search loop

criterion_group!(
    rag_benches,
    context::bench_context_retrieval,
    session_scan::bench_session_list_scan,
    chat_turn::bench_chat_turn,
);
criterion_main!(rag_benches);
