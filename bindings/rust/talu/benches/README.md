# Benchmarks

Criterion-based benchmarks for the Talu Rust bindings. Benchmarks read from
persistent datasets stored as real `~/.talu/` profiles so startup cost is paid
once, not on every run.

## Quick start

```bash
cd bindings/rust/talu/benches

# 1. Generate datasets (one-time setup)
make bench-prep dataset=small         # 50 sessions, 500 msgs — ~1 MB, ~4s
make bench-prep dataset=medium        # 200 sessions, 10k msgs — ~19 MB, ~60s

# 2. Sanity check (<10s, runs bench_quick example)
make bench

# 3. Run a specific scope
make bench scope=search
```

## Datasets

Benchmarks read from pre-generated datasets at `~/.talu/db/bench-{name}/`. Each
dataset is a real talu bucket containing both chat data (sessions with
alternating user/assistant messages) and vector data (384-dimensional embeddings).

| Dataset  | Sessions | Msgs/session | Total msgs | Vectors | Approx disk |
|----------|----------|--------------|------------|---------|-------------|
| `small`  | 50       | 10           | 500        | 500     | ~1 MB       |
| `medium` | 200      | 50           | 10,000     | 10,000  | ~19 MB      |
| `large`  | 1,000    | 100          | 100,000    | 100,000 | ~190 MB     |

Most benchmarks use the `medium` dataset. Generate it before running bench suites.

### Dataset management

```bash
make bench-prep dataset=medium          # generate (skips if exists)
make bench-prep dataset=medium force=1  # regenerate from scratch
make bench-prep dataset=all             # generate all sizes
make bench-info                         # show stats for all datasets
make bench-info dataset=medium          # show stats for one
```

The prep tool (`examples/bench_prep.rs`) creates each dataset as a real profile
registered in `~/.talu/config.toml`. Sessions are named `session-0000` through
`session-NNNN`, each containing alternating user/assistant messages across 5
rotating topics. A matching 384d vector is generated for each message.

Override the dataset path for CI with `TALU_BENCH_DB=<path>`.

## Scopes

Benchmarks are organized into 4 domain scopes, each a separate Criterion
harness (separate binary, independent compilation):

### `search` — Vector similarity search

| Benchmark | What it measures |
|-----------|-----------------|
| `search_10k_vectors` | Single query on 10k corpus (384d) |
| `search_batch_throughput` | 100 queries on 10k corpus |
| `scale_1m/search_1m_single` | Single query on 1M corpus (128d, ~512 MB) |

### `storage` — Ingest and lifecycle

| Benchmark | What it measures |
|-----------|-----------------|
| `ingest_throughput/batch_size_1` | 100 individual appends (IOPS-bound) |
| `ingest_throughput/batch_size_100` | Single batch of 100 (bandwidth-bound) |
| `ingest_10k_vectors` | Bulk ingest 10k vectors (384d) |
| `lifecycle/cold_open_10k` | Open existing store with 10k vectors |
| `lifecycle/cold_load_10k` | Open + load 10k vectors into memory |

### `api` — Chat and session operations

| Benchmark | What it measures |
|-----------|-----------------|
| `append_message_latency` | Per-message FFI append (lock + WAL + fsync) |
| `metadata_update_latency` | `notify_session_update` cost |
| `sessions/list_no_query` | List 200 sessions, no search (baseline) |
| `sessions/search_title_hit` | Text search matching title metadata |
| `sessions/search_content_partial` | Text search matching 1/5 topics |
| `sessions/search_content_all` | Text search matching all sessions |
| `sessions/search_no_match` | Text search with zero results (worst-case scan) |

### `rag` — Retrieval-augmented generation patterns

| Benchmark | What it measures |
|-----------|-----------------|
| `rag_context_retrieval_20_of_50` | Load conversation + read 20 messages by index |
| `rag_session_scan/list_200_sessions` | List all sessions (limit=200) |
| `rag_session_scan/list_and_filter_200` | List + client-side title filter |
| `rag_chat_turn_latency` | Append + embed + search loop |

## Running benchmarks

All commands from `bindings/rust/talu/benches/`:

```bash
# Quick sanity check — runs each operation once, prints a table (<10s)
make bench

# Full Criterion suite for one scope
make bench scope=search

# Filter to specific benchmarks within a scope
make bench scope=search filter=batch
make bench scope=api filter=sessions/search

# List all scopes and their benchmarks (no compilation)
make bench scope=help

# List benchmarks in one scope
make bench scope=search filter=help

# Full report with JSON summary
make bench-report
```

## Architecture

```
benches/
├── Makefile              # All bench targets
├── common/
│   ├── mod.rs            # Shared helpers (corpus builders, fresh handles)
│   └── dataset.rs        # Persistent dataset path resolver
├── api/
│   ├── main.rs           # Criterion harness + @bench tags
│   ├── chat.rs           # append_message benchmark
│   ├── metadata.rs       # notify_session_update benchmark
│   └── sessions.rs       # list/search session benchmarks
├── search/
│   ├── main.rs           # Criterion harness + @bench tags
│   ├── vector.rs         # 10k corpus search benchmarks
│   └── scale.rs          # 1M corpus scale benchmark
├── storage/
│   ├── main.rs           # Criterion harness + @bench tags
│   ├── ingest.rs         # Append throughput benchmarks
│   └── lifecycle.rs      # Cold open/load benchmarks
└── rag/
    ├── main.rs           # Criterion harness + @bench tags
    ├── context.rs        # Context retrieval benchmark
    ├── session_scan.rs   # Session listing benchmarks
    └── chat_turn.rs      # Full RAG turn benchmark
```

Each scope's `main.rs` contains `// @bench:` tags listing every benchmark name
and a short description. `make bench scope=help` extracts these tags to produce
the benchmark listing without compiling anything.

### Read vs write separation

Benchmarks that **read** (search, list, load) use persistent datasets. Benchmarks
that **write** (append, ingest) create fresh temp directories so they never
corrupt shared data.

### Bench harness configuration

Each scope is declared as a `[[bench]]` in `Cargo.toml` with `harness = false`
(custom Criterion main). This means each scope compiles as a separate binary
and can be run independently.
