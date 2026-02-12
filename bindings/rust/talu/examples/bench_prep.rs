//! Persistent benchmark dataset generator.
//!
//! Creates reusable datasets as real `~/.talu/` profiles so benchmarks don't
//! rebuild data on every run. Each dataset contains both chat data (sessions +
//! messages) and vector data in the same bucket.
//!
//! Usage:
//!   cargo run --release --example bench_prep -- --dataset medium
//!   cargo run --release --example bench_prep -- --dataset all
//!   cargo run --release --example bench_prep -- --dataset medium --force
//!   cargo run --release --example bench_prep -- --info medium
//!   cargo run --release --example bench_prep -- --info all

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use talu::responses::{MessageRole, ResponsesView};
use talu::vector::VectorStore;
use talu::ChatHandle;

// ---------------------------------------------------------------------------
// Dataset definitions
// ---------------------------------------------------------------------------

struct DatasetSpec {
    name: &'static str,
    sessions: usize,
    msgs_per_session: usize,
}

const DATASETS: &[DatasetSpec] = &[
    DatasetSpec {
        name: "small",
        sessions: 50,
        msgs_per_session: 10,
    },
    DatasetSpec {
        name: "medium",
        sessions: 200,
        msgs_per_session: 50,
    },
    DatasetSpec {
        name: "large",
        sessions: 1_000,
        msgs_per_session: 100,
    },
];

const DIMS: u32 = 384;

const TOPICS: &[&str] = &[
    "quantum mechanics and wave functions",
    "neural networks and gradient descent",
    "distributed systems and consensus protocols",
    "compiler optimization and register allocation",
    "database indexing and query planning",
];

// ---------------------------------------------------------------------------
// Config types (matches ~/.talu/config.toml structure from CLI)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Serialize, Deserialize)]
struct TaluConfig {
    #[serde(default)]
    profiles: BTreeMap<String, ProfileConfig>,
    #[serde(default)]
    default_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProfileConfig {
    bucket: PathBuf,
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn talu_home() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".talu")
}

fn dataset_bucket(name: &str) -> PathBuf {
    talu_home().join("db").join(format!("bench-{name}"))
}

fn config_path() -> PathBuf {
    talu_home().join("config.toml")
}

// ---------------------------------------------------------------------------
// Config read/write
// ---------------------------------------------------------------------------

fn load_config() -> TaluConfig {
    let path = config_path();
    if !path.exists() {
        return TaluConfig::default();
    }
    let contents = std::fs::read_to_string(&path).expect("read config.toml");
    toml::from_str(&contents).expect("parse config.toml")
}

fn save_config(config: &TaluConfig) {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create config dir");
    }
    let contents = toml::to_string_pretty(config).expect("serialize config");
    std::fs::write(&path, contents).expect("write config.toml");
}

// ---------------------------------------------------------------------------
// Bucket initialization (matches CLI ensure_bucket logic)
// ---------------------------------------------------------------------------

fn ensure_bucket(bucket_path: &Path) {
    if bucket_path.exists() {
        return;
    }
    std::fs::create_dir_all(bucket_path).expect("create bucket dir");

    // store.key: 16 random bytes
    let key_path = bucket_path.join("store.key");
    let mut key_data = [0u8; 16];
    getrandom::fill(&mut key_data).expect("generate random key");
    std::fs::write(&key_path, &key_data).expect("write store.key");

    // manifest.json
    let manifest_path = bucket_path.join("manifest.json");
    let manifest = r#"{"version": 1, "segments": [], "last_compaction_ts": 0}"#;
    std::fs::write(&manifest_path, manifest).expect("write manifest.json");
}

/// Register profile `bench-{name}` in config.toml.
fn register_profile(name: &str, bucket: &Path) {
    let profile_name = format!("bench-{name}");
    let mut config = load_config();
    config.profiles.insert(
        profile_name,
        ProfileConfig {
            bucket: bucket.to_path_buf(),
        },
    );
    save_config(&config);
}

// ---------------------------------------------------------------------------
// Dataset generation
// ---------------------------------------------------------------------------

fn generate_dataset(spec: &DatasetSpec, force: bool) {
    let bucket = dataset_bucket(spec.name);
    let db_path = bucket.to_str().expect("bucket path");

    if bucket.exists() && !force {
        println!(
            "  bench-{}: already exists at {} (use --force to recreate)",
            spec.name,
            bucket.display()
        );
        return;
    }

    if bucket.exists() && force {
        std::fs::remove_dir_all(&bucket).expect("remove existing dataset");
        println!("  bench-{}: removed existing dataset", spec.name);
    }

    ensure_bucket(&bucket);
    register_profile(spec.name, &bucket);

    let start = Instant::now();
    let total_msgs = spec.sessions * spec.msgs_per_session;
    println!(
        "  bench-{}: generating {} sessions x {} msgs = {} messages + {} vectors ({}d)...",
        spec.name, spec.sessions, spec.msgs_per_session, total_msgs, total_msgs, DIMS
    );

    // -- Chat data: sessions + messages --
    let mut global_msg_idx: u64 = 0;
    let mut all_vector_ids: Vec<u64> = Vec::with_capacity(total_msgs);
    let mut all_vectors: Vec<f32> = Vec::with_capacity(total_msgs * DIMS as usize);

    for i in 0..spec.sessions {
        let sid = format!("session-{i:04}");
        let topic = TOPICS[i % TOPICS.len()];

        let chat = ChatHandle::new(None).expect("ChatHandle::new");
        chat.set_storage_db(db_path, &sid).expect("set_storage_db");

        for j in 0..spec.msgs_per_session {
            let role = if j % 2 == 0 {
                MessageRole::User
            } else {
                MessageRole::Assistant
            };
            let msg = if role == MessageRole::User {
                format!(
                    "Message {j} in session {i}: asking about {topic}. \
                     This is a benchmark payload with enough content to exercise \
                     the substring scanner realistically."
                )
            } else {
                format!(
                    "Message {j} in session {i}: explaining {topic}. \
                     The key concepts involve detailed analysis and reasoning \
                     about the underlying principles and practical applications."
                )
            };
            let rc = unsafe {
                talu_sys::talu_responses_append_message(
                    chat.responses().as_ptr(),
                    role as u8,
                    msg.as_bytes().as_ptr(),
                    msg.as_bytes().len(),
                )
            };
            assert!(rc >= 0, "append_message failed: {rc}");

            // Assistant messages need explicit finalization to trigger storage.
            // The core creates them as in_progress (for streaming); setting
            // status=completed (2) calls notifyStorage and persists the item.
            if role == MessageRole::Assistant {
                let status_rc = unsafe {
                    talu_sys::talu_responses_set_item_status(
                        chat.responses().as_ptr(),
                        rc as usize,
                        2, // ItemStatus::completed
                    )
                };
                assert!(status_rc == 0, "set_item_status failed: {status_rc}");
            }

            // Corresponding vector: deterministic based on global index.
            all_vector_ids.push(global_msg_idx);
            let val = global_msg_idx as f32 * 0.0001;
            all_vectors.extend(std::iter::repeat(val).take(DIMS as usize));
            global_msg_idx += 1;
        }

        chat.notify_session_update(
            Some("bench-model"),
            Some(&format!("Session {i}")),
            Some("active"),
        )
        .expect("notify_session_update");

        if (i + 1) % 100 == 0 {
            println!("    ... {}/{} sessions", i + 1, spec.sessions);
        }
    }

    // -- Vector data: bulk ingest --
    let store = VectorStore::open(db_path).expect("VectorStore::open");

    // Ingest in chunks to avoid excessive memory for large datasets.
    let chunk_size = 10_000;
    for chunk_start in (0..total_msgs).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(total_msgs);
        let id_slice = &all_vector_ids[chunk_start..chunk_end];
        let vec_start = chunk_start * DIMS as usize;
        let vec_end = chunk_end * DIMS as usize;
        let vec_slice = &all_vectors[vec_start..vec_end];
        store
            .append(id_slice, vec_slice, DIMS)
            .expect("vector append");
    }
    drop(store);

    let elapsed = start.elapsed();
    let disk_bytes = dir_size(&bucket);
    println!(
        "  bench-{}: done in {:.1}s — {} ({} msgs, {} vectors)",
        spec.name,
        elapsed.as_secs_f64(),
        format_bytes(disk_bytes),
        total_msgs,
        total_msgs,
    );
}

// ---------------------------------------------------------------------------
// Info display
// ---------------------------------------------------------------------------

fn show_info(name: &str) {
    let bucket = dataset_bucket(name);
    if !bucket.exists() {
        println!("  bench-{name}: not found (run: make bench-prep dataset={name})");
        return;
    }

    let disk_bytes = dir_size(&bucket);

    // Count sessions via StorageHandle.
    let db_path = bucket.to_str().expect("bucket path");
    let storage = talu::StorageHandle::open(db_path);
    let (session_count, msg_estimate) = match storage {
        Ok(s) => {
            let count = s.session_count().unwrap_or(0);
            // Find the matching spec for message estimate.
            let spec = DATASETS.iter().find(|d| d.name == name);
            let msgs = spec.map(|s| s.sessions * s.msgs_per_session).unwrap_or(0);
            (count, msgs)
        }
        Err(_) => (0, 0),
    };

    // Count vectors.
    let vector_store = VectorStore::open(db_path);
    let vector_count = match vector_store {
        Ok(vs) => vs.load().map(|l| l.ids.len()).unwrap_or(0),
        Err(_) => 0,
    };

    println!("  bench-{name}: {path}", path = bucket.display(),);
    println!(
        "    sessions: {session_count}, messages: ~{msg_estimate}, vectors: {vector_count}, disk: {disk}",
        disk = format_bytes(disk_bytes),
    );
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let meta = entry.metadata();
            if let Ok(m) = meta {
                if m.is_dir() {
                    total += dir_size(&entry.path());
                } else {
                    total += m.len();
                }
            }
        }
    }
    total
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn find_spec(name: &str) -> Option<&'static DatasetSpec> {
    DATASETS.iter().find(|d| d.name == name)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  bench_prep --dataset <small|medium|large|all> [--force]");
    eprintln!("  bench_prep --info <small|medium|large|all>");
    eprintln!();
    eprintln!("Datasets:");
    for spec in DATASETS {
        let total = spec.sessions * spec.msgs_per_session;
        eprintln!(
            "  {:<8} {} sessions x {} msgs = {} messages + {} vectors",
            spec.name, spec.sessions, spec.msgs_per_session, total, total
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        print_usage();
        std::process::exit(1);
    }

    let mode = &args[1];
    let target = &args[2];
    let force = args.iter().any(|a| a == "--force");

    match mode.as_str() {
        "--dataset" => {
            println!("bench-prep: generating datasets");
            if target == "all" {
                for spec in DATASETS {
                    generate_dataset(spec, force);
                }
            } else if let Some(spec) = find_spec(target) {
                generate_dataset(spec, force);
            } else {
                eprintln!("Unknown dataset: {target}");
                print_usage();
                std::process::exit(1);
            }
        }
        "--info" => {
            println!("bench-info:");
            if target == "all" {
                for spec in DATASETS {
                    show_info(spec.name);
                }
            } else if find_spec(target).is_some() {
                show_info(target);
            } else {
                eprintln!("Unknown dataset: {target}");
                print_usage();
                std::process::exit(1);
            }
        }
        _ => {
            print_usage();
            std::process::exit(1);
        }
    }
}
