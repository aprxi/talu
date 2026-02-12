//! Quick vector benchmarks: ingest, search, batch-search, cold-load, fsync.

use std::time::Instant;
use tempfile::TempDir;

use talu::vector::VectorStore;
use talu::Durability;

use super::BenchResult;

const DIMS: u32 = 128;

pub fn run() -> Vec<BenchResult> {
    let mut results = Vec::new();

    // Ingest 10k vectors (batch)
    {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path().to_str().unwrap()).unwrap();
        let ids: Vec<u64> = (0..10_000).collect();
        let vecs: Vec<f32> = vec![0.5; 10_000 * DIMS as usize];

        let start = Instant::now();
        store.append(&ids, &vecs, DIMS).unwrap();
        results.push(BenchResult {
            name: "ingest_10k_vectors",
            elapsed: start.elapsed(),
            ops: 1,
        });
    }

    // Search 10k (pre-loaded)
    let search_dir = TempDir::new().unwrap();
    {
        let store = VectorStore::open(search_dir.path().to_str().unwrap()).unwrap();
        let ids: Vec<u64> = (0..10_000).collect();
        let vecs: Vec<f32> = (0..10_000)
            .flat_map(|i| {
                let v = i as f32 * 0.0001;
                std::iter::repeat(v).take(DIMS as usize)
            })
            .collect();
        store.append(&ids, &vecs, DIMS).unwrap();

        let query = vec![0.5_f32; DIMS as usize];

        // Single search (3 iterations)
        let start = Instant::now();
        for _ in 0..3 {
            let r = store.search(&query, 10).unwrap();
            assert_eq!(r.ids.len(), 10);
        }
        results.push(BenchResult {
            name: "search_10k_top10",
            elapsed: start.elapsed(),
            ops: 3,
        });

        // Batch search (100 queries)
        let queries: Vec<f32> = (0..100)
            .flat_map(|i| {
                let v = i as f32 * 0.01;
                std::iter::repeat(v).take(DIMS as usize)
            })
            .collect();

        let start = Instant::now();
        let r = store.search_batch(&queries, DIMS, 10).unwrap();
        assert_eq!(r.query_count, 100);
        results.push(BenchResult {
            name: "search_batch_100q",
            elapsed: start.elapsed(),
            ops: 1,
        });

        // Cold load
        drop(store);
        let start = Instant::now();
        let store2 = VectorStore::open(search_dir.path().to_str().unwrap()).unwrap();
        let loaded = store2.load().unwrap();
        assert_eq!(loaded.ids.len(), 10_000);
        results.push(BenchResult {
            name: "cold_load_10k",
            elapsed: start.elapsed(),
            ops: 1,
        });
    }

    // Ingest batch=1 with fsync (100 individual appends)
    {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path().to_str().unwrap()).unwrap();
        let vec_data: Vec<f32> = vec![0.5; DIMS as usize];

        let start = Instant::now();
        for i in 0..100u64 {
            store.append(&[i], &vec_data, DIMS).unwrap();
        }
        results.push(BenchResult {
            name: "ingest_100x1_fsync",
            elapsed: start.elapsed(),
            ops: 100,
        });
    }

    // Ingest batch=1 async (100 individual appends, no fsync)
    {
        let dir = TempDir::new().unwrap();
        let store = VectorStore::open(dir.path().to_str().unwrap()).unwrap();
        store.set_durability(Durability::AsyncOs).unwrap();
        let vec_data: Vec<f32> = vec![0.5; DIMS as usize];

        let start = Instant::now();
        for i in 0..100u64 {
            store.append(&[i], &vec_data, DIMS).unwrap();
        }
        results.push(BenchResult {
            name: "ingest_100x1_async",
            elapsed: start.elapsed(),
            ops: 100,
        });
    }

    results
}
