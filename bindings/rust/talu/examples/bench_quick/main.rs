//! Quick single-pass benchmark for all DB operations.
//!
//! Runs each operation once (or a small fixed count) and prints a summary table.
//! Completes in <10 seconds — use for rapid feedback during development.
//!
//! For statistically rigorous benchmarks, use: cargo bench
//!
//! Usage: cargo run --release --example bench_quick

mod chat;
mod vector;

pub struct BenchResult {
    pub name: &'static str,
    pub elapsed: std::time::Duration,
    pub ops: usize,
}

impl BenchResult {
    fn per_op_display(&self) -> String {
        let ns = self.elapsed.as_nanos() as f64 / self.ops as f64;
        if ns >= 1_000_000.0 {
            format!("{:.2} ms", ns / 1_000_000.0)
        } else if ns >= 1_000.0 {
            format!("{:.2} \u{00b5}s", ns / 1_000.0)
        } else {
            format!("{:.0} ns", ns)
        }
    }

    fn total_display(&self) -> String {
        let ms = self.elapsed.as_secs_f64() * 1000.0;
        if ms >= 1000.0 {
            format!("{:.2} s", ms / 1000.0)
        } else {
            format!("{:.1} ms", ms)
        }
    }
}

fn main() {
    let mut results: Vec<BenchResult> = Vec::new();
    results.extend(vector::run());
    results.extend(chat::run());

    let name_w = results.iter().map(|r| r.name.len()).max().unwrap();
    let per_op_w = results
        .iter()
        .map(|r| r.per_op_display().len())
        .max()
        .unwrap();
    let total_w = results
        .iter()
        .map(|r| r.total_display().len())
        .max()
        .unwrap();

    println!(
        "\n{:<name_w$}  {:>per_op_w$}  {:>total_w$}  {}",
        "Benchmark", "Per-op", "Total", "Ops"
    );
    println!(
        "{:-<name_w$}  {:-<per_op_w$}  {:-<total_w$}  {}",
        "", "", "", "---"
    );
    for r in &results {
        println!(
            "{:<name_w$}  {:>per_op_w$}  {:>total_w$}  {}",
            r.name,
            r.per_op_display(),
            r.total_display(),
            r.ops
        );
    }
    println!();
}
