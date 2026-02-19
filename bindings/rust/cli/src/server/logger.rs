//! Custom logger matching the Zig core log format.
//!
//! Output: `HH:MM:SS LEVEL [target] message`
//!
//! Uses UTC timestamps from `std::time::SystemTime` — no extra dependencies.

use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

/// Stored filter pattern for Rust-side log scope filtering.
static FILTER: OnceLock<String> = OnceLock::new();

struct TaluServerLogger;

impl log::Log for TaluServerLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        // Apply scope filter (glob-style: trailing * = prefix match).
        if let Some(filter) = FILTER.get() {
            if !filter.is_empty() && filter != "*" {
                let target = record.target();
                if filter.ends_with('*') {
                    let prefix = &filter[..filter.len() - 1];
                    if !target.starts_with(prefix) {
                        return;
                    }
                } else if target != filter {
                    return;
                }
            }
        }

        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let sec_of_day = secs % 86400;
        let h = sec_of_day / 3600;
        let m = (sec_of_day % 3600) / 60;
        let s = sec_of_day % 60;

        eprintln!(
            "{:02}:{:02}:{:02} {:<5} [{}] {}",
            h,
            m,
            s,
            record.level(),
            record.target(),
            record.args()
        );
    }

    fn flush(&self) {}
}

static LOGGER: TaluServerLogger = TaluServerLogger;

pub fn init(level: log::LevelFilter, filter: Option<&str>) {
    if let Some(f) = filter {
        let _ = FILTER.set(f.to_string());
    }
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(level);
}
