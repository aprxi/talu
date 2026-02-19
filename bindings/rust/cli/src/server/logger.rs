//! Custom logger matching the Zig core log format.
//!
//! Output: `HH:MM:SS LEVEL [target] message`
//!
//! Uses UTC timestamps from `std::time::SystemTime` — no extra dependencies.
//!
//! Filter syntax (comma-separated, `!` negation, trailing `*` glob):
//!   "core::*"                     — only core scopes
//!   "!core::inference"            — everything except core::inference
//!   "core::*,server::gen"         — core scopes OR server::gen
//!   "!core::inference,!server::*" — exclude both

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

        // Apply scope filter.
        if let Some(filter) = FILTER.get() {
            if !filter.is_empty() && filter != "*" && !matches_filter(filter, record.target()) {
                return;
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

/// Match a single glob pattern against a scope (trailing `*` = prefix match).
fn glob_match(pattern: &str, scope: &str) -> bool {
    if pattern.is_empty() {
        return false;
    }
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        scope.starts_with(prefix)
    } else {
        scope == pattern
    }
}

/// Check if a scope matches a comma-separated filter with `!` negation support.
fn matches_filter(filter: &str, scope: &str) -> bool {
    let mut has_positive = false;
    let mut positive_match = false;

    for segment in filter.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }

        if let Some(pattern) = segment.strip_prefix('!') {
            // Negation: if scope matches, exclude it.
            if glob_match(pattern, scope) {
                return false;
            }
        } else {
            has_positive = true;
            if glob_match(segment, scope) {
                positive_match = true;
            }
        }
    }

    // If there were positive patterns, scope must have matched at least one.
    // If only negations, everything not excluded passes.
    if has_positive { positive_match } else { true }
}

static LOGGER: TaluServerLogger = TaluServerLogger;

pub fn init(level: log::LevelFilter, filter: Option<&str>) {
    if let Some(f) = filter {
        let _ = FILTER.set(f.to_string());
    }
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(level);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_exact_match() {
        assert!(matches_filter("core::inference", "core::inference"));
        assert!(!matches_filter("core::inference", "core::json"));
    }

    #[test]
    fn glob_prefix_match() {
        assert!(matches_filter("core::*", "core::inference"));
        assert!(matches_filter("core::*", "core::json"));
        assert!(!matches_filter("core::*", "server::http"));
    }

    #[test]
    fn negation_excludes() {
        assert!(!matches_filter("!core::inference", "core::inference"));
        assert!(matches_filter("!core::inference", "core::json"));
        assert!(matches_filter("!core::inference", "server::http"));
    }

    #[test]
    fn negation_glob() {
        assert!(!matches_filter("!core::*", "core::inference"));
        assert!(matches_filter("!core::*", "server::http"));
    }

    #[test]
    fn comma_separated_or() {
        assert!(matches_filter("core::inference,server::gen", "core::inference"));
        assert!(matches_filter("core::inference,server::gen", "server::gen"));
        assert!(!matches_filter("core::inference,server::gen", "server::http"));
    }

    #[test]
    fn multiple_negations() {
        let f = "!core::inference,!server::*";
        assert!(!matches_filter(f, "core::inference"));
        assert!(!matches_filter(f, "server::http"));
        assert!(!matches_filter(f, "server::gen"));
        assert!(matches_filter(f, "core::json"));
        assert!(matches_filter(f, "core::load"));
    }

    #[test]
    fn positive_with_negation() {
        let f = "core::*,!core::inference";
        assert!(matches_filter(f, "core::json"));
        assert!(matches_filter(f, "core::load"));
        assert!(!matches_filter(f, "core::inference"));
        assert!(!matches_filter(f, "server::http"));
    }

    #[test]
    fn wildcard_passes_all() {
        assert!(matches_filter("*", "anything"));
    }

    #[test]
    fn empty_segments_ignored() {
        assert!(matches_filter("core::inference,,server::gen", "server::gen"));
        assert!(matches_filter(",core::inference,", "core::inference"));
    }
}
