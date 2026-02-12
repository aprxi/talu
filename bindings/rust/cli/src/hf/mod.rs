//! Interactive HuggingFace model search TUI.
//!
//! Provides a search-as-you-type interface for browsing HuggingFace models
//! with sort/filter toggles and debounced API calls.

mod filters;
mod render;
mod selector;

use std::io::IsTerminal;
use std::path::Path;

use anyhow::{bail, Result};

use talu::HfSearchResult;

use selector::{HfSearchSelector, PickerMode};

use crate::tui_common::{enter_tui, leave_tui};

/// Maximum results fetched per API call (overfetch to allow client-side size filtering).
const FETCH_LIMIT: usize = 500;
/// Maximum results displayed to the user.
const DISPLAY_LIMIT: usize = 100;

// ---------------------------------------------------------------------------
// Search response channel
// ---------------------------------------------------------------------------

struct SearchResponse {
    generation: u64,
    /// Primary = user's chosen sort; secondary = supplemental downloads fetch.
    is_primary: bool,
    results: std::result::Result<Vec<HfSearchResult>, String>,
}

// ---------------------------------------------------------------------------
// Label positions (x-offset of each label in the status line)
// ---------------------------------------------------------------------------

/// Tracks x-offsets of the filter labels in the search bar status line.
#[derive(Default)]
struct LabelPositions {
    task_x: u16,
    size_x: u16,
    date_x: u16,
    lib_x: u16,
    sort_x: u16,
}

impl LabelPositions {
    fn x_for(&self, mode: PickerMode) -> u16 {
        match mode {
            PickerMode::Task => self.task_x,
            PickerMode::Size => self.size_x,
            PickerMode::Date => self.date_x,
            PickerMode::Library => self.lib_x,
            PickerMode::Sort => self.sort_x,
        }
    }
}

// ---------------------------------------------------------------------------
// Display helpers
// ---------------------------------------------------------------------------

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Civil calendar algorithm from Howard Hinnant.
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn format_downloads(n: i64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

fn format_params(n: i64) -> String {
    if n <= 0 {
        return "-".into();
    }
    if n >= 1_000_000_000 {
        let b = n as f64 / 1_000_000_000.0;
        if b >= 10.0 {
            format!("{:.0}B", b)
        } else {
            format!("{:.1}B", b)
        }
    } else if n >= 1_000_000 {
        format!("{:.0}M", n as f64 / 1_000_000.0)
    } else {
        format!("{}K", n / 1_000)
    }
}

fn format_date_iso(iso: &str) -> String {
    if iso.len() < 10 {
        return "-".into();
    }
    let year: i32 = iso[0..4].parse().unwrap_or(0);
    let month: usize = iso[5..7].parse().unwrap_or(0);
    let day: u32 = iso[8..10].parse().unwrap_or(0);

    let months = [
        "", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    if (1..=12).contains(&month) {
        let now_year = 2026;
        if year == now_year {
            format!("{} {:2}", months[month], day)
        } else {
            format!("{} {:2} {}", months[month], day, year)
        }
    } else {
        "-".into()
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the interactive HuggingFace search TUI.
/// Returns `Some(model_id)` when the user selects a model, `None` on cancel.
pub fn run_hf_search(pin_db_path: &Path) -> Result<Option<String>> {
    if !std::io::stdout().is_terminal() {
        bail!("Interactive search requires a terminal.");
    }

    // Suppress core logging while the TUI is active (stderr output corrupts the display).
    let saved_log_level = talu::logging::get_log_level();
    talu::logging::set_log_level(talu::LogLevel::Off);
    let mut terminal = enter_tui()?;
    let mut selector = HfSearchSelector::new(pin_db_path)?;
    let result = selector.run(&mut terminal);
    leave_tui(&mut terminal);
    talu::logging::set_log_level(saved_log_level);

    result
}
