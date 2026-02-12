//! Shared TUI helpers for interactive terminal selectors.
//!
//! Provides common display formatting, text highlighting, and terminal
//! setup/teardown used by both the model selector and HF search TUI.

use std::io;

use crossterm::execute;
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::Terminal;

/// Truncate a string to `max_len` chars, appending "..." if truncated.
pub fn truncate_name(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Build a Line with highlighted matching substrings.
pub fn highlight_text(text: &str, query: &str, base: Style) -> Line<'static> {
    if query.is_empty() {
        return Line::styled(text.to_string(), base);
    }

    let text_lower = text.to_lowercase();
    let q_lower = query.to_lowercase();
    let highlight = base.fg(Color::Yellow).add_modifier(Modifier::BOLD);

    let mut spans = Vec::new();
    let mut pos = 0;

    while pos < text.len() {
        if let Some(start) = text_lower[pos..].find(&q_lower) {
            let abs_start = pos + start;
            let abs_end = abs_start + query.len();
            if abs_start > pos {
                spans.push(Span::styled(text[pos..abs_start].to_string(), base));
            }
            spans.push(Span::styled(
                text[abs_start..abs_end].to_string(),
                highlight,
            ));
            pos = abs_end;
        } else {
            spans.push(Span::styled(text[pos..].to_string(), base));
            break;
        }
    }

    Line::from(spans)
}

/// Format a byte count as a human-readable string (e.g., "1.2 GB").
pub fn format_size(bytes: u64) -> String {
    let b = bytes as f64;
    let kb = 1024.0;
    let mb = kb * 1024.0;
    let gb = mb * 1024.0;

    if b >= gb {
        format!("{:.1} GB", b / gb)
    } else if b >= mb {
        format!("{:.1} MB", b / mb)
    } else if b >= kb {
        format!("{:.1} KB", b / kb)
    } else {
        format!("{} B", bytes)
    }
}

/// Format a Unix timestamp as a short date string.
pub fn format_date(timestamp: i64) -> String {
    if timestamp == 0 {
        return "-".into();
    }

    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    let time = UNIX_EPOCH + Duration::from_secs(timestamp as u64);
    let now = SystemTime::now();
    let age_secs = now.duration_since(time).unwrap_or(Duration::ZERO).as_secs();
    let age_days = age_secs / 86400;

    let secs_since_epoch = timestamp;
    let days = secs_since_epoch / 86400;
    let remaining = secs_since_epoch % 86400;
    let hours = remaining / 3600;
    let mins = (remaining % 3600) / 60;

    let mut year = 1970i32;
    let mut day_of_year = days;

    loop {
        let days_in_year: i64 = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
            366
        } else {
            365
        };
        if day_of_year < days_in_year {
            break;
        }
        day_of_year -= days_in_year;
        year += 1;
    }

    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let month_days: [i64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];

    let mut month = 0usize;
    for (i, &md) in month_days.iter().enumerate() {
        if day_of_year < md {
            month = i;
            break;
        }
        day_of_year -= md;
    }
    let day = day_of_year + 1;

    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    if age_days < 180 {
        format!("{} {:2} {:02}:{:02}", months[month], day, hours, mins)
    } else {
        format!("{} {:2}  {}", months[month], day, year)
    }
}

/// Enter TUI mode: enable raw mode, switch to alternate screen, return terminal.
pub fn enter_tui() -> anyhow::Result<Terminal<CrosstermBackend<io::Stdout>>> {
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

/// Leave TUI mode: disable raw mode, leave alternate screen, restore cursor.
pub fn leave_tui(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) {
    let _ = crossterm::terminal::disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
}
