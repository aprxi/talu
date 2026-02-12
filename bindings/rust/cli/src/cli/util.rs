/// Default max output tokens used by CLI commands when TOKENS is unset.
pub(super) const DEFAULT_MAX_TOKENS: usize = 1024;

/// Truncate a string to max_len characters
pub(super) fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        s.chars().take(max_len).collect()
    }
}

pub(super) fn capitalize_first(s: &str) -> String {
    let mut chars: Vec<char> = s.chars().collect();
    if !chars.is_empty() {
        chars[0] = chars[0].to_ascii_uppercase();
    }
    chars.into_iter().collect()
}

pub(super) fn format_size(bytes: u64) -> String {
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

/// Format Unix timestamp as a date string (Jan 13 or 2025)
pub(super) fn format_date(timestamp: i64) -> String {
    if timestamp == 0 {
        return "-".to_string();
    }

    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    let time = UNIX_EPOCH + Duration::from_secs(timestamp as u64);
    let now = SystemTime::now();

    // Calculate days since the file was modified
    let age_secs = now.duration_since(time).unwrap_or(Duration::ZERO).as_secs();
    let age_days = age_secs / 86400;

    // Get the date components
    let secs_since_epoch = timestamp;
    let days = secs_since_epoch / 86400;
    let remaining = secs_since_epoch % 86400;
    let hours = remaining / 3600;
    let mins = (remaining % 3600) / 60;

    // Simple date calculation (approximate, good enough for display)
    // Days since epoch (Jan 1, 1970)
    let mut year = 1970i32;
    let mut day_of_year = days;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if day_of_year < days_in_year {
            break;
        }
        day_of_year -= days_in_year;
        year += 1;
    }

    let (month, day) = day_of_year_to_month_day(day_of_year as i32, is_leap_year(year));

    let month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month_name = month_names[(month - 1) as usize];

    // If less than 6 months old, show "Mon DD HH:MM"
    // Otherwise show "Mon DD  YYYY"
    if age_days < 180 {
        format!("{} {:2} {:02}:{:02}", month_name, day, hours, mins)
    } else {
        format!("{} {:2}  {}", month_name, day, year)
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn day_of_year_to_month_day(day: i32, leap: bool) -> (i32, i32) {
    let days_in_months: [i32; 12] = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut remaining = day;
    for (i, &days) in days_in_months.iter().enumerate() {
        if remaining < days {
            return ((i + 1) as i32, remaining + 1);
        }
        remaining -= days;
    }
    (12, 31) // Fallback
}
