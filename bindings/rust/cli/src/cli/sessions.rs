use anyhow::{bail, Result};

use talu::responses::ItemType;
use talu::responses::MessageRole;
use talu::responses::ResponsesView;
use talu::{SessionRecord, StorageError, StorageHandle};

use super::db::{print_session_items, print_session_pretty};
use super::util::truncate_str;

/// Print sessions (newest first) with item counts.
pub(super) fn print_sessions_with_stats(handle: &StorageHandle, sessions: &[SessionRecord]) {
    if sessions.is_empty() {
        println!("Error: No sessions found.");
        return;
    }

    println!(
        "{:<36} | {:<16} | {:<14} | {:<7} | {:<7} | {:<8} | {:<12}",
        "SESSION ID", "TITLE", "MODEL", "ITEMS", "TURNS", "TOKENS", "UPDATED"
    );
    println!("{}", "-".repeat(120));

    for session in sessions {
        let title = session.title.as_deref().unwrap_or("-");
        let model = session.model.as_deref().unwrap_or("-");
        let (item_count, turns, tokens) = session_stats(handle, &session.session_id);

        println!(
            "{:<36} | {:<16} | {:<14} | {:<7} | {:<7} | {:<8} | {:<12}",
            truncate_str(&session.session_id, 36),
            truncate_str(title, 16),
            truncate_str(model, 14),
            item_count,
            turns,
            tokens,
            format_timestamp_relative(session.updated_at)
        );
    }
}

fn session_stats(handle: &StorageHandle, session_id: &str) -> (String, String, String) {
    let conv = match handle.load_session(session_id) {
        Ok(conv) => conv,
        Err(_) => return ("-".to_string(), "-".to_string(), "-".to_string()),
    };

    let item_count = conv.item_count();
    let mut user_messages = 0usize;
    let mut assistant_messages = 0usize;
    let mut total_tokens: u64 = 0;

    for idx in 0..item_count {
        if let Ok(item) = conv.get_item(idx) {
            total_tokens += u64::from(item.input_tokens) + u64::from(item.output_tokens);
        }

        if conv.item_type(idx) != ItemType::Message {
            continue;
        }
        if let Ok(msg) = conv.get_message(idx) {
            match msg.role {
                MessageRole::User => user_messages += 1,
                MessageRole::Assistant => assistant_messages += 1,
                _ => {}
            }
        }
    }

    (
        item_count.to_string(),
        user_messages.min(assistant_messages).to_string(),
        total_tokens.to_string(),
    )
}

/// Format timestamp as relative time (e.g., "2 min ago", "Just now")
fn format_timestamp_relative(ms: i64) -> String {
    if ms == 0 {
        return "-".to_string();
    }

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let diff_secs = (now_ms - ms) / 1000;

    if diff_secs < 60 {
        "Just now".to_string()
    } else if diff_secs < 3600 {
        let mins = diff_secs / 60;
        format!("{} min ago", mins)
    } else if diff_secs < 86400 {
        let hours = diff_secs / 3600;
        format!("{} hr ago", hours)
    } else {
        let days = diff_secs / 86400;
        format!("{} day ago", days)
    }
}

/// Resolve a session target (UUID or prefix) to a full session ID.
pub(super) fn resolve_session_target(handle: &StorageHandle, target: &str) -> Result<String> {
    if target.chars().all(|c| c.is_ascii_digit()) {
        bail!("Error: Session IDs are UUIDs. Use a UUID or prefix from 'talu ask'.");
    }
    match handle.get_session(target) {
        Ok(s) => Ok(s.session_id),
        Err(StorageError::SessionNotFound(_)) => {
            let all_sessions = handle.list_sessions(None)?;
            let matches: Vec<_> = all_sessions
                .iter()
                .filter(|s| s.session_id.starts_with(target))
                .collect();
            match matches.len() {
                0 => bail!(
                    "Error: Session not found: {}\nUse 'talu ask' to see available sessions.",
                    target
                ),
                1 => Ok(matches[0].session_id.clone()),
                _ => bail!(
                    "Error: Ambiguous session ID '{}'. {} sessions match. Please be more specific.",
                    target,
                    matches.len()
                ),
            }
        }
        Err(e) => Err(e.into()),
    }
}

/// Show transcript of a specific session by ID.
pub(super) fn show_session_transcript(
    handle: &StorageHandle,
    session_id: &str,
    verbose: u8,
) -> Result<()> {
    match handle.get_session(session_id) {
        Ok(session) => {
            print_session_pretty(&session);
            println!("\nTRANSCRIPT");
            println!("{}", "-".repeat(80));

            match handle.load_session(session_id) {
                Ok(conv) => print_session_items(&conv, false, verbose)?,
                Err(e) => eprintln!("(Could not load session: {})", e),
            }
        }
        Err(StorageError::SessionNotFound(_)) => {
            bail!(
                "Error: Session not found: {}\nUse 'talu ask' to see available sessions.",
                session_id
            );
        }
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
