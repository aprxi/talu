use anyhow::{anyhow, bail, Result};

use talu::{SessionRecord, StorageError, StorageHandle};

use super::util::{format_date, truncate_str};
use super::{DbDeleteArgs, DbInitArgs, DbListArgs, DbOutputFormat, DbShowArgs, DbShowFormat};

pub(super) fn cmd_db_init(args: DbInitArgs) -> Result<()> {
    use std::fs;
    use std::path::PathBuf;

    let path = args.path.unwrap_or_else(|| PathBuf::from("./taludb"));

    // Check if directory already exists
    if path.exists() {
        bail!("Error: Directory already exists: {}\nUse a different path or remove the existing directory.", path.display());
    }

    // Create directory structure
    fs::create_dir_all(&path)?;

    // Handle store.key: import or generate
    let key_path = path.join("store.key");
    let key_fingerprint = if let Some(import_path) = args.import_key {
        // Import existing key
        if !import_path.exists() {
            // Clean up created directory
            let _ = fs::remove_dir_all(&path);
            bail!("Error: Key file not found: {}", import_path.display());
        }

        let key_data = fs::read(&import_path)?;
        if key_data.len() != 16 {
            let _ = fs::remove_dir_all(&path);
            bail!(
                "Error: Invalid key file. Expected 16 bytes, got {}.",
                key_data.len()
            );
        }

        fs::write(&key_path, &key_data)?;
        format_key_fingerprint(&key_data)
    } else {
        // Generate new random key using getrandom
        let mut key_data = [0u8; 16];
        getrandom::fill(&mut key_data)
            .map_err(|e| anyhow!("Failed to generate random key: {}", e))?;

        fs::write(&key_path, key_data)?;
        format_key_fingerprint(&key_data)
    };

    // Initialize empty manifest.json
    let manifest_path = path.join("manifest.json");
    let manifest = r#"{"version": 1, "segments": [], "last_compaction_ts": 0}"#;
    fs::write(&manifest_path, manifest)?;

    println!("Initialized TaluDB storage at: {}", path.display());
    println!("  Key fingerprint: {}", key_fingerprint);

    Ok(())
}

/// Format first 8 hex characters of key as fingerprint
fn format_key_fingerprint(key: &[u8]) -> String {
    key.iter()
        .take(4)
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

pub(super) fn cmd_db_list(args: DbListArgs) -> Result<()> {
    use std::path::PathBuf;

    let path = args.path.unwrap_or_else(|| PathBuf::from("./taludb"));

    let handle = match StorageHandle::open(&path) {
        Ok(h) => h,
        Err(StorageError::StorageNotFound(p)) => {
            bail!(
                "Error: Storage not found: {}\nRun 'talu db init' to create a new storage.",
                p.display()
            );
        }
        Err(e) => return Err(e.into()),
    };

    // Default limit is 50, 0 means unlimited
    let limit = args.limit.unwrap_or(50);
    let effective_limit = if limit == 0 { None } else { Some(limit) };

    let sessions = handle.list_sessions(effective_limit)?;
    let total_count = handle.session_count()?;

    match args.format {
        DbOutputFormat::Table => print_sessions_table(&sessions),
        DbOutputFormat::Json => print_sessions_json(&sessions),
        DbOutputFormat::Csv => print_sessions_csv(&sessions),
    }

    // Show truncation notice if applicable
    if let Some(l) = effective_limit {
        if total_count > l {
            eprintln!(
                "Showing {} of {} sessions. Use --limit 0 for all.",
                l, total_count
            );
        }
    }

    Ok(())
}

fn print_sessions_table(sessions: &[SessionRecord]) {
    if sessions.is_empty() {
        println!("No sessions found.");
        return;
    }

    // Print header
    println!(
        "{:<16} | {:<20} | {:<20} | {:<8} | {:<12} | {:<12}",
        "ID", "TITLE", "MODEL", "MARKER", "CREATED", "UPDATED"
    );
    println!("{}", "-".repeat(100));

    // Print each session
    for session in sessions {
        let title = session.title.as_deref().unwrap_or("-");
        let model = session.model.as_deref().unwrap_or("-");
        let marker = session.marker.as_deref().unwrap_or("-");

        println!(
            "{:<16} | {:<20} | {:<20} | {:<8} | {:<12} | {:<12}",
            truncate_str(&session.session_id, 16),
            truncate_str(title, 20),
            truncate_str(model, 20),
            truncate_str(marker, 8),
            format_timestamp_short(session.created_at),
            format_timestamp_short(session.updated_at)
        );
    }
}

fn print_sessions_json(sessions: &[SessionRecord]) {
    print!("[");
    for (i, session) in sessions.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        println!();
        print!(
            r#"  {{"session_id": "{}", "title": {}, "model": {}, "marker": {}, "created_at": {}, "updated_at": {}}}"#,
            escape_json(&session.session_id),
            json_string_or_null(session.title.as_deref()),
            json_string_or_null(session.model.as_deref()),
            json_string_or_null(session.marker.as_deref()),
            session.created_at,
            session.updated_at
        );
    }
    println!("\n]");
}

fn print_sessions_csv(sessions: &[SessionRecord]) {
    // Print header
    println!("session_id,title,model,marker,created_at,updated_at");

    // Print each session
    for session in sessions {
        println!(
            "{},{},{},{},{},{}",
            csv_escape(&session.session_id),
            csv_escape(session.title.as_deref().unwrap_or("")),
            csv_escape(session.model.as_deref().unwrap_or("")),
            csv_escape(session.marker.as_deref().unwrap_or("")),
            session.created_at,
            session.updated_at
        );
    }
}

/// Escape a string for CSV (RFC 4180)
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        // Quote the field and escape internal quotes
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Escape a string for JSON
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Format an optional string as JSON string or null
fn json_string_or_null(s: Option<&str>) -> String {
    match s {
        Some(v) => format!("\"{}\"", escape_json(v)),
        None => "null".to_string(),
    }
}

/// Format timestamp (ms since epoch) as short date string
fn format_timestamp_short(ms: i64) -> String {
    if ms == 0 {
        return "-".to_string();
    }
    format_date(ms / 1000)
}

pub(super) fn cmd_db_show(args: DbShowArgs) -> Result<()> {
    use std::path::PathBuf;

    let path = args.path.unwrap_or_else(|| PathBuf::from("./taludb"));

    let handle = match StorageHandle::open(&path) {
        Ok(h) => h,
        Err(StorageError::StorageNotFound(p)) => {
            bail!(
                "Error: Storage not found: {}\nRun 'talu db init' to create a new storage.",
                p.display()
            );
        }
        Err(e) => return Err(e.into()),
    };

    let session = match handle.get_session(&args.session_id) {
        Ok(s) => s,
        Err(StorageError::SessionNotFound(id)) => {
            bail!("Error: Session not found: {}", id);
        }
        Err(e) => return Err(e.into()),
    };

    match args.format {
        DbShowFormat::Pretty => {
            print_session_pretty(&session);

            // Load and display session items
            println!("\nTRANSCRIPT");
            println!("{}", "-".repeat(80));

            match handle.load_session(&args.session_id) {
                Ok(conv) => print_session_items(&conv, args.raw, 0)?,
                Err(e) => eprintln!("(Could not load session: {})", e),
            }
        }
        DbShowFormat::Json => {
            if args.raw {
                // Full session JSON
                match handle.load_session(&args.session_id) {
                    Ok(conv) => {
                        use talu::responses::ResponsesView;
                        let json = conv.to_responses_json(1)?; // 1 = response format
                        println!("{}", json);
                    }
                    Err(e) => bail!("Failed to load session: {}", e),
                }
            } else {
                // Just session metadata
                print_session_json(&session);
            }
        }
    }

    Ok(())
}

pub(super) fn print_session_pretty(session: &SessionRecord) {
    println!("SESSION DETAILS");
    println!("  ID:        {}", session.session_id);
    println!("  Title:     {}", session.title.as_deref().unwrap_or("-"));
    println!("  Model:     {}", session.model.as_deref().unwrap_or("-"));
    println!("  Marker:    {}", session.marker.as_deref().unwrap_or("-"));
    println!("  Created:   {}", format_timestamp_long(session.created_at));
    println!("  Updated:   {}", format_timestamp_long(session.updated_at));
}

fn print_session_json(session: &SessionRecord) {
    println!(
        r#"{{"session_id": "{}", "title": {}, "model": {}, "marker": {}, "created_at": {}, "updated_at": {}}}"#,
        escape_json(&session.session_id),
        json_string_or_null(session.title.as_deref()),
        json_string_or_null(session.model.as_deref()),
        json_string_or_null(session.marker.as_deref()),
        session.created_at,
        session.updated_at
    );
}

pub(super) fn print_session_items(
    conv: &talu::responses::ResponsesHandle,
    raw: bool,
    verbose: u8,
) -> Result<()> {
    use talu::responses::{ItemType, MessageRole, ResponsesView};

    let count = conv.item_count();
    if count == 0 {
        println!("(No items in session)");
        return Ok(());
    }

    if raw {
        // Dump full JSON for debugging
        let json = conv.to_responses_json(1)?;
        println!("{}", json);
        return Ok(());
    }

    for (index, item_result) in conv.items().enumerate() {
        let item = item_result?;

        match item.item_type {
            ItemType::Message => {
                let msg = conv.get_message(index)?;
                let meta = if verbose > 0 {
                    format_item_meta(&item)
                } else {
                    String::new()
                };
                let role_str = match msg.role {
                    MessageRole::User => format!("\x1b[1;36mUSER\x1b[0m{}", meta),
                    MessageRole::Assistant => format!("\x1b[1;37mASSISTANT\x1b[0m{}", meta),
                    MessageRole::System => format!("\x1b[2mSYSTEM\x1b[0m{}", meta),
                    MessageRole::Developer => format!("\x1b[1;33mDEVELOPER\x1b[0m{}", meta),
                    _ => format!("UNKNOWN{}", meta),
                };
                println!("\n{}", role_str);

                let text = conv.message_text(index)?;
                println!("{}", text);
            }
            ItemType::Reasoning => {
                if verbose > 0 {
                    let meta = format_item_meta(&item);
                    println!("\n\x1b[2m[THINKING]\x1b[0m{}", meta);
                    // Show reasoning content text
                    let content_count = conv.reasoning_content_count(index);
                    let mut has_text = false;
                    for part_idx in 0..content_count {
                        if let Ok(part) = conv.get_reasoning_content(index, part_idx) {
                            let text = part.data_utf8_lossy();
                            if !text.is_empty() {
                                println!("\x1b[2;3m{}\x1b[0m", text);
                                has_text = true;
                            }
                        }
                    }
                    if !has_text {
                        // Fall back to summary text
                        match conv.reasoning_summary_text(index) {
                            Ok(text) if !text.is_empty() => println!("\x1b[2;3m{}\x1b[0m", text),
                            _ => println!("\x1b[2m(reasoning content)\x1b[0m"),
                        }
                    }
                } else {
                    println!("\n\x1b[2m[THINKING]\x1b[0m");
                    println!("\x1b[2m(reasoning content)\x1b[0m");
                }
            }
            ItemType::FunctionCall => {
                let fc = conv.get_function_call(index)?;
                let meta = if verbose > 0 {
                    format_item_meta(&item)
                } else {
                    String::new()
                };
                println!("\n\x1b[1;33m[TOOL CALL: {}]\x1b[0m{}", fc.name, meta);
                if !fc.arguments.is_empty() {
                    // Pretty-print JSON arguments if valid
                    println!("\x1b[33m{}\x1b[0m", fc.arguments);
                }
            }
            ItemType::FunctionCallOutput => {
                let fco = conv.get_function_call_output(index)?;
                println!("\n\x1b[2m[TOOL RESULT]\x1b[0m");
                if let Some(text) = fco.output_text {
                    // Truncate long outputs
                    if text.len() > 500 {
                        println!("\x1b[2m{}...\x1b[0m", &text[..500]);
                        println!("\x1b[2m({} chars total)\x1b[0m", text.len());
                    } else {
                        println!("\x1b[2m{}\x1b[0m", text);
                    }
                }
            }
            ItemType::ItemReference => {
                let ir = conv.get_item_reference(index)?;
                println!("\n\x1b[2m[REF: {}]\x1b[0m", ir.id);
            }
            _ => {
                // Unknown item type - skip
            }
        }
    }

    println!();
    Ok(())
}

/// Build a dim metadata annotation for an item, e.g. " [45 tok | 1.2s]".
/// Returns empty string if no metadata is available.
fn format_item_meta(item: &talu::responses::Item) -> String {
    let mut parts: Vec<String> = Vec::new();

    let tokens = item.input_tokens + item.output_tokens;
    if tokens > 0 {
        parts.push(format!("{} tok", tokens));
    }

    if item.generation_ns > 0 {
        let secs = item.generation_ns as f64 / 1_000_000_000.0;
        if secs >= 10.0 {
            parts.push(format!("{:.0}s", secs));
        } else {
            parts.push(format!("{:.1}s", secs));
        }
    }

    if let Some(ref reason) = item.finish_reason {
        if reason != "stop" {
            parts.push(reason.clone());
        }
    }

    if parts.is_empty() {
        String::new()
    } else {
        format!(" \x1b[2m[{}]\x1b[0m", parts.join(" | "))
    }
}

/// Format timestamp as longer date string with time
fn format_timestamp_long(ms: i64) -> String {
    if ms == 0 {
        return "-".to_string();
    }
    // Use existing format_date for the date part
    let secs = ms / 1000;
    format_date(secs)
}

pub(super) fn cmd_db_delete(args: DbDeleteArgs) -> Result<()> {
    use std::io::{self, Write};
    use std::path::PathBuf;

    let path = args.path.unwrap_or_else(|| PathBuf::from("./taludb"));

    // Confirmation prompt
    if !args.force {
        eprint!(
            "Delete session '{}'? This cannot be undone. [y/N] ",
            args.session_id
        );
        io::stderr().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            eprintln!("Aborted.");
            return Ok(());
        }
    }

    // Open storage (granular locking - lock acquired per-operation, not here)
    let handle = match StorageHandle::open(&path) {
        Ok(h) => h,
        Err(StorageError::StorageNotFound(p)) => {
            bail!(
                "Error: Storage not found: {}\nRun 'talu db init' to create a new storage.",
                p.display()
            );
        }
        Err(e) => return Err(e.into()),
    };

    // Perform deletion (acquires lock atomically, waits if another write in progress)
    match handle.delete_session(&args.session_id) {
        Ok(()) => {
            eprintln!("Session '{}' deleted.", args.session_id);
            Ok(())
        }
        Err(StorageError::SessionNotFound(id)) => {
            bail!("Error: Session not found: {}", id);
        }
        Err(e) => Err(e.into()),
    }
}
