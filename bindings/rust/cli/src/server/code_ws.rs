//! WebSocket handler for real-time code analysis.
//!
//! Each WebSocket connection owns its own parser and tree, providing
//! sub-millisecond incremental re-parsing and highlighting on every edit.
//! Resources are freed automatically when the connection closes.
//!
//! ## Protocol
//!
//! All messages are JSON. Client sends requests, server sends responses.
//!
//! ### Client → Server
//!
//! ```json
//! // Create session
//! {"type":"create","language":"python","source":"def foo(): pass"}
//!
//! // Edit: full source replacement (simpler, but sends more data)
//! {"type":"edit","source":"def foo(): return 42"}
//!
//! // Edit: delta (minimal payload, true incremental parsing)
//! // All row/column coordinates are required.
//! {"type":"edit","edits":[{"start_byte":15,"old_end_byte":19,"new_text":"return 42",
//!   "start_row":0,"start_column":15,"old_end_row":0,"old_end_column":19,
//!   "new_end_row":0,"new_end_column":24}]}
//!
//! // Highlight current tree (no re-parse)
//! {"type":"highlight"}
//! {"type":"highlight","rich":true}
//!
//! // Query: run S-expression query against the live tree
//! {"type":"query","query":"(function_definition) @fn"}
//!
//! // Graph extraction
//! {"type":"graph","mode":"callables","file_path":"src/main.py","project_root":"myproject"}
//! ```
//!
//! ### Server → Client
//!
//! ```json
//! {"type":"created","language":"python","tokens":[...]}
//! {"type":"highlight","tokens":[...]}
//! {"type":"query_result","matches":[...]}
//! {"type":"graph","data":{...}}
//! {"type":"error","code":"...","message":"..."}
//! ```

use std::ffi::CString;

use futures_util::{SinkExt, StreamExt};
use hyper::upgrade::Upgraded;
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use talu::treesitter::{InputEdit, ParserHandle, TreeHandle};

/// Per-connection state.
struct WsSession {
    parser: ParserHandle,
    tree: TreeHandle,
    language: String,
    /// Pre-allocated CString for the language identifier.
    /// Avoids per-keystroke CString allocation in highlight/query FFI calls.
    c_language: CString,
    source: Vec<u8>,
}

#[derive(Deserialize)]
struct WsRequest {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    language: String,
    #[serde(default)]
    source: String,
    #[serde(default)]
    rich: bool,
    #[serde(default)]
    edits: Vec<TextEdit>,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    file_path: String,
    #[serde(default)]
    project_root: String,
    #[serde(default)]
    definer_fqn: String,
    /// S-expression query pattern (for "query" message type).
    #[serde(default)]
    query: String,
}

/// A single text edit delta (LSP-style content change).
///
/// All row/column coordinates are required — the server does not compute them
/// from the source buffer. Editors (Monaco, CodeMirror, VSCode) always know
/// these values at edit time.
#[derive(Deserialize, Default)]
struct TextEdit {
    start_byte: u32,
    old_end_byte: u32,
    #[serde(default)]
    new_text: String,
    /// 0-indexed row of the start position.
    start_row: u32,
    /// 0-indexed byte column of the start position.
    start_column: u32,
    /// 0-indexed row of the old end position.
    old_end_row: u32,
    /// 0-indexed byte column of the old end position.
    old_end_column: u32,
    /// 0-indexed row of the new end position.
    new_end_row: u32,
    /// 0-indexed byte column of the new end position.
    new_end_column: u32,
}

type WsStream = WebSocketStream<TokioIo<Upgraded>>;

/// Handle a WebSocket connection after HTTP upgrade.
pub async fn handle_ws_connection(upgraded: Upgraded) {
    let mut ws: WsStream = WebSocketStream::from_raw_socket(
        TokioIo::new(upgraded),
        Role::Server,
        None,
    )
    .await;

    let mut session: Option<WsSession> = None;

    while let Some(msg) = ws.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(Message::Ping(data)) => {
                let _ = ws.send(Message::Pong(data)).await;
                continue;
            }
            _ => continue,
        };

        let request: WsRequest = match serde_json::from_str(&msg) {
            Ok(r) => r,
            Err(e) => {
                let _ = ws
                    .send(Message::Text(ws_error("invalid_json", &format!("{e}"))))
                    .await;
                continue;
            }
        };

        let response = match request.msg_type.as_str() {
            "create" => handle_create(&mut session, &request),
            "edit" => handle_edit(&mut session, &request),
            "highlight" => handle_highlight(&session, &request),
            "query" => handle_query(&session, &request),
            "graph" => handle_graph(&session, &request),
            other => Err(format!("Unknown message type: {other}")),
        };

        let reply = match response {
            Ok(json) => Message::Text(json),
            Err(msg) => Message::Text(ws_error("request_failed", &msg)),
        };

        if ws.send(reply).await.is_err() {
            break;
        }
    }

    log::debug!(target: "server::code_ws", "WebSocket connection closed");
}

fn handle_create(session: &mut Option<WsSession>, req: &WsRequest) -> Result<String, String> {
    if req.language.is_empty() {
        return Err("'language' is required for create".into());
    }

    let c_language = CString::new(req.language.as_str())
        .map_err(|e| format!("Invalid language string: {e}"))?;
    let parser = ParserHandle::new(&req.language).map_err(|e| format!("Parser failed: {e}"))?;
    let source = req.source.as_bytes().to_vec();
    let tree = parser.parse(&source, None).map_err(|e| format!("Parse failed: {e}"))?;

    let tokens = tree
        .highlight_with_c_lang(&source, &c_language)
        .map_err(|e| format!("Highlight failed: {e}"))?;

    let language = req.language.clone();
    *session = Some(WsSession {
        parser,
        tree,
        language: language.clone(),
        c_language,
        source,
    });

    Ok(format!(
        "{{\"type\":\"created\",\"language\":{},\"tokens\":{}}}",
        serde_json::json!(language),
        tokens,
    ))
}

fn handle_edit(session: &mut Option<WsSession>, req: &WsRequest) -> Result<String, String> {
    let sess = session.as_mut().ok_or("No active session. Send 'create' first.")?;

    if !req.edits.is_empty() {
        // Delta mode: apply edits to source buffer + tree, then re-parse.
        apply_deltas(sess, &req.edits)?;
    } else {
        // Full source replacement: apply a synthetic edit covering the whole buffer.
        let old_source = &sess.source;
        let old_len = old_source.len() as u32;
        let new_source = req.source.as_bytes();
        let new_len = new_source.len() as u32;

        let (old_end_row, old_end_col) = byte_to_point(old_source, old_len);
        let (new_end_row, new_end_col) = byte_to_point(new_source, new_len);

        sess.tree.edit(InputEdit {
            start_byte: 0,
            old_end_byte: old_len,
            new_end_byte: new_len,
            start_row: 0,
            start_column: 0,
            old_end_row,
            old_end_column: old_end_col,
            new_end_row,
            new_end_column: new_end_col,
        });

        sess.source = new_source.to_vec();
    }

    let new_tree = sess
        .parser
        .parse(&sess.source, Some(&sess.tree))
        .map_err(|e| format!("Incremental parse failed: {e}"))?;

    let tokens = if req.rich {
        new_tree.highlight_rich_with_c_lang(&sess.source, &sess.c_language)
    } else {
        new_tree.highlight_with_c_lang(&sess.source, &sess.c_language)
    }
    .map_err(|e| format!("Highlight failed: {e}"))?;

    sess.tree = new_tree;

    Ok(format!("{{\"type\":\"highlight\",\"tokens\":{}}}", tokens))
}

/// Apply delta edits to the session's source buffer and tree.
///
/// Edits are applied in reverse byte order so earlier edits don't shift
/// the byte offsets of later edits.
fn apply_deltas(sess: &mut WsSession, edits: &[TextEdit]) -> Result<(), String> {
    // Sort by start_byte descending so we can splice without offset shifts.
    let mut sorted: Vec<&TextEdit> = edits.iter().collect();
    sorted.sort_by(|a, b| b.start_byte.cmp(&a.start_byte));

    for edit in &sorted {
        let start = edit.start_byte as usize;
        let old_end = edit.old_end_byte as usize;
        let new_bytes = edit.new_text.as_bytes();
        let new_end_byte = start as u32 + new_bytes.len() as u32;

        if start > sess.source.len() || old_end > sess.source.len() {
            return Err(format!(
                "Edit out of bounds: start_byte={}, old_end_byte={}, source_len={}",
                start,
                old_end,
                sess.source.len()
            ));
        }

        // Splice the source buffer.
        sess.source.splice(start..old_end, new_bytes.iter().copied());

        // Apply tree edit for correct incremental parsing.
        // All coordinates are client-provided — no server-side O(N) scan.
        sess.tree.edit(InputEdit {
            start_byte: edit.start_byte,
            old_end_byte: edit.old_end_byte,
            new_end_byte,
            start_row: edit.start_row,
            start_column: edit.start_column,
            old_end_row: edit.old_end_row,
            old_end_column: edit.old_end_column,
            new_end_row: edit.new_end_row,
            new_end_column: edit.new_end_column,
        });
    }

    Ok(())
}

/// Compute (row, column) for a byte offset in a source buffer.
fn byte_to_point(source: &[u8], byte: u32) -> (u32, u32) {
    let byte = byte as usize;
    let slice = if byte <= source.len() {
        &source[..byte]
    } else {
        source
    };
    let mut row: u32 = 0;
    let mut last_newline: usize = 0;
    for (i, &ch) in slice.iter().enumerate() {
        if ch == b'\n' {
            row += 1;
            last_newline = i + 1;
        }
    }
    let col = (slice.len() - last_newline) as u32;
    (row, col)
}

fn handle_highlight(session: &Option<WsSession>, req: &WsRequest) -> Result<String, String> {
    let sess = session.as_ref().ok_or("No active session. Send 'create' first.")?;

    let tokens = if req.rich {
        sess.tree.highlight_rich_with_c_lang(&sess.source, &sess.c_language)
    } else {
        sess.tree.highlight_with_c_lang(&sess.source, &sess.c_language)
    }
    .map_err(|e| format!("Highlight failed: {e}"))?;

    Ok(format!("{{\"type\":\"highlight\",\"tokens\":{}}}", tokens))
}

fn handle_query(session: &Option<WsSession>, req: &WsRequest) -> Result<String, String> {
    let sess = session.as_ref().ok_or("No active session. Send 'create' first.")?;

    if req.query.is_empty() {
        return Err("'query' is required".into());
    }

    let matches = sess.tree
        .query_with_c_lang(&sess.source, &sess.c_language, &req.query)
        .map_err(|e| format!("Query failed: {e}"))?;

    Ok(format!("{{\"type\":\"query_result\",\"matches\":{}}}", matches))
}

fn handle_graph(session: &Option<WsSession>, req: &WsRequest) -> Result<String, String> {
    let sess = session.as_ref().ok_or("No active session. Send 'create' first.")?;

    let data = match req.mode.as_str() {
        "callables" => talu::treesitter::extract_callables(
            &sess.source,
            &sess.language,
            &req.file_path,
            &req.project_root,
        ),
        "call_sites" => talu::treesitter::extract_call_sites(
            &sess.source,
            &sess.language,
            &req.definer_fqn,
            &req.file_path,
            &req.project_root,
        ),
        other => return Err(format!("Unknown mode: {other}. Use \"callables\" or \"call_sites\"")),
    }
    .map_err(|e| format!("Graph extraction failed: {e}"))?;

    Ok(format!("{{\"type\":\"graph\",\"data\":{}}}", data))
}

fn ws_error(code: &str, message: &str) -> String {
    serde_json::json!({
        "type": "error",
        "code": code,
        "message": message
    })
    .to_string()
}

/// Compute the Sec-WebSocket-Accept value from the client's Sec-WebSocket-Key.
pub fn compute_accept_key(key: &[u8]) -> String {
    tokio_tungstenite::tungstenite::handshake::derive_accept_key(key)
}
