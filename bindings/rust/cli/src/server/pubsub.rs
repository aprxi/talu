//! `/v1/collab/pubsub/ws` — lightweight collaboration relay over WebSocket.
//!
//! Clients connect, subscribe to topics, and publish messages. The server
//! relays each published message to all *other* subscribers of that topic.
//! Used for instant cross-window editor sync. This is a temporary transport
//! surface pending a core-backed collaboration API.

use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::SinkExt;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::upgrade::Upgraded;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::protocol::Role;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;
use utoipa::ToSchema;

use crate::server::code_ws;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;
type WsStream = WebSocketStream<TokioIo<Upgraded>>;

// ---------------------------------------------------------------------------
// PubSub state
// ---------------------------------------------------------------------------

pub struct PubSubState {
    clients: HashMap<u64, mpsc::UnboundedSender<String>>,
    subscriptions: HashMap<String, HashSet<u64>>,
    next_id: u64,
}

impl PubSubState {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            subscriptions: HashMap::new(),
            next_id: 1,
        }
    }

    /// Register a new client. Returns (client_id, receiver for relayed messages).
    pub fn connect(&mut self) -> (u64, mpsc::UnboundedReceiver<String>) {
        let id = self.next_id;
        self.next_id += 1;
        let (tx, rx) = mpsc::unbounded_channel();
        self.clients.insert(id, tx);
        (id, rx)
    }

    /// Remove a client and all its subscriptions.
    pub fn disconnect(&mut self, client_id: u64) {
        self.clients.remove(&client_id);
        for subscribers in self.subscriptions.values_mut() {
            subscribers.remove(&client_id);
        }
        // Clean up empty topics.
        self.subscriptions.retain(|_, subs| !subs.is_empty());
    }

    pub fn subscribe(&mut self, client_id: u64, topic: String) {
        self.subscriptions
            .entry(topic)
            .or_default()
            .insert(client_id);
    }

    pub fn unsubscribe(&mut self, client_id: u64, topic: &str) {
        if let Some(subs) = self.subscriptions.get_mut(topic) {
            subs.remove(&client_id);
            if subs.is_empty() {
                self.subscriptions.remove(topic);
            }
        }
    }

    /// Send a message to all subscribers of `topic` except `sender_id`.
    pub fn publish(&self, sender_id: u64, topic: &str, message: &str) {
        let Some(subs) = self.subscriptions.get(topic) else {
            return;
        };
        for &cid in subs {
            if cid == sender_id {
                continue;
            }
            if let Some(tx) = self.clients.get(&cid) {
                // Ignore send errors — client may have disconnected.
                let _ = tx.send(message.to_string());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Protocol types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub(crate) struct PubSubRequest {
    #[serde(rename = "type")]
    msg_type: String,
    topic: Option<String>,
    data: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct PubSubRelayMessage {
    #[serde(rename = "type")]
    msg_type: String,
    topic: String,
    data: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// WebSocket handler
// ---------------------------------------------------------------------------

#[utoipa::path(get, path = "/v1/collab/pubsub/ws", tag = "Collab::PubSub",
    responses(
        (status = 101, description = "WebSocket upgrade. Clients send subscribe/unsubscribe/publish envelopes and receive relayed message envelopes."),
        (status = 400, body = crate::server::http::ErrorResponse),
    ))]
pub async fn handle_ws(
    state: Arc<crate::server::state::AppState>,
    req: Request<Incoming>,
) -> Response<BoxBody> {
    let key = match req.headers().get("sec-websocket-key") {
        Some(value) => value.as_bytes().to_vec(),
        None => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                "Missing Sec-WebSocket-Key header",
            )
        }
    };
    let accept = code_ws::compute_accept_key(&key);

    let upgrade = hyper::upgrade::on(req);
    tokio::spawn(async move {
        match upgrade.await {
            Ok(upgraded) => {
                handle_ws_connection(state, upgraded).await;
            }
            Err(e) => {
                log::error!(target: "server::pubsub", "WebSocket upgrade failed: {e}");
            }
        }
    });

    Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header("upgrade", "websocket")
        .header("connection", "Upgrade")
        .header("sec-websocket-accept", accept)
        .body(Full::new(Bytes::new()).boxed())
        .unwrap()
}

async fn handle_ws_connection(state: Arc<crate::server::state::AppState>, upgraded: Upgraded) {
    let mut ws: WsStream =
        WebSocketStream::from_raw_socket(TokioIo::new(upgraded), Role::Server, None).await;

    // Register this client.
    let (client_id, mut rx) = {
        let mut pubsub = state.pubsub.lock().await;
        pubsub.connect()
    };

    log::debug!(target: "server::pubsub", "client {} connected", client_id);

    loop {
        tokio::select! {
            // Relay messages from other clients → this WebSocket.
            maybe_msg = rx.recv() => {
                match maybe_msg {
                    Some(text) => {
                        if ws.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    None => break, // channel closed
                }
            }
            // Incoming messages from this WebSocket.
            maybe_ws = futures_util::StreamExt::next(&mut ws) => {
                match maybe_ws {
                    Some(Ok(Message::Text(text))) => {
                        handle_client_message(&state, client_id, &text).await;
                    }
                    Some(Ok(Message::Ping(data))) => {
                        let _ = ws.send(Message::Pong(data)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {} // ignore binary, pong, etc.
                }
            }
        }
    }

    // Cleanup.
    {
        let mut pubsub = state.pubsub.lock().await;
        pubsub.disconnect(client_id);
    }
    log::debug!(target: "server::pubsub", "client {} disconnected", client_id);
}

async fn handle_client_message(
    state: &Arc<crate::server::state::AppState>,
    client_id: u64,
    text: &str,
) {
    let req: PubSubRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(_) => return, // silently ignore malformed messages
    };

    let topic = match req.topic {
        Some(ref t) if !t.is_empty() => t.as_str(),
        _ => return,
    };

    let mut pubsub = state.pubsub.lock().await;

    match req.msg_type.as_str() {
        "subscribe" => {
            pubsub.subscribe(client_id, topic.to_string());
        }
        "unsubscribe" => {
            pubsub.unsubscribe(client_id, topic);
        }
        "publish" => {
            // Build relay message.
            let relay = PubSubRelayMessage {
                msg_type: "message".to_string(),
                topic: topic.to_string(),
                data: req.data,
            };
            if let Ok(relay_json) = serde_json::to_string(&relay) {
                pubsub.publish(client_id, topic, &relay_json);
            }
        }
        _ => {} // ignore unknown types
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message,
        }
    });
    let bytes = serde_json::to_vec(&body).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(bytes)).boxed())
        .unwrap()
}
