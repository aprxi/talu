//! Generic server events plane (`/v1/events`).
//!
//! This module exposes:
//! - Replay API: `GET /v1/events`
//! - Live SSE API: `GET /v1/events/stream`
//! - Capabilities API: `GET /v1/events/topics`
//!
//! Events are process-local and in-memory only.

use std::collections::{HashSet, VecDeque};
use std::convert::Infallible;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::{BodyExt, Full, StreamBody};
use hyper::body::Frame;
use hyper::body::Incoming;
use hyper::{Request, Response, StatusCode};
use serde::Serialize;
use serde_json::json;
use tokio::sync::broadcast;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use utoipa::ToSchema;

use crate::server::auth_gateway::AuthContext;
use crate::server::state::AppState;

type BoxBody = http_body_util::combinators::BoxBody<Bytes, Infallible>;

const DEFAULT_RING_CAPACITY: usize = 10_000;
const DEFAULT_BROADCAST_CAPACITY: usize = 512;
const DEFAULT_REPLAY_LIMIT: usize = 100;
const MAX_REPLAY_LIMIT: usize = 2_000;
const HEARTBEAT_SECONDS: u64 = 15;

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EventCorrelation {
    pub request_id: Option<String>,
    pub response_id: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EventEnvelope {
    pub id: String,
    pub cursor: String,
    pub ts_ms: i64,
    pub verbosity_min: u8,
    pub level: String,
    pub domain: String,
    pub topic: String,
    pub event_class: String,
    pub message: String,
    pub tenant_id: Option<String>,
    pub correlation: Option<EventCorrelation>,
    #[schema(value_type = Option<Object>)]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EventGap {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
    pub oldest_available: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EventsReplayResponse {
    pub events: Vec<EventEnvelope>,
    pub next_cursor: Option<String>,
    pub gap: Option<EventGap>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EventFieldValuesResponse {
    pub domains: Vec<String>,
    pub topics: Vec<String>,
    pub event_class: Vec<String>,
    pub filter_fields: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EventDraft {
    pub verbosity_min: u8,
    pub level: String,
    pub domain: String,
    pub topic: String,
    pub event_class: String,
    pub message: String,
    pub tenant_id: Option<String>,
    pub correlation: Option<EventCorrelation>,
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
struct EventBusSnapshot {
    events: Vec<EventEnvelope>,
    gap: Option<EventGap>,
    last_seq: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct EventBus {
    seq: Arc<AtomicU64>,
    capacity: usize,
    ring: Arc<RwLock<VecDeque<EventEnvelope>>>,
    tx: broadcast::Sender<EventEnvelope>,
}

impl EventBus {
    pub fn new(capacity: usize, broadcast_capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(broadcast_capacity);
        Self {
            seq: Arc::new(AtomicU64::new(0)),
            capacity,
            ring: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
            tx,
        }
    }

    pub fn publish(&self, draft: EventDraft) -> EventEnvelope {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        let event = EventEnvelope {
            id: format!("evt_{seq}"),
            cursor: encode_cursor(seq),
            ts_ms: now_ms(),
            verbosity_min: draft.verbosity_min,
            level: draft.level,
            domain: draft.domain,
            topic: draft.topic,
            event_class: draft.event_class,
            message: draft.message,
            tenant_id: draft.tenant_id,
            correlation: draft.correlation,
            data: draft.data,
        };

        {
            let mut ring = self.ring.write().expect("event ring poisoned");
            ring.push_back(event.clone());
            while ring.len() > self.capacity {
                let _ = ring.pop_front();
            }
        }

        let _ = self.tx.send(event.clone());
        event
    }

    pub fn subscribe(&self) -> broadcast::Receiver<EventEnvelope> {
        self.tx.subscribe()
    }

    pub fn oldest_reconnect_cursor(&self) -> Option<String> {
        let ring = self.ring.read().expect("event ring poisoned");
        ring.front()
            .and_then(|first| decode_cursor(&first.cursor).ok())
            .map(|first_seq| encode_cursor(first_seq.saturating_sub(1)))
    }

    fn snapshot(
        &self,
        filter: &EventFilter,
        tenant: Option<&AuthContext>,
        cursor: Option<u64>,
        limit: usize,
    ) -> EventBusSnapshot {
        let ring = self.ring.read().expect("event ring poisoned");

        let oldest_seq = ring
            .front()
            .and_then(|e| decode_cursor(&e.cursor).ok())
            .unwrap_or(1);
        let mut gap = None;
        let mut start_seq = cursor.map(|c| c.saturating_add(1)).unwrap_or(oldest_seq);
        if let Some(c) = cursor {
            if c.saturating_add(1) < oldest_seq {
                let oldest_available = encode_cursor(oldest_seq.saturating_sub(1));
                gap = Some(EventGap {
                    kind: "cursor_evicted".to_string(),
                    message: "Requested cursor no longer in buffer".to_string(),
                    oldest_available,
                });
                start_seq = oldest_seq;
            }
        }

        let mut events = Vec::new();
        let mut last_seq = None;
        for event in ring.iter() {
            let seq = decode_cursor(&event.cursor).unwrap_or(0);
            if seq < start_seq {
                continue;
            }
            if !filter.matches(event) {
                continue;
            }
            if !tenant_visible(event, tenant) {
                continue;
            }
            last_seq = Some(seq);
            events.push(event.clone());
            if events.len() >= limit {
                break;
            }
        }

        EventBusSnapshot {
            events,
            gap,
            last_seq,
        }
    }

    fn replay_response(
        &self,
        filter: &EventFilter,
        tenant: Option<&AuthContext>,
        cursor: Option<u64>,
        limit: usize,
    ) -> EventsReplayResponse {
        let snapshot = self.snapshot(filter, tenant, cursor, limit);
        let next_cursor = snapshot.last_seq.map(encode_cursor);
        EventsReplayResponse {
            events: snapshot.events,
            next_cursor,
            gap: snapshot.gap,
        }
    }

    fn stream_snapshot(
        &self,
        filter: &EventFilter,
        tenant: Option<&AuthContext>,
        cursor: Option<u64>,
    ) -> EventBusSnapshot {
        self.snapshot(filter, tenant, cursor, self.capacity)
    }

    pub fn reset_for_tests(&self) {
        self.seq.store(0, Ordering::Relaxed);
        let mut ring = self.ring.write().expect("event ring poisoned");
        ring.clear();
    }
}

static EVENT_BUS: OnceLock<Arc<EventBus>> = OnceLock::new();

pub fn global_event_bus() -> Arc<EventBus> {
    EVENT_BUS
        .get_or_init(|| {
            Arc::new(EventBus::new(
                DEFAULT_RING_CAPACITY,
                DEFAULT_BROADCAST_CAPACITY,
            ))
        })
        .clone()
}

pub fn maybe_reset_global_for_tests() {
    if std::env::var("TALU_EVENTS_RESET_ON_START").ok().as_deref() == Some("1") {
        global_event_bus().reset_for_tests();
    }
}

#[derive(Debug, Clone)]
struct EventFilter {
    verbosity: u8,
    domains: Option<HashSet<String>>,
    topics: Option<HashSet<String>>,
    event_class: Option<HashSet<String>>,
    response_id: Option<String>,
    session_id: Option<String>,
}

impl EventFilter {
    fn from_uri(uri: &hyper::Uri) -> Result<(Self, Option<u64>, usize), String> {
        let mut verbosity = 1u8;
        let mut domains = None;
        let mut topics = None;
        let mut event_class = None;
        let mut response_id = None;
        let mut session_id = None;
        let mut cursor = None;
        let mut limit = DEFAULT_REPLAY_LIMIT;

        if let Some(query) = uri.query() {
            for pair in query.split('&') {
                if pair.is_empty() {
                    continue;
                }
                let mut parts = pair.splitn(2, '=');
                let key = percent_decode(parts.next().unwrap_or_default());
                let value = percent_decode(parts.next().unwrap_or_default());
                match key.as_str() {
                    "verbosity" => {
                        let parsed = u8::from_str(&value)
                            .map_err(|_| "verbosity must be an integer in [1,3]".to_string())?;
                        if !(1..=3).contains(&parsed) {
                            return Err("verbosity must be in [1,3]".to_string());
                        }
                        verbosity = parsed;
                    }
                    "domains" => domains = parse_csv_filter(&value),
                    "topics" => topics = parse_csv_filter(&value),
                    "event_class" => event_class = parse_csv_filter(&value),
                    "response_id" => {
                        if !value.is_empty() {
                            response_id = Some(value);
                        }
                    }
                    "session_id" => {
                        if !value.is_empty() {
                            session_id = Some(value);
                        }
                    }
                    "cursor" => {
                        if !value.is_empty() {
                            cursor = Some(decode_cursor(&value)?);
                        }
                    }
                    "limit" => {
                        let parsed = usize::from_str(&value)
                            .map_err(|_| "limit must be a positive integer".to_string())?;
                        if parsed == 0 {
                            return Err("limit must be greater than zero".to_string());
                        }
                        limit = parsed.min(MAX_REPLAY_LIMIT);
                    }
                    _ => {}
                }
            }
        }

        Ok((
            Self {
                verbosity,
                domains,
                topics,
                event_class,
                response_id,
                session_id,
            },
            cursor,
            limit,
        ))
    }

    fn matches(&self, event: &EventEnvelope) -> bool {
        if event.verbosity_min > self.verbosity {
            return false;
        }
        if let Some(ref domains) = self.domains {
            if !domains.contains(&event.domain) {
                return false;
            }
        }
        if let Some(ref topics) = self.topics {
            if !topics.contains(&event.topic) {
                return false;
            }
        }
        if let Some(ref classes) = self.event_class {
            if !classes.contains(&event.event_class) {
                return false;
            }
        }
        if let Some(ref response_id) = self.response_id {
            let matches = event
                .correlation
                .as_ref()
                .and_then(|c| c.response_id.as_deref())
                == Some(response_id.as_str());
            if !matches {
                return false;
            }
        }
        if let Some(ref session_id) = self.session_id {
            let matches = event
                .correlation
                .as_ref()
                .and_then(|c| c.session_id.as_deref())
                == Some(session_id.as_str());
            if !matches {
                return false;
            }
        }
        true
    }
}

pub fn publish_ambient_rust_log(level: log::Level, target: &str, message: &str) {
    let level_str = level.as_str().to_ascii_lowercase();
    let verbosity_min = verbosity_min_for_level(level_str.as_str());
    let domain = domain_from_scope(target);
    let topic = format!("{domain}.log");

    global_event_bus().publish(EventDraft {
        verbosity_min,
        level: level_str,
        domain,
        topic,
        event_class: "log".to_string(),
        message: message.to_string(),
        tenant_id: None,
        correlation: None,
        data: Some(json!({ "target": target })),
    });
}

pub fn publish_ambient_core_log(record: &talu::logging::CoreLogRecord) {
    let level_str = match record.level {
        talu::logging::LogLevel::Trace => "trace",
        talu::logging::LogLevel::Debug => "debug",
        talu::logging::LogLevel::Info => "info",
        talu::logging::LogLevel::Warn => "warn",
        talu::logging::LogLevel::Error => "error",
        talu::logging::LogLevel::Fatal => "fatal",
        talu::logging::LogLevel::Off => "off",
    };
    let verbosity_min = verbosity_min_for_level(level_str);
    let domain = domain_from_scope(record.scope.as_str());
    let topic = format!("{domain}.log");

    let mut data = serde_json::Map::new();
    data.insert("scope".to_string(), json!(record.scope));
    if let Some(ref file) = record.file {
        data.insert("file".to_string(), json!(file));
    }
    if let Some(line) = record.line {
        data.insert("line".to_string(), json!(line));
    }
    if let Some(ref attrs) = record.attrs_json {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(attrs) {
            data.insert("attrs".to_string(), parsed);
        }
    }

    global_event_bus().publish(EventDraft {
        verbosity_min,
        level: level_str.to_string(),
        domain,
        topic,
        event_class: "log".to_string(),
        message: record.message.clone(),
        tenant_id: None,
        correlation: None,
        data: if data.is_empty() {
            None
        } else {
            Some(serde_json::Value::Object(data))
        },
    });
}

pub fn publish_inference_progress(
    tenant_id: Option<&str>,
    request_id: Option<&str>,
    response_id: Option<&str>,
    session_id: Option<&str>,
    phase: &str,
    current: u64,
    total: u64,
) {
    let pct = if total == 0 {
        0.0
    } else {
        (current as f64 / total as f64) * 100.0
    };
    global_event_bus().publish(EventDraft {
        verbosity_min: 1,
        level: "info".to_string(),
        domain: "inference".to_string(),
        topic: "inference.progress".to_string(),
        event_class: "progress".to_string(),
        message: format!("{phase}: {current}/{total}"),
        tenant_id: tenant_id.map(|s| s.to_string()),
        correlation: Some(EventCorrelation {
            request_id: request_id.map(|s| s.to_string()),
            response_id: response_id.map(|s| s.to_string()),
            session_id: session_id.map(|s| s.to_string()),
        }),
        data: Some(json!({
            "phase": phase,
            "current": current,
            "total": total,
            "pct": pct
        })),
    });
}

pub fn install_core_log_bridge() {
    talu::logging::set_core_log_callback(Some(publish_ambient_core_log));
}

#[utoipa::path(
    get,
    path = "/v1/events",
    tag = "Events",
    responses(
        (status = 200, description = "Replay events", body = EventsReplayResponse),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("verbosity" = Option<u8>, Query, description = "Verbosity level (1..3)"),
        ("domains" = Option<String>, Query, description = "Comma-separated domain filters"),
        ("topics" = Option<String>, Query, description = "Comma-separated topic filters"),
        ("event_class" = Option<String>, Query, description = "Comma-separated class filters"),
        ("response_id" = Option<String>, Query, description = "Correlation response ID"),
        ("session_id" = Option<String>, Query, description = "Correlation session ID"),
        ("cursor" = Option<String>, Query, description = "Exclusive replay cursor"),
        ("limit" = Option<usize>, Query, description = "Replay limit")
    )
)]
pub async fn handle_replay(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let (filter, cursor, limit) = match EventFilter::from_uri(req.uri()) {
        Ok(v) => v,
        Err(message) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", &message),
    };
    let bus = global_event_bus();
    publish_events_request_lifecycle(&bus, auth.as_ref(), "/v1/events");
    let response = bus.replay_response(&filter, auth.as_ref(), cursor, limit);
    json_response(StatusCode::OK, &response)
}

#[utoipa::path(
    get,
    path = "/v1/events/stream",
    tag = "Events",
    responses(
        (status = 200, description = "Live event stream"),
        (status = 400, description = "Invalid request", body = crate::server::http::ErrorResponse)
    ),
    params(
        ("verbosity" = Option<u8>, Query, description = "Verbosity level (1..3)"),
        ("domains" = Option<String>, Query, description = "Comma-separated domain filters"),
        ("topics" = Option<String>, Query, description = "Comma-separated topic filters"),
        ("event_class" = Option<String>, Query, description = "Comma-separated class filters"),
        ("response_id" = Option<String>, Query, description = "Correlation response ID"),
        ("session_id" = Option<String>, Query, description = "Correlation session ID"),
        ("cursor" = Option<String>, Query, description = "Exclusive replay cursor")
    )
)]
pub async fn handle_stream(
    _state: Arc<AppState>,
    req: Request<Incoming>,
    auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let (filter, cursor, _) = match EventFilter::from_uri(req.uri()) {
        Ok(v) => v,
        Err(message) => return json_error(StatusCode::BAD_REQUEST, "invalid_request", &message),
    };
    let bus = global_event_bus();
    publish_events_request_lifecycle(&bus, auth.as_ref(), "/v1/events/stream");
    let mut rx = bus.subscribe();
    let snapshot = bus.stream_snapshot(&filter, auth.as_ref(), cursor);
    let mut last_seq = snapshot.last_seq.unwrap_or(cursor.unwrap_or(0));
    let tenant = auth;

    let (tx_stream, rx_stream) = tokio::sync::mpsc::unbounded_channel::<Bytes>();
    tokio::spawn(async move {
        if let Some(gap) = snapshot.gap {
            let _ = tx_stream.send(sse_json(
                "gap",
                &json!({
                    "type": gap.kind,
                    "message": gap.message,
                    "oldest_available": gap.oldest_available
                }),
            ));
        }
        for event in snapshot.events {
            if let Ok(seq) = decode_cursor(&event.cursor) {
                last_seq = last_seq.max(seq);
            }
            let _ = tx_stream.send(sse_json(
                "event",
                &serde_json::to_value(&event).unwrap_or(json!({})),
            ));
        }

        let mut heartbeat =
            tokio::time::interval(std::time::Duration::from_secs(HEARTBEAT_SECONDS));
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    if tx_stream.send(Bytes::from_static(b": keepalive\n\n")).is_err() {
                        break;
                    }
                }
                recv = rx.recv() => {
                    match recv {
                        Ok(event) => {
                            let seq = decode_cursor(&event.cursor).unwrap_or(0);
                            if seq <= last_seq {
                                continue;
                            }
                            if !filter.matches(&event) {
                                last_seq = seq;
                                continue;
                            }
                            if !tenant_visible(&event, tenant.as_ref()) {
                                last_seq = seq;
                                continue;
                            }
                            last_seq = seq;
                            if tx_stream.send(sse_json("event", &serde_json::to_value(&event).unwrap_or(json!({})))).is_err() {
                                break;
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(_)) => {
                            let reconnect_cursor = bus
                                .oldest_reconnect_cursor()
                                .unwrap_or_else(|| encode_cursor(last_seq));
                            let _ = tx_stream.send(sse_json("error", &json!({
                                "type": "stream_lagged",
                                "message": "Consumer too slow; events were dropped",
                                "reconnect_cursor": reconnect_cursor
                            })));
                            break;
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            let _ = tx_stream.send(sse_json("done", &json!({
                                "type": "stream_closed",
                                "message": "Server shutting down"
                            })));
                            break;
                        }
                    }
                }
            }
        }
    });

    let stream = UnboundedReceiverStream::new(rx_stream)
        .map(|chunk| Ok::<_, Infallible>(Frame::data(chunk)));
    let body = StreamBody::new(stream).boxed();

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap()
}

#[utoipa::path(
    get,
    path = "/v1/events/topics",
    tag = "Events",
    responses(
        (status = 200, description = "Supported events filters and field values", body = EventFieldValuesResponse)
    )
)]
pub async fn handle_topics(
    _state: Arc<AppState>,
    _req: Request<Incoming>,
    _auth: Option<AuthContext>,
) -> Response<BoxBody> {
    let bus = global_event_bus();
    publish_events_request_lifecycle(&bus, _auth.as_ref(), "/v1/events/topics");
    json_response(
        StatusCode::OK,
        &EventFieldValuesResponse {
            domains: vec![
                "core".to_string(),
                "events".to_string(),
                "inference".to_string(),
                "repo".to_string(),
                "server".to_string(),
            ],
            topics: vec![
                "core.log".to_string(),
                "events.log".to_string(),
                "events.query".to_string(),
                "inference.log".to_string(),
                "inference.progress".to_string(),
                "server.log".to_string(),
            ],
            event_class: vec![
                "log".to_string(),
                "progress".to_string(),
                "query".to_string(),
            ],
            filter_fields: vec![
                "verbosity".to_string(),
                "domains".to_string(),
                "topics".to_string(),
                "event_class".to_string(),
                "response_id".to_string(),
                "session_id".to_string(),
                "cursor".to_string(),
                "limit".to_string(),
            ],
        },
    )
}

fn tenant_visible(event: &EventEnvelope, auth: Option<&AuthContext>) -> bool {
    match auth {
        Some(ctx) => match event.tenant_id.as_deref() {
            Some(tenant) => tenant == ctx.tenant_id.as_str(),
            None => true,
        },
        None => true,
    }
}

fn publish_events_request_lifecycle(bus: &EventBus, auth: Option<&AuthContext>, path: &str) {
    let request_id = format!("req_{}", random_id());
    bus.publish(EventDraft {
        verbosity_min: 2,
        level: "debug".to_string(),
        domain: "events".to_string(),
        topic: "events.query".to_string(),
        event_class: "query".to_string(),
        message: format!("request {path}"),
        tenant_id: auth.map(|ctx| ctx.tenant_id.clone()),
        correlation: Some(EventCorrelation {
            request_id: Some(request_id),
            response_id: None,
            session_id: None,
        }),
        data: Some(json!({ "path": path })),
    });
}

fn parse_csv_filter(value: &str) -> Option<HashSet<String>> {
    let set = value
        .split(',')
        .filter_map(|s| {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_ascii_lowercase())
            }
        })
        .collect::<HashSet<_>>();
    if set.is_empty() {
        None
    } else {
        Some(set)
    }
}

fn domain_from_scope(scope: &str) -> String {
    scope
        .split("::")
        .next()
        .unwrap_or("server")
        .to_ascii_lowercase()
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn random_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}")
}

fn verbosity_min_for_level(level: &str) -> u8 {
    match level {
        "trace" => 3,
        "debug" => 2,
        _ => 1,
    }
}

fn encode_cursor(seq: u64) -> String {
    format!("c_{seq}")
}

fn decode_cursor(value: &str) -> Result<u64, String> {
    let raw = value
        .strip_prefix("c_")
        .ok_or_else(|| "cursor must have c_<number> format".to_string())?;
    u64::from_str(raw).map_err(|_| "cursor must have c_<number> format".to_string())
}

fn percent_decode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let bytes = value.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let h = from_hex(bytes[i + 1]);
                let l = from_hex(bytes[i + 2]);
                if let (Some(h), Some(l)) = (h, l) {
                    out.push((h << 4 | l) as char);
                    i += 3;
                } else {
                    out.push('%');
                    i += 1;
                }
            }
            other => {
                out.push(other as char);
                i += 1;
            }
        }
    }
    out
}

fn from_hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn sse_json(event: &str, payload: &serde_json::Value) -> Bytes {
    let data = serde_json::to_string(payload).unwrap_or_else(|_| "{}".to_string());
    Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

fn json_response<T: Serialize>(status: StatusCode, body: &T) -> Response<BoxBody> {
    match serde_json::to_vec(body) {
        Ok(bytes) => Response::builder()
            .status(status)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(bytes)).boxed())
            .unwrap(),
        Err(_) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "Failed to serialize JSON response",
        ),
    }
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<BoxBody> {
    json_response(
        status,
        &json!({
            "error": {
                "code": code,
                "message": message
            }
        }),
    )
}
